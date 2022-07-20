# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torchvision
import numpy as np
import os
from torch import nn
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from axial_positional_embedding import AxialPositionalEmbedding
from vae import OpenAIDiscreteVAE
from vae import VQGanVAE
from vae import Net as VQGanVAE336
import pytorch_lightning as pl

from loss import CaptionCrossEntropyLoss, ImageCrossEntropyLoss
from metric import TracedCaptionBleu4Metric
from utils import byte_tensor_to_object
from pytorch_lightning.utilities import rank_zero_only



def adaptively_load_state_dict(target, state_dict, adapt=True):

    if adapt:
        target_dict = target.state_dict()
        # common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

        if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
                target.state_dict()['param_groups'][0]['params']:
            print('Detected mismatch params, auto adapte state_dict to current')
            common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
        target_dict.update(common_dict)
        target.load_state_dict(target_dict)
        missing_keys = [k for k in target_dict.keys() if k not in common_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

        if len(unexpected_keys) != 0:
            print(
                f"Some weights of state_dict were not used in target: {unexpected_keys}"
            )
        if len(missing_keys) != 0:
            print(
                f"Some weights of state_dict are missing used in target {missing_keys}"
            )
        if len(unexpected_keys) == 0 and len(missing_keys) == 0:
            print("Strictly Loaded state_dict.")
    else:
        target.load_state_dict(state_dict)


class CrossVLGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/cvlg/defaults.yaml"

    def build(self):
        if self.config.concate_trace:
            self.trace_feature_module = TraceBoxEncoder(
                self.config.trace_feature_encoder)

        if self.config.vae_type == "vqgan":
            self.vae = VQGanVAE(vqgan_model_path=getattr(self.config, "vae_model", None), 
                vqgan_config_path=getattr(self.config, "vae_cfg", None),
                use_deepspeed=self.config.deepspeed, 
                addition_ckpt=getattr(self.config, "vae_ckpt", None)
                )
            self.num_tokens = self.vae.num_tokens
            self.BOS_ID = self.num_tokens
            self.image_size = self.vae.image_size
            image_fmap_size = self.image_size // (2 ** self.vae.num_layers)
            self.image_seq_len = image_fmap_size ** 2
        elif self.config.vae_type == "vqgan336":
            self.vae = VQGanVAE336()
            state_dict = torch.load(getattr(self.config, "vae_ckpt", None), map_location = 'cpu')["model"]
            adaptively_load_state_dict(self.vae, state_dict, adapt=True)
            for n, p in self.vae.named_parameters():
                p.requires_grad = False 
            self.num_tokens = 12288
            self.BOS_ID = self.num_tokens
            self.image_size = 336
            image_fmap_size = 21
            self.image_seq_len = image_fmap_size ** 2
        else:
            self.vae = OpenAIDiscreteVAE()
            self.num_tokens = 8192
            self.BOS_ID = self.num_tokens
            image_code_dim = 768
            self.num_layers = 3
            self.image_size = 256
            image_fmap_size = self.image_size // (2 ** self.num_layers)
            self.image_seq_len = image_fmap_size ** 2

        self.task = "caption"
        # backbone setting
        if self.config.base_model_name == "bert-base-uncased":
            self.encoderdecoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
                "bert-base-uncased", "bert-base-uncased"
            )
        elif self.config.base_model_name == "2layer-base":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.max_position_embeddings = 1090
            config_encoder.num_hidden_layers = 2
            config_decoder.num_hidden_layers = 2
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
        elif self.config.base_model_name == "3layer-base":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.num_hidden_layers = 3
            config_decoder.num_hidden_layers = 3
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
        elif self.config.base_model_name == "t2i-2layer":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.max_position_embeddings = 500
            config_decoder.max_position_embeddings = 1050
            config_decoder.vocab_size = 8192
            config_encoder.num_hidden_layers = 2
            config_decoder.num_hidden_layers = 2
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
            self.task = "image_gen"
        elif self.config.base_model_name == "t2i-6layer":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.max_position_embeddings = 500
            config_decoder.max_position_embeddings = 1050
            config_decoder.vocab_size = 8192
            config_encoder.num_hidden_layers = 4
            config_decoder.num_hidden_layers = 4
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            self.encoderdecoder = EncoderDecoderModel(config=self.codec_config)
            self.task = "image_gen"
        elif self.config.base_model_name == "t2i":
            config_encoder = BertConfig()
            config_decoder = BertConfig()
            config_encoder.max_position_embeddings = 500
            config_decoder.max_position_embeddings = self.image_seq_len + 5
            config_decoder.vocab_size = self.num_tokens + 1
            config_encoder.num_hidden_layers = self.config.layers
            config_decoder.num_hidden_layers = self.config.layers
            if getattr(self.config, "clip_encoder", False):
                from clip.clip import load
                from clip.model import convert_weights
                self.clip_encoder, _ = load("ViT-B/32","cpu",jit=False)
                self.clip_proj = nn.Linear(512, 768)
                self.clip_norm = nn.LayerNorm(768)
                convert_weights(self.clip_encoder)
                config_encoder.num_hidden_layers = self.config.encoder_layers
            self.codec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            if self.config.axial_decoder:
                decoder = AutoModelForCausalLM.from_config(
                    self.codec_config.decoder)
                decoder.bert.embeddings = BertAxialEmbeddings(
                    self.codec_config.decoder, image_fmap_size)
                self.encoderdecoder = EncoderDecoderModel(
                    config=self.codec_config, decoder=decoder)
            else:
                self.encoderdecoder = EncoderDecoderModel(
                    config=self.codec_config)

            self.task = "image_gen"
        # import ipdb; ipdb.set_trace()

        if self.task == "caption":
            self.image_emb = torch.nn.Embedding(
                self.num_tokens, image_code_dim)
            self.image_pos_emb = AxialPositionalEmbedding(
                image_code_dim, axial_shape=(image_fmap_size, image_fmap_size)
            )
            self.criterion = CaptionCrossEntropyLoss()
            self.metric = TracedCaptionBleu4Metric(self.config.metric)
        else:
            # self.caption_embedding = torch.nn.Embedding(config_encoder.vocab_size, 768)
            # self.caption_pos_emb = SinusoidalPositionalEmbedding(230, 768)
            self.criterion = ImageCrossEntropyLoss()
            self.metric = None
        self.inference_cnt = 0

    def configure_optimizers(self):
        # print(self.lr)
        if self.config.deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam([p for p in self.parameters() if p.requires_grad],
                                         lr=self.hparams.optimizer.params.lr,
                                         betas=self.hparams.optimizer.params.beta,
                                         weight_decay=self.hparams.optimizer.params.weight_decay,)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.optimizer.params.lr)
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        self.hparams.optimizer.warmup_step),
                                                    num_training_steps=int(
                                                        self.hparams.optimizer.total_step)
                                                    )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def embedding(self, sample_list):
        if "image_code" not in sample_list:
            with torch.no_grad():
                visual_code = self.vae.get_codebook_indices(sample_list["image"])
            sample_list["image_code"] = visual_code
        else:
            visual_code = sample_list["image_code"]
        if self.task == "caption":
            visual_emb = self.image_emb(visual_code)
            visual_emb += self.image_pos_emb(visual_emb)

            decoder_input_ids = sample_list["input_ids"][:, :-1]
            other_kwargs = {}

            inputs_embeds = visual_emb
            batch_size = inputs_embeds.shape[0]
            if self.config.concate_trace:
                trace_boxes = sample_list["trace_boxes"]
                trace_boxes_mask = sample_list["trace_boxes_mask"]
                trace_feature = self.trace_feature_module(trace_boxes)
                trace_seg_id = sample_list["trace_boxes_seg_ids"]
                inputs_embeds = torch.cat(
                    (inputs_embeds, trace_feature), dim=1)
                image_feats_mask = trace_boxes_mask.new_ones(
                    (batch_size, visual_code.shape[1])
                )
                image_feats_seg_id = trace_seg_id.new_zeros(
                    (batch_size, visual_code.shape[1])
                )
                attention_mask = torch.cat(
                    (image_feats_mask, trace_boxes_mask), dim=1)
                token_type_ids = torch.cat(
                    (image_feats_seg_id, trace_seg_id), dim=1)
                position_ids = trace_seg_id.new_zeros(
                    (batch_size, attention_mask.shape[1]))
                other_kwargs.update(
                    {
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "position_ids": position_ids,
                    }
                )

        else:
            bos_tokens = visual_code.new_full([visual_code.shape[0],1], self.BOS_ID)
            decoder_input_ids = torch.cat([bos_tokens, visual_code], dim=-1)
            decoder_input_ids = decoder_input_ids[:, :-1]
            caption_ids = sample_list["input_ids"]
            if getattr(self.config, "clip_encoder", False):
                token_emb = self.clip_encoder.token_embedding(caption_ids)
                # import ipdb; ipdb.set_trace()
                # ! note clip max context length is 77
                token_emb_chunks = torch.chunk(token_emb, 3, dim=1)
                token_emb_chunked = torch.cat(token_emb_chunks, dim=0)
                x = token_emb_chunked + self.clip_encoder.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_encoder.transformer(x)
                input_embed_chunked = x.permute(1, 0, 2)  # LND -> NLD
                inputs_embeds = torch.cat(torch.chunk(input_embed_chunked, 3, dim=0), dim=1)
                inputs_embeds = self.clip_encoder.ln_final(inputs_embeds)
                inputs_embeds = self.clip_proj(inputs_embeds)
                inputs_embeds = self.clip_norm(inputs_embeds)
            else:
                inputs_embeds = self.encoderdecoder.encoder.embeddings.word_embeddings(
                    caption_ids)
            caption_mask = sample_list["input_mask"]
            batch_size = caption_ids.shape[0]
            # import ipdb;ipdb.set_trace()
            other_kwargs = {}

            if self.config.concate_trace:
                trace_boxes = sample_list["trace_boxes"] # [x1, y1, x2, y2, area]
                trace_boxes_mask = sample_list["trace_boxes_mask"]
                trace_feature = self.trace_feature_module(trace_boxes)
                trace_seg_id = sample_list["trace_boxes_seg_ids"]
                inputs_embeds = torch.cat(
                    (inputs_embeds, trace_feature), dim=1)
                caption_feats_seg_id = trace_seg_id.new_zeros(
                    (batch_size, caption_ids.shape[1])
                )
                attention_mask = torch.cat(
                    (caption_mask, trace_boxes_mask), dim=1)
                token_type_ids = torch.cat(
                    (caption_feats_seg_id, trace_seg_id), dim=1)
                # import ipdb; ipdb.set_trace()
                # position_ids = trace_seg_id.new_zeros((batch_size, attention_mask.shape[1]))
                other_kwargs.update(
                    {
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        # "position_ids": position_ids,
                    }
                )
        return decoder_input_ids, inputs_embeds, other_kwargs

    def forward(self, sample_list):
        decoder_input_ids, inputs_embeds, other_kwargs = self.embedding(
            sample_list)
        if self.config.inference.type == "beam_search":
            generate_output = self.encoderdecoder.generate(
                input_ids=None,
                input_embeds=inputs_embeds,
                bos_token_id=self.BOS_ID,
                decoder_start_token_id=self.BOS_ID,
                **self.config.inference.args,
                **other_kwargs
            )
        elif self.config.inference.type == "greedy":
            generate_output = self.encoderdecoder.generate(
                input_ids=None,
                input_embeds=inputs_embeds,
                max_length=self.config.max_gen_length,
                bos_token_id=self.BOS_ID,
                decoder_start_token_id=self.BOS_ID,
                **other_kwargs
            )
        elif self.config.inference.type == "nucleus_sampling":
            generate_output = self.encoderdecoder.generate(
                input_ids=None,
                input_embeds=inputs_embeds,
                bos_token_id=self.BOS_ID,
                decoder_start_token_id=self.BOS_ID,
                **self.config.inference.args,
                **other_kwargs
            )
        if self.task == "caption":
            model_output = {}
            if (
                "return_attention" in self.config.inference
                and self.config.inference.return_attention
            ):
                with torch.no_grad():
                    attention_temp_output = self.encoderdecoder(
                        decoder_input_ids=generate_output,
                        inputs_embeds=inputs_embeds,
                        output_attentions=True,
                        return_dict=True,
                    )
                    cross_attentions = []
                    for cross_attention in attention_temp_output["cross_attentions"]:
                        if self.config.concate_trace:
                            cross_attention = cross_attention[:, :, :, :100]
                        cross_attentions.append(cross_attention.mean(dim=1))
                    cross_attentions = (
                        torch.stack(cross_attentions).max(
                            dim=0)[0].max(dim=-1)[1]
                    )
                    model_output["cross_attention"] = cross_attentions

            model_output["captions"] = generate_output
            model_output["idxs"] = sample_list["idx"]
            model_output["image_id"] = sample_list["image_id"]
            predictions = self.format_for_prediction_caption(model_output)
            return predictions
        else:
            if self.config.vae_type == "vqgan":
                b = generate_output.shape[0]
                # import ipdb; ipdb.set_trace()
                images = self.vae.decode(generate_output[:,1:])
                gt_images = self.vae.decode(sample_list["image_code"])
            else:
                images = self.vae.decode(generate_output)
                gt_images = self.vae.decode(sample_list["image_code"])
            images = torch.cat([images, gt_images], dim=0)
            return images.cpu(), sample_list["image_id"]

    def format_for_prediction_caption(self, report):
        # import ipdb; ipdb.set_trace()
        captions = report["captions"].tolist()
        # cross_attentions = report["cross_attention"].tolist()
        predictions = []

        for idx, image_id in enumerate(report["image_id"]):
            # cross_attention = cross_attentions[idx]
            caption = self.metric.caption_processor.id2tokens(
                captions[idx]).split()
            raw_caption = self.metric.caption_processor.id2rawtoken(
                captions[idx])
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            predictions.append(
                {
                    "idx": report["idxs"][idx],
                    "image_id": image_id,
                    "caption": caption,
                    # "cross_attention": cross_attention,
                    "raw_caption": raw_caption,
                }
            )
        if predictions[-1]["idx"] > self.inference_cnt + 100:
            self.inference_cnt = predictions[-1]["idx"]
            print(predictions[-1]["image_id"])
            print(" ".join(predictions[-1]["caption"]))

        return predictions

    def training_step(self, sample_list, batch_idx):
        decoder_input_ids, inputs_embeds, other_kwargs = self.embedding(
            sample_list)
        decoder_output = self.encoderdecoder(
            decoder_input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            **other_kwargs
        )
        logits = decoder_output["logits"]
        cross_attentions = []
        for cross_attention in decoder_output["cross_attentions"]:
            if self.config.concate_trace:
                cross_attention = cross_attention[:, :, :, :100]
            cross_attentions.append(cross_attention)
        if (
            hasattr(self.config, "pretrans_attention")
            and self.config.pretrans_attention
        ):
            cross_attentions = self.attention_trans(cross_attentions)
        else:
            cross_attentions = [crs.mean(dim=1) for crs in cross_attentions]
        model_output = {}
        if self.task == "caption":
            model_output["captions"] = torch.max(logits, dim=-1)[1]
            model_output["scores"] = logits
            model_output["cross_attentions"] = cross_attentions
            sample_list["targets"] = sample_list["input_ids"][:, 1:]
            if self.config.loop_contrastive:
                cap_feat, vision_trace_feat = self.trace_caption_contrastive(
                    decoder_output["encoder_hidden_states"][-1],
                    sample_list["trace_boxes_loop_contrastive_seg_id"],
                    decoder_output["decoder_hidden_states"][-1],
                    sample_list["segment_ids"],
                )
                model_output["contrastive_a"] = cap_feat
                model_output["contrastive_b"] = vision_trace_feat
        else:
            model_output["scores"] = logits
            sample_list["targets"] = sample_list["image_code"]

        loss = self.criterion(sample_list, model_output)
        self.log("cross_entropy", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, sample_list, batch_idx):
        if self.task == "caption":
            decoder_input_ids, inputs_embeds, other_kwargs = self.embedding(
                sample_list)
            other_kwargs["remove_invalid_values"] = True
            if self.config.inference.type == "beam_search":
                generate_output = self.encoderdecoder.generate(
                    input_ids=None,
                    input_embeds=inputs_embeds,
                    bos_token_id=self.BOS_ID,
                    decoder_start_token_id=self.BOS_ID,
                    **self.config.inference.args,
                    **other_kwargs
                )
            elif self.config.inference.type == "greedy":
                generate_output = self.encoderdecoder.generate(
                    input_ids=None,
                    input_embeds=inputs_embeds,
                    max_length=self.config.max_gen_length,
                    bos_token_id=self.BOS_ID,
                    decoder_start_token_id=self.BOS_ID,
                    **other_kwargs
                )
            elif self.config.inference.type == "nucleus_sampling":
                generate_output = self.encoderdecoder.generate(
                    input_ids=None,
                    input_embeds=inputs_embeds,
                    bos_token_id=self.BOS_ID,
                    decoder_start_token_id=self.BOS_ID,
                    **self.config.inference.args,
                    **other_kwargs
                )
            model_output = {}
            model_output["captions"] = generate_output
            model_output["losses"] = {}

            blue4 = self.metric(sample_list, model_output)
            self.log("val_bleu4", blue4)
        else:
            # import ipdb; ipdb.set_trace()
            if batch_idx % 50 == 0:
                decoder_input_ids, inputs_embeds, other_kwargs = self.embedding(
                    sample_list)
                if self.config.inference.prompt:
                    input_ids = decoder_input_ids[:, :1]
                else:
                    input_ids = None
                if self.config.inference.type == "beam_search":
                    generate_output = self.encoderdecoder.generate(
                        decoder_input_ids=input_ids,
                        input_embeds=inputs_embeds,
                        bos_token_id=self.BOS_ID,
                        decoder_start_token_id=self.BOS_ID,
                        **self.config.inference.args,
                        **other_kwargs
                    )
                elif self.config.inference.type == "greedy":
                    generate_output = self.encoderdecoder.generate(
                        decoder_input_ids=input_ids,
                        input_embeds=inputs_embeds,
                        max_length=self.config.max_gen_length,
                        bos_token_id=self.BOS_ID,
                        decoder_start_token_id=self.BOS_ID,
                        **other_kwargs
                    )
                elif self.config.inference.type == "nucleus_sampling":
                    generate_output = self.encoderdecoder.generate(
                        decoder_input_ids=input_ids,
                        input_embeds=inputs_embeds,
                        bos_token_id=self.BOS_ID,
                        decoder_start_token_id=self.BOS_ID,
                        **self.config.inference.args,
                        **other_kwargs
                    )
                if self.config.deepspeed:
                    self.save_validate_result({"generate": generate_output,
                                               "groundtruth": sample_list["image_code"],
                                               "caption": str([" ".join(s)+"\n" for s in sample_list["text"]])
                                               }, os.path.join(self.logger.log_dir, "validate_{}.res".format(self.global_step)))
                    # images = self.vae.decode(generate_output)
                    # gt_images = self.vae.decode(sample_list["image_code"])
                else:
                    if self.config.vae_type == "vqgan":
                        b = generate_output.shape[0]
                        images = self.vae.decode(generate_output[:,1:])
                        gt_images = self.vae.decode(sample_list["image_code"])
                    else:
                        images = self.vae.decode(generate_output)
                        gt_images = self.vae.decode(sample_list["image_code"])
                    combine_images = gt_images.repeat_interleave(2, dim=0)
                    combine_images[::2] = images
                    grid = torchvision.utils.make_grid(combine_images)
                    tb = self.logger.experiment
                    tb.add_image("image_gen", grid, self.global_step)
                    if type(sample_list["text"][0]) == list:
                        tb.add_text("caption", str(
                            [" ".join(s)+"\n" for s in sample_list["text"]]), self.global_step)
                    else:
                        tb.add_text("caption", str(
                            [s+"\n" for s in sample_list["text"]]), self.global_step)

    @rank_zero_only
    def save_validate_result(self, obj, path):
        torch.save(obj, path)


class BertAxialEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, image_fmap_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = AxialPositionalEmbedding(
            config.hidden_size, axial_shape=(image_fmap_size, image_fmap_size))
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(inputs_embeds)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AttentionTransform(torch.nn.Module):
    def __init__(self, num_layers=2, num_heads=12, num_entries=100):
        super().__init__()
        self.linear = torch.nn.Linear(num_layers, 1)
        self.multi_head_linear = torch.nn.Linear(num_heads, 1)
        self.norm = torch.nn.LayerNorm(num_entries)

    def forward(self, attention):
        # import ipdb; ipdb.set_trace()
        attention = torch.stack(attention, dim=-1)
        attention = attention.permute(0, 2, 3, 4, 1)
        attention = self.multi_head_linear(attention).squeeze()
        attention = self.linear(attention).squeeze()
        attention = self.norm(attention)
        return attention


class TraceCaptionContrastiveModel(torch.nn.Module):
    def __init__(self, aggregate_method):
        super().__init__()
        self.vision_trace_aggregator = VisionTraceAggregator(aggregate_method)
        self.caption_aggregator = CaptionAggregator(aggregate_method)

    def forward(self, vision_trace_feat, vision_trace_mask, caption_feat, caption_mask):

        # import ipdb; ipdb.set_trace()
        # caption information aggregate
        # generate a feat list [bs, Tensor(num_sentences, feat)]
        caption_feats = self.caption_aggregator(caption_feat, caption_mask)

        # vision & trace infomation aggregate
        # generate a feat list [bs, Tensor(num_trace_segment, feat)]
        vision_trace_feats = self.vision_trace_aggregator(
            vision_trace_feat, vision_trace_mask
        )

        # in batch permutation?
        # move to loss part

        return caption_feats, vision_trace_feats


class VisionTraceAggregator(torch.nn.Module):
    def __init__(self, aggregate_method, hidden_size=768):
        super().__init__()
        self.aggregate_method = aggregate_method
        if aggregate_method == "maxpool":
            self.aggregator = lambda x: torch.max(x, dim=1)
        elif aggregate_method == "meanpool":
            self.aggregator = lambda x: torch.mean(x, dim=0)
        elif aggregate_method == "lstm":
            self.aggregator = torch.nn.LSTM(
                hidden_size, hidden_size, 2, bidirectional=True
            )
        elif aggregate_method == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8
            )
            self.aggregator = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=2
            )

        self.vt_merge = torch.nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, vision_trace_feat, vision_trace_mask):
        vision_feat = vision_trace_feat[:, :100].mean(axis=1)
        trace_feat = vision_trace_feat[:, 100:]
        trace_mask = vision_trace_mask
        max_seg_id = trace_mask.max().item()
        feat_list = []
        section_sizes = []
        # import ipdb; ipdb.set_trace()
        for seg_id in range(1, max_seg_id + 1):
            mask = trace_mask == seg_id
            section_sizes.append(mask.sum(axis=1))
        # [bs, max_seg_id] the element is the seq_length of segment
        # with seg_id in current instance
        section_sizes = torch.stack(section_sizes).t()
        batch_section_count = (section_sizes > 0).sum(axis=1)
        # [bs * num(seglen > 0)]
        section_sizes_flatten_wo_0 = section_sizes[section_sizes > 0]

        trace_feat = trace_feat[trace_mask > 0]

        # assert caption_feat.shape[0] == sum(section_sizes)
        # (num_total_sentences, Tensor(sentence_len, feat_dim))
        debatched_trace_feat = torch.split(
            trace_feat, section_sizes_flatten_wo_0.tolist()
        )
        if self.aggregate_method == "maxpool":
            # [num_total_sentences, sentence_max_len, feat_dim]
            debatched_trace_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_trace_feat, batch_first=True
            )
            feat_aggs, _ = self.aggregator(debatched_trace_feat)
        elif self.aggregate_method == "meanpool":
            feat_aggs = []
            for feat in debatched_trace_feat:
                feat_agg = self.aggregator(feat)
                feat_aggs.append(feat_agg)
            feat_aggs = torch.stack(feat_aggs)
        elif self.aggregate_method == "lstm":
            debatched_trace_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_trace_feat)
            packed_trace_feat = torch.nn.utils.rnn.pack_padded_sequence(
                debatched_trace_feat,
                section_sizes_flatten_wo_0.tolist(),
                enforce_sorted=False,
            )
            output, (h_n, c_n) = self.aggregator(packed_trace_feat)
            h_n = h_n.view(2, 2, section_sizes_flatten_wo_0.shape[0], 768)
            feat_aggs = h_n[-1].squeeze(0).mean(0)
        elif self.aggregate_method == "transformer":
            debatched_trace_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_trace_feat)
            max_seg_len, batch, _ = debatched_trace_feat.shape
            mask = torch.arange(
                0, max_seg_len, dtype=torch.long, device=trace_feat.device
            ).repeat(batch)
            mask = (
                mask < section_sizes_flatten_wo_0.repeat_interleave(
                    max_seg_len)
            ).view(batch, max_seg_len)
            mask = ~mask
            # padding mask not working
            h = self.aggregator(debatched_trace_feat,
                                None, mask).transpose(0, 1)
            feat_aggs = h.mean(axis=1)
        vision_feat_expand = []
        for v_, size in zip(vision_feat, batch_section_count.tolist()):
            vision_feat_expand.append(v_.repeat(size, 1))
        vision_feat_expand = torch.cat(vision_feat_expand)

        # import ipdb; ipdb.set_trace()
        vt_feat_aggs = torch.cat([feat_aggs, vision_feat_expand], dim=1)
        vt_feat_aggs = self.vt_merge(vt_feat_aggs)

        feat_list = torch.split(vt_feat_aggs, batch_section_count.tolist())

        return feat_list


class CaptionAggregator(torch.nn.Module):
    def __init__(self, aggregate_method, hidden_size=768):
        super().__init__()
        self.aggregate_method = aggregate_method
        if aggregate_method == "maxpool":
            self.aggregator = lambda x: torch.max(x, dim=1)
        elif aggregate_method == "meanpool":
            self.aggregator = lambda x: torch.mean(x, dim=0)
        elif aggregate_method == "lstm":
            self.aggregator = torch.nn.LSTM(
                hidden_size, hidden_size, 2, bidirectional=True
            )
        elif aggregate_method == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8
            )
            self.aggregator = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=2
            )

    def forward(self, caption_feat, caption_mask):
        # remove bos
        # import ipdb; ipdb.set_trace()
        caption_mask = caption_mask[:, 1:]
        max_seg_id = caption_mask.max().item()
        feat_list = []
        section_sizes = []
        for seg_id in range(1, max_seg_id + 1):
            mask = caption_mask == seg_id
            section_sizes.append(mask.sum(axis=1))
        # [bs, max_seg_id] the element is the seq_length of segment
        # with seg_id in current instance
        section_sizes = torch.stack(section_sizes).t()
        batch_section_count = (section_sizes > 0).sum(axis=1)
        # [bs * num(seglen > 0)]
        section_sizes_flatten_wo_0 = section_sizes[section_sizes > 0]

        caption_feat = caption_feat[caption_mask > 0]

        # assert caption_feat.shape[0] == sum(section_sizes)
        # (num_total_sentences, Tensor(sentence_len, feat_dim))
        debatched_caption_feat = torch.split(
            caption_feat, section_sizes_flatten_wo_0.tolist()
        )
        if self.aggregate_method == "maxpool":
            # [num_total_sentences, sentence_max_len, feat_dim]
            debatched_caption_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_caption_feat, batch_first=True
            )
            feat_aggs, _ = self.aggregator(debatched_caption_feat)
        elif self.aggregate_method == "meanpool":
            feat_aggs = []
            for feat in debatched_caption_feat:
                feat_agg = self.aggregator(feat)
                feat_aggs.append(feat_agg)
            feat_aggs = torch.stack(feat_aggs)
        elif self.aggregate_method == "lstm":
            debatched_caption_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_caption_feat
            )
            packed_caption_feat = torch.nn.utils.rnn.pack_padded_sequence(
                debatched_caption_feat,
                section_sizes_flatten_wo_0.tolist(),
                enforce_sorted=False,
            )
            output, (h_n, c_n) = self.aggregator(packed_caption_feat)
            h_n = h_n.view(2, 2, section_sizes_flatten_wo_0.shape[0], 768)
            feat_aggs = h_n[-1].squeeze(0).mean(0)
        elif self.aggregate_method == "transformer":
            debatched_caption_feat = torch.nn.utils.rnn.pad_sequence(
                debatched_caption_feat
            )
            max_sentence_len, batch, _ = debatched_caption_feat.shape
            mask = torch.arange(
                0, max_sentence_len, dtype=torch.long, device=caption_feat.device
            ).repeat(batch)
            mask = (
                mask < section_sizes_flatten_wo_0.repeat_interleave(
                    max_sentence_len)
            ).view(batch, max_sentence_len)
            mask = ~mask
            # padding mask not working
            h = self.aggregator(debatched_caption_feat,
                                None, mask).transpose(0, 1)
            feat_aggs = h.mean(axis=1)
        # import ipdb; ipdb.set_trace()

        feat_list = torch.split(feat_aggs, batch_section_count.tolist())
        return feat_list


class TraceBoxEncoder(torch.nn.Module):
    # @dataclass
    # class Config(Encoder.Config):
    #     name: str = "tracebox_encoder"
    #     input_size: int = 5
    #     hidden_size: int = 512
    #     num_positions: int = 64
    # Keeping this Any for now as this
    # needs a separate refactor PR.
    # embedding_params: Any = MISSING

    def __init__(self, config):
        super().__init__()
        self.linear_projection = nn.Linear(
            config.input_size, config.hidden_size)
        self.position_embedding = SinusoidalPositionalEmbedding(
            config.num_positions, config.hidden_size
        )
        self.box_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.position_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, trace_boxes):
        # import ipdb; ipdb.set_trace()
        boxes_encoding = self.box_layer_norm(
            self.linear_projection(trace_boxes))
        position_encoding = self.position_embedding(trace_boxes)
        output = self.position_layer_norm(boxes_encoding + position_encoding)
        return output


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(
                seq_len - 1
            )  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(
                seq_len, dtype=torch.long, device=self.weight.device
            )
        return super().forward(positions)


def check_contiguous(model):
    unc_cnt = 0
    for k, item in model.state_dict().items():
        if not item.is_contiguous():
            print(k)
            unc_cnt += 1
    if unc_cnt == 0:
        print("ALL CONTIGUOUS!!!")


def make_contiguous(model):
    old_state = model.state_dict()
    for k, item in old_state.items():
        if not item.is_contiguous():
            # print(item)
            old_state[k] = item.contiguous()
    for k, item in old_state.items():
        if not item.is_contiguous():
            # print(k)
            print(item)
    model.load_state_dict(old_state)
    check_contiguous(model)


if __name__ == "__main__":
    from utils import load_yaml
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.plugins import DDPPlugin
    # config = load_yaml("./config/caption_coco2017.yaml")
    # config = load_yaml("./config/image_coco2017.yaml")
    config = load_yaml("config/image_coco2017_nucleus_sampling_12l_axial_256tk.yaml")
    model = CrossVLGenerator(config.model_config.cvlg)
    # import ipdb
    # ipdb.set_trace()
    # check_contiguous(model)
    # make_contiguous(model)
    # check_contiguous(model)
    # print(model)
    from cvlgdata import CVLGDataModule
    dm = CVLGDataModule(config.dataset_config.cvlg_coco2017, batch_size=1, local=True)
    from pytorch_lightning.plugins import DeepSpeedPlugin
    trainer = pl.Trainer(fast_dev_run=True, gpus=1, )
    # trainer = pl.Trainer(gpus=3, accelerator="ddp", default_root_dir="./runs", callbacks=[EarlyStopping(monitor='val_bleu4', mode="max", patience=5)],
    #     plugins=DDPPlugin(find_unused_parameters=True),
    # )
    # trainer = pl.Trainer(gpus=3, accelerator="ddp", default_root_dir="./image_gen_runs",
    #     plugins=DDPPlugin(find_unused_parameters=True),
    # )
    # config = load_yaml("config/image_coco2017_beamsearch.yaml")
    # config = load_yaml("config/image_coco2017_nucleus_sampling.yaml")
    # # model = CrossVLGenerator.load_from_checkpoint("/home/v-kunyan/kvlb/CVLG/image_gen_runs/lightning_logs/version_1/checkpoints/epoch=25-step=145469.ckpt")
    # # import ipdb; ipdb.set_trace()
    # model.config = config.model_config.cvlg
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model,datamodule=dm)
