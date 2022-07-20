import torch
from torch.serialization import load
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Optional, Union, Dict, List
from sample import Sample, SampleList, convert_batch_to_sample_list
import pytorch_lightning as pl
import os
import PIL
from datasets import load_dataset, load_from_disk
from tvtransforms import *
import numpy as np
from transformers.models.auto import AutoTokenizer
import math
import random
import warnings


class DynamicCVLGDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_type, local=False):
        self.config = config
        self.dataset_type = dataset_type
        self.subset = getattr(config, "subset", "coco")
        if self.subset == "openimage":
            self.image_path = os.path.join("/mnt/default/data/LN/images/openimage",f"{dataset_type}" if dataset_type=="train" else "validation")
            self.annotation_path = os.path.join("/mnt/default/data/LN/annotation/openimage",f"{dataset_type}")
        else:
            if not local:
                self.image_path = os.path.join("/mnt/default/data/LN/images/coco",f"{dataset_type}2017")
                self.annotation_path = os.path.join("/mnt/default/data/LN/annotation",f"{dataset_type}")
                # self.annotation_path = os.path.join("/mnt/default/mmf_cache/data/datasets/localized_narratives/defaults/annotations/bbox_aligned",f"coco_{dataset_type}_localized_narratives*.jsonl")
            else:
                self.image_path = os.path.join("/home/v-kunyan/kvlb/mmf_cache/data/datasets/coco/defaults/images", f"{dataset_type}2017")
                self.annotation_path = os.path.join("/home/v-kunyan/kvlb/CVLG/data",f"{dataset_type}")
        
        self.annotations = load_from_disk(self.annotation_path)["train"] # default split train here
        self.image_size = getattr(self.config, "image_size", 256)
        # self.annotations = load_dataset("json", data_files={dataset_type: self.annotation_path})[dataset_type] # default split train here
        if self.dataset_type == "train":
            self.transform = Compose([
                RandomChoice([
                    SegmentRandomCrop(self.image_size, pad_if_needed=True),
                    SegmentRandomResizedCrop(self.image_size, scale=getattr(self.config, "crop_resize_scale", [0.7, 1])),
                    Resize(self.image_size)
                ],
                weights=[2,1,7]),
                ToTensor([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], do_norm=False)
            ])
        else:
            self.transform =  Compose([
                Resize(self.image_size),
                ToTensor([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], do_norm=False)
            ])
        self.tokenizer = SpatialTraceTokenizer(config.tokenizer)

    def get_image_file_name(self, image_id):
        if self.subset == "coco":
            return image_id.zfill(12) + ".jpg"
        elif self.subset == "openimage":
            image_name = image_id + ".jpg"
            if self.dataset_type == "train":
                return os.path.join(f"train_{image_name[0]}", image_name)
            else:
                return image_name
        
    def __getitem__(self, idx):
        try:
            anno = self.annotations[idx]
            sentences = self.aggregate_sentence(anno)
            image_filename = self.get_image_file_name(anno["image_id"])
            image_filepath = os.path.join(self.image_path, image_filename)
            image = PIL.Image.open(image_filepath).convert("RGB")
            image, sentences = self.transform(image, sentences)
            sample = self.tokenizer(sentences)
            sample["image"] = image
            sample["image_id"] = anno["image_id"]
            sample["text"] = anno["caption"]
            if self.dataset_type == "val":
                sample["sentences"] = sentences
            # import ipdb; ipdb.set_trace()
            # sample = {
            #     "image": None,
            #     "input_ids": None,
            #     "input_mask": sample.input_mask.tolist(),
            #     "segment_ids": sample.segment_ids.tolist(),
            #     "trace_boxes": sample.trace_boxes.tolist(),
            #     "trace_boxes_mask": sample.trace_boxes_mask.tolist(),
            #     "trace_boxes_seg_ids": sample.trace_boxes_seg_id.tolist(),
            #     "trace_boxes_loop_contrastive_seg_id": sample.trace_boxes_loop_contrastive_seg_id.tolist(),
            #     "image_id":sample.feature_path.split(".")[0],
            #     "text": sample.text
            # }
            sample = Sample(sample)
        except Exception as e:
            # raise e
            random_idx = random.randint(0, len(self))
            sample = self.__getitem__(random_idx)
            warnings.warn(f"WARNING||||Load {idx} item error, use {random_idx} instead.")
            # import pprint
            # pprint.pprint(anno)
            # pprint.pprint(sentences)
        return sample


    def __len__(self):
        return len(self.annotations)
    
    def aggregate_sentence(self, anno):
        timed_caption = anno["timed_caption"]
        # import ipdb; ipdb.set_trace()
        # corner case
        if not timed_caption[-1]["utterance"].endswith("."):
            timed_caption[-1]["utterance"] += "."
            
        sentences = []
        sentence = None
        for i, word in enumerate(timed_caption):
            utterance = word["utterance"]
            if sentence is None:
                sentence = dict()
                sentence["utterance"] = []
                sentence["traces"] = []
                sentence["start_time"] = word["start_time"]
            if "." in utterance:
                if utterance.endswith("."):
                    sentence["utterance"].append(utterance)
                    sentence["traces"].extend(word["traces"])
                    sentence["utterance"] = " ".join(sentence["utterance"])
                    sentence["end_time"] = word["end_time"]
                    sentences.append(sentence)
                    sentence = None
                else:
                    # assert len(utterance.split(".")) == 2, utterance
                    idx = utterance.find(".")
                    prefix = utterance[:idx+1]
                    postfix = utterance[idx+1:]
                    break_ratio = len(prefix) / len(utterance)
                    break_position = math.floor(break_ratio * len(word["traces"]))
                    end_time = word["start_time"] + break_ratio * ( word["end_time"] - word["end_time"] )
                    sentence["utterance"].append(prefix)
                    sentence["traces"].extend(word["traces"][:break_position])
                    sentence["utterance"] = " ".join(sentence["utterance"])
                    sentence["end_time"] = end_time
                    sentences.append(sentence)
                    sentence = {
                        "utterance": [ postfix ],
                        "traces": word["traces"][break_position:],
                        "start_time": end_time
                    }
            else:
                sentence["utterance"].append(utterance)
                sentence["traces"].extend(word["traces"])
        
        sentences = list(filter(lambda x: len(x["traces"])>0, sentences))

        return sentences
                
                
                
        
        


NONINT_KEYS = {"idx","trace_boxes","image_id","text"}
class CVLGDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_type, local=False) -> None:
        super().__init__()
        self.config = config
        if self.config.vae_type == "vqgan":
            self.path = "train_vqgan_1024.data" if dataset_type == "train" else "val_vqgan_1024.data"
        else:
            self.path = "train.data" if dataset_type == "train" else "val.data"
        # remote_dir = os.environ.get('AMLT_DATA_DIR')
        remote_dir = "/vc_data/users/t-kunyan/cvlg" if not local else "../data"
        # remote_dir = "/mnt/default/data/cvlg" if not local else "../data"
        if remote_dir:
            self.path = os.path.join(remote_dir,self.path)
        else:
            self.path = os.path.join("../data", self.path)
        self.data = pickle.load(open(self.path,"rb"))
        self.keys = list(self.data.keys())


    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]
        if not isinstance(item["trace_boxes"], torch.Tensor):
            item["trace_boxes"] = torch.tensor(item["trace_boxes"], dtype=torch.float32)
            for key in item:
                if key not in NONINT_KEYS:
                    item[key] = torch.tensor(item[key], dtype=torch.long)
        current_sample = Sample()
        current_sample.update(item)
        return current_sample

    def __len__(self):
        return len(self.keys)


DATASET_CLASS = {
    "dynamic": DynamicCVLGDataset,
    "normal": CVLGDataset,
}

class CVLGDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size: int=32, num_workers: int=2, local=False):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.local = local
        self.dataset_class = DATASET_CLASS[config.get('dataset_class', "normal")]
    def setup(self, stage: Optional[str]) -> None:
        if stage == "validate":
            self.cvlg_val = self.dataset_class(self.config, dataset_type="val", local=self.local)
        else:
            self.cvlg_train = self.dataset_class(self.config, dataset_type="train", local=self.local)
            self.cvlg_val = self.dataset_class(self.config, dataset_type="val", local=self.local)
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.cvlg_train, batch_size=self.batch_size, collate_fn=convert_batch_to_sample_list, num_workers=self.num_workers)
        # return DataLoader(self.cvlg_val, batch_size=self.batch_size, collate_fn=convert_batch_to_sample_list)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.cvlg_val, batch_size=self.batch_size, collate_fn=convert_batch_to_sample_list, num_workers=self.num_workers)
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.cvlg_val, batch_size=self.batch_size, collate_fn=convert_batch_to_sample_list, num_workers=self.num_workers)
        



class SpatialTraceTokenizer():

    _CLS_TOKEN = "[CLS]"
    _SEP_TOKEN = "[SEP]"
    _MASK_TOKEN = "[MASK]"
    _PAD_TOKEN_ID = 0

    def __init__(self, config, *args, **kwargs):
        self._max_trace_seq_length = config.max_trace_seq_length
        self._max_caption_seq_length = config.max_caption_seq_length
        self.delta = config.delta
        self.time_window = config.time_window if config.time_window else 0.4
        caption_tokenizer_config = config.caption_tokenizer_config
        if caption_tokenizer_config.type == "clip":
            self.use_clip = True
            from clip.simple_tokenizer import SimpleTokenizer
            self.caption_tokenizer = SimpleTokenizer()
            self.sot_token = self.caption_tokenizer.encoder["<|startoftext|>"]
            self.eot_token = self.caption_tokenizer.encoder["<|endoftext|>"]
        else:
            self.use_clip = False
            self.caption_tokenizer = AutoTokenizer.from_pretrained(
                caption_tokenizer_config.type, **caption_tokenizer_config.params
            )
        
    def __call__(
        self, sentences
    ):
        encoding_dict = self.trace_tokenize(sentences)
        if self.use_clip:
            encoding_dict.update(self.clip_tokenize(sentences))
        else:
            encoding_dict.update(self.caption_tokenize(sentences))

        return encoding_dict

    def clip_tokenize(self, sentences):
        tokens_list = [self.sot_token]
        segment_ids = [0]
        for sid, sentence in enumerate(sentences):
            tokens = self.caption_tokenizer.encode(sentence["utterance"])
            segment_ids += [sid] * len(tokens)
            tokens_list.extend(tokens)
        tokens_list.append(self.eot_token)
        segment_ids.append(0)
        input_mask = [1] * len(tokens_list)
        input_ids = tokens_list

        while len(input_ids) < self._max_caption_seq_length:
            input_ids.append(self.sot_token)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self._max_caption_seq_length
        assert len(input_mask) == self._max_caption_seq_length
        assert len(segment_ids) == self._max_caption_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "text": [s["utterance"] for s in sentences],
        }

    def caption_tokenize(self, sentences):
        tokens_list = []
        seg_ids = []
        for sid, sentence in enumerate(sentences):
            tokens = self.caption_tokenizer.tokenize(sentence["utterance"])
            seg_ids += [sid] * len(tokens)
            tokens_list.extend(tokens)

        return self.caption_convert_to_indices(tokens_list, seg_ids)

    def caption_convert_to_indices(self, tokens, seg_ids):
        tokens = [self._CLS_TOKEN] + tokens
        # attend_length = len(token_attends[0])
        segment_ids = [0] + seg_ids

        input_ids = self.caption_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_caption_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self._max_caption_seq_length
        assert len(input_mask) == self._max_caption_seq_length
        assert len(segment_ids) == self._max_caption_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "text": tokens,
        }

    def trace_tokenize(self, sentences):
        trace_boxes_list = []
        seg_id = []
        for sid, sentence in enumerate(sentences):
            traces = sentence["traces"]
            current_t = 0
            current_trace_window = []
            trace_boxes = []
            trace_len = len(traces)
            for i, t in enumerate(traces):
                if t["t"] > current_t or i==(trace_len - 1):
                    current_t += self.time_window
                    if len(current_trace_window) > 0 :
                        points = np.array(current_trace_window)
                        x1, y1 = points.min(axis=0) * (1 - self.delta)
                        x2, y2 = points.max(axis=0) * (1 + self.delta)
                        area = (x2 - x1) * (y2 - y1)
                        trace_boxes.append([x1, y1, x2, y2, area, t["t"]])
                        current_trace_window = []
                else:
                    current_trace_window.append([t["x"], t["y"]])
            trace_boxes_list.extend(trace_boxes)
            seg_id.extend([sid]*len(trace_boxes))
        trace_boxes = [box[:-1] for box in trace_boxes_list]
        trace_boxes, trace_boxes_mask, boxes_seg_id, contr_seg_id = self.trace_trancate(
            trace_boxes, seg_id
        )
        trace_boxes = torch.tensor(trace_boxes, dtype=torch.float)
        trace_boxes_mask = torch.tensor(trace_boxes_mask, dtype=torch.long)
        boxes_seg_id = torch.tensor(boxes_seg_id, dtype=torch.long)
        contr_seg_id = torch.tensor(contr_seg_id, dtype=torch.long)

        return {
            "trace_boxes": trace_boxes,
            "trace_boxes_mask": trace_boxes_mask,
            "trace_boxes_seg_ids": boxes_seg_id,
            "trace_boxes_loop_contrastive_seg_id": contr_seg_id,
        }

    def trace_trancate(self, boxes, seg_id):
        boxes = boxes[: self._max_trace_seq_length]
        seg_id = seg_id[: self._max_trace_seq_length]
        num_boxes = len(boxes)
        appendix = [[0.0] * 5] * (self._max_trace_seq_length - num_boxes)
        boxes += appendix
        box_mask = [1] * num_boxes + [0] * (self._max_trace_seq_length - num_boxes)
        loop_contrastive_seg_id = seg_id + [0] * (self._max_trace_seq_length - num_boxes)
        box_seg_id = [1] * self._max_trace_seq_length

        return boxes, box_mask, box_seg_id, loop_contrastive_seg_id



if __name__ == "__main__":
    # dm = CVLGDataModule({}, batch_size=32, local=True)
    # dm.setup(None)
    # loader = dm.train_dataloader()
    # # print(dm)
    # for batch in loader:
    #     import ipdb;ipdb.set_trace()
    #     print(batch)
    #     break
    from utils import load_yaml
    config = load_yaml("config/image_openimage_nucleus_sampling_12l_axial_256tk_aug_clip_text_encode.yaml")
    # ds = DynamicCVLGDataset(config.dataset_config.cvlg_coco2017,"train")
    from tqdm import tqdm
    # print(ds[690].keys())
    dm = CVLGDataModule(config.dataset_config.cvlg_coco2017, batch_size=1, num_workers=0)
    dm.setup(None)
    loader = dm.train_dataloader()
    for batch in tqdm(loader):
        import ipdb; ipdb.set_trace()
    # for i in tqdm(range(len(ds))):
    #     ds[i]
    #     pass


