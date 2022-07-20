from cvlgdata import DynamicCVLGDataset
from utils import load_yaml
import pytorch_lightning as pl
from typing import Optional, Union, Dict, List
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from cvlg import CrossVLGenerator
from pytorch_lightning.utilities.cloud_io import load as pl_load
from sample import Sample, SampleList, convert_batch_to_sample_list
import random
import PIL
import torchvision
config = load_yaml("config/image_coco2017_nucleus_sampling_12l_axial_256tk_aug_clip_text_encode.yaml")


def select_random_Ns(lst, n):
    random.shuffle(lst)
    result = []
    for i in range(0, len(lst), n):
        result.append(lst[i:i + n])
    return result

class ControlledDataset(DynamicCVLGDataset):
    def __init__(self, *args, **kwargs):
        super(ControlledDataset, self).__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        try:
            anno = self.annotations[idx]
            sentences = self.aggregate_sentence(anno)
            image_filename = self.get_image_file_name(anno["image_id"])
            image_filepath = os.path.join(self.image_path, image_filename)
            image = PIL.Image.open(image_filepath).convert("RGB")
            sentences = select_random_Ns(sentences,random.randint(1, len(sentences)))[0]
            image, sentences = self.transform(image, sentences)
            sample = self.tokenizer(sentences)
            sample["image"] = image
            sample["image_id"] = anno["image_id"]
            sample["text"] = anno["caption"]
            if self.dataset_type == "val":
                sample["sentences"] = sentences
                sample["text"] = " ".join([sent["utterance"] for sent in sentences])
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
            raise e
            random_idx = random.randint(0, len(self))
            sample = self.__getitem__(random_idx)
            warnings.warn(f"WARNING||||Load {idx} item error, use {random_idx} instead.")
            # import pprint
            # pprint.pprint(anno)
            # pprint.pprint(sentences)
        return sample

dataset = ControlledDataset(config.dataset_config.cvlg_coco2017, "val")
class CVLGDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size: int=32, num_workers: int=2, local=False):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.local = local
        self.dataset_class = ControlledDataset
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

idx = 115

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="/mnt/default/runs/cvlg/image_coco2017_nucleus_sampling_12l_axial_256tk_aug_clip_text_encode_5e-4_lr/lightning_logs/version_6/checkpoints/ckpt-epoch=569-cross_entropy=3.41.ckpt")
parser.add_argument("--cfg_path", type=str, default="/mnt/default/runs/cvlg/image_coco2017_nucleus_sampling_12l_axial_256tk_aug_clip_text_encode_5e-4_lr/image_coco2017_nucleus_sampling_12l_axial_256tk_aug_clip_text_encode_5e-4_lr.yaml")
parser.add_argument("--gen", action="store_true")
parser.add_argument("--override", action="store_true")
parser.add_argument("--tokens", type=int, default=256)
parser.add_argument("--trecs", action="store_true")
args = parser.parse_args()

vis_config_path = f"config/visualize_{args.tokens}tk.yaml"
checkpoint_path = getattr(args, "ckpt_path", "/mnt/default/runs/cvlg/lightning_logs/version_1/checkpoints/epoch=983-step=516599.ckpt")
visualize_config = load_yaml(vis_config_path).model_config.cvlg
checkpoint_config = load_yaml(args.cfg_path).model_config.cvlg
checkpoint_config.inference = visualize_config.inference
dataset_config = load_yaml(args.cfg_path).dataset_config.cvlg_coco2017

# if not args.trecs:
model = CrossVLGenerator(checkpoint_config)
checkpoint = pl_load(checkpoint_path)
keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
print(keys)
model = model.cuda()
model.eval()
model.freeze()

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        # p.grad.data = p.grad.data.float() 
convert_models_to_fp32(model)
idx_list = [49,115,3354,6054]
for idx in idx_list:
    for i in range(5):
        data = convert_batch_to_sample_list([dataset[idx]])
        # print(data["sentences"])
        # import ipdb; ipdb.set_trace()
        print(data['text'][0])
        images, _ =  model(data.to(model.device))
        grid = torchvision.utils.make_grid(images) 
        torchvision.utils.save_image(grid,f"/mnt/default/cvlg_produce/compositions/val_2_{str(idx)}_{data['text'][0][:150]}.png")