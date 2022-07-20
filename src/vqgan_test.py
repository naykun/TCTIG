from vae import OpenAIDiscreteVAE, VQGanVAE
from tqdm import tqdm
import pickle
import torch
import torchvision
import argparse
import glob
import os
from utils import load_yaml
from cvlgdata import CVLGDataModule

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--prt_cfg", action="store_true")
parser.add_argument("--override", action="store_true")
args = parser.parse_args()
config_dict = {}
if args.prt_cfg:
    # config_dict = {
    #     "vqgan_model_path": "/mnt/default/pretrained_weight/vqgan/vqgan.16384.ckpt", 
    #     "vqgan_config_path": "config/vqgan/vqgan.16384.config.yml",
    #     # "addition_ckpt": "/mnt/default/pretrained_weight/vqgan/epoch10.pth"
    # }
    config_dict = {
        "vqgan_model_path": "/mnt/default/pretrained_weight/vqgan_gumbel_f8_8192/ckpts/last.ckpt", 
        "vqgan_config_path": "config/vqgan/vqgan.f8.8192.config.yml",
        # "addition_ckpt": "/mnt/default/pretrained_weight/vqgan/epoch10.pth"
    }
# vae = OpenAIDiscreteVAE().cuda().eval()
vae = VQGanVAE(**config_dict).cuda().eval()
# filename = "/home/v-kunyan/kvlb/CVLG/runs/validate_153153.res"

dataset_config = load_yaml(args.config).dataset_config.cvlg_coco2017
dm = CVLGDataModule(dataset_config, batch_size=1)
dm.setup(None)
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()
os.makedirs(args.path, exist_ok=True)
for i, item in enumerate(val_dataloader):
    import ipdb; ipdb.set_trace()
    image_name = os.path.join(args.path, f"{i}.png")
    image =  item["image"].cuda()
    image_code = vae.get_codebook_indices(image)
    recon_image = vae.decode(image_code)
    images = torch.cat([recon_image, image], dim=0)
    grid = torchvision.utils.make_grid(images)
    torchvision.utils.save_image(grid, image_name)
# with torch.no_grad():
#     for filename in tqdm(glob.glob(os.path.join(args.path,"*.res"))):
#         image_name = filename + ".png"
#         if not os.path.exists(image_name) or args.override:
#             res = torch.load(filename)
#             pred = res["generate"].cuda()
#             # pred = torch.cat([pred[:,1:],pred[:,-1:]],dim=-1)
#             # import ipdb; ipdb.set_trace()
#             gt = res["groundtruth"].cuda()

#             pred_img = vae.decode(pred[:,1:])
#             gt_img = vae.decode(gt)

#             images = torch.cat([pred_img, gt_img], dim=0)
#             grid = torchvision.utils.make_grid(images) 

#             torchvision.utils.save_image(grid,image_name)
#     # import ipdb; ipdb.set_trace()

from vae import Net as VQGanVAE336
import io
import json
import ipdb
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
import os
# from options.test_options import TestOptions
# from models.pix2pix_model import Pix2PixModel
# from data.base_dataset import get_transform, get_params
# from util import util
# from scipy.misc import imresize
from io import BytesIO

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
from torchvision.transforms.functional import to_pil_image
config = load_yaml(
    "config/image_coco2017_nucleus_sampling_12l_axial_441tk_aug_clip_text_encode_5e-4_lr.yaml")

class ControlledDataset(DynamicCVLGDataset):
    def __init__(self, *args, **kwargs):
        super(ControlledDataset, self).__init__(*args, **kwargs)
        # self.transform = T.Compose([
        #     T.Resize([336,336]),
        #     T.ToTensor(),
        #     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])

    def __getitem__(self, idx):
        try:
            anno = self.annotations[idx]
            sentences = self.aggregate_sentence(anno)
            image_filename = self.get_image_file_name(anno["image_id"])
            # import ipdb; ipdb.set_trace()
            image_filepath = os.path.join(self.image_path, image_filename)
            image = Image.open(image_filepath).convert("RGB")
        #     sentences = select_random_Ns(
        #         sentences, random.randint(1, len(sentences)))[0]
            # sample = {}
            image, sentences = self.transform(image, sentences)
            sample = self.tokenizer(sentences)
            sample["image_id"] = anno["image_id"]
            sample["text"] = anno["caption"]
            if self.dataset_type == "val":
                sample["sentences"] = sentences
                sample["text"] = " ".join(
                    [sent["utterance"] for sent in sentences])
            sample["image"] = image
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
            warnings.warn(
                f"WARNING||||Load {idx} item error, use {random_idx} instead.")
            # import pprint
            # pprint.pprint(anno)
            # pprint.pprint(sentences)
        return sample


dataset = ControlledDataset(
    config.dataset_config.cvlg_coco2017, "val")

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

vae = VQGanVAE336()
state_dict = torch.load("/mnt/default/pretrained_weight/vqgan/epoch14.pth", map_location = 'cpu')["model"]
adaptively_load_state_dict(vae, state_dict, adapt=True)
def convert_models_to_fp16(model): 
    for p in model.parameters(): 
        p.data = p.data.half() 
# convert_models_to_fp16(vae)

for idx in range(10):
    import ipdb; ipdb.set_trace()
    image = convert_batch_to_sample_list([dataset[idx]])["image"]
    code = vae.get_codebook_indices(image)
    res_image = vae.decode(code)
    dec_input = res_image.clamp(-1, 1).cpu()
    out_image = (dec_input + 1.0) / 2.0 * 255.0
    # res_image = to_pil_image(out_image[0])
    grid = out_image[0].numpy().astype(np.uint8).transpose((1, 2, 0))

    Image.fromarray(grid).save("test.png")
    # res_image.save("test.png")