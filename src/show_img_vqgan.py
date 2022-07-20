from vae import OpenAIDiscreteVAE, VQGanVAE
from tqdm import tqdm
import pickle
import torch
import torchvision
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--prt_cfg", action="store_true")
parser.add_argument("--override", action="store_true")
args = parser.parse_args()
config_dict = {}
if args.prt_cfg:
    config_dict = {
        "vqgan_model_path": "/mnt/default/pretrained_weight/vqgan/vqgan.16384.ckpt", 
        "vqgan_config_path": "config/vqgan/vqgan.16384.config.yml",
        # "addition_ckpt": "/mnt/default/pretrained_weight/vqgan/epoch20.pth"
    }
# vae = OpenAIDiscreteVAE().cuda().eval()
vae = VQGanVAE(**config_dict).cuda().eval()
# filename = "/home/v-kunyan/kvlb/CVLG/runs/validate_153153.res"
with torch.no_grad():
    for filename in tqdm(glob.glob(os.path.join(args.path,"*.res"))):
        image_name = filename + ".png"
        if not os.path.exists(image_name) or args.override:
            res = torch.load(filename)
            pred = res["generate"].cuda()
            # pred = torch.cat([pred[:,1:],pred[:,-1:]],dim=-1)
            # import ipdb; ipdb.set_trace()
            gt = res["groundtruth"].cuda()

            pred_img = vae.decode(pred[:,1:])
            gt_img = vae.decode(gt)

            images = torch.cat([pred_img, gt_img], dim=0)
            grid = torchvision.utils.make_grid(images) 

            torchvision.utils.save_image(grid,image_name)
    # import ipdb; ipdb.set_trace()