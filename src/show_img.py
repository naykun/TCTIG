from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024
from tqdm import tqdm
import pickle
import torch
import torchvision
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)

args = parser.parse_args()

# vae = OpenAIDiscreteVAE().cuda().eval()
vae = VQGanVAE1024()
# filename = "/home/v-kunyan/kvlb/CVLG/runs/validate_153153.res"
with torch.no_grad():
    for filename in tqdm(glob.glob(os.path.join(args.path,"*.res"))):
        image_name = filename + ".png"
        if not os.path.exists(image_name):
            res = torch.load(filename)
            pred = res["generate"].cuda()
            import ipdb; ipdb.set_trace()
            gt = res["groundtruth"].cuda()

            pred_img = vae.decode(pred)
            gt_img = vae.decode(gt)

            images = torch.cat([pred_img, gt_img], dim=0)
            grid = torchvision.utils.make_grid(images) 

            torchvision.utils.save_image(grid,image_name)
    # import ipdb; ipdb.set_trace()