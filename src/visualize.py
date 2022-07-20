from cvlg import CrossVLGenerator
from utils import load_yaml
from cvlgdata import CVLGDataModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchvision
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--cfg_path", type=str)
parser.add_argument("--gen", action="store_true")
parser.add_argument("--origin", action="store_true")
parser.add_argument("--override", action="store_true")

args = parser.parse_args()

vis_config_path = "config/visualize_256tk.yaml"
checkpoint_path = getattr(args, "ckpt_path", "/mnt/default/runs/cvlg/lightning_logs/version_1/checkpoints/epoch=983-step=516599.ckpt")
visualize_config = load_yaml(vis_config_path).model_config.cvlg
checkpoint_config = load_yaml(args.cfg_path).model_config.cvlg
checkpoint_config.inference = visualize_config.inference
dataset_config = load_yaml(args.cfg_path).dataset_config.cvlg_coco2017

model = CrossVLGenerator(checkpoint_config)
checkpoint = pl_load(checkpoint_path)
keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
print(keys)
model = model.cuda()
model.eval()
model.freeze()
# model = model.load_from_checkpoint(checkpoint_path, strict=False)
image_savepath = os.path.join("/mnt/default/runs/vis",*checkpoint_path.split("/")[-3:])
os.makedirs(image_savepath, exist_ok=True)
dm = CVLGDataModule(dataset_config, batch_size=1)
dm.setup(None)
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()
for i, item in enumerate(train_dataloader):
    # import ipdb; ipdb.set_trace()
    images =  model(item.to(model.device))
    grid = torchvision.utils.make_grid(images) 
    # import ipdb; ipdb.set_trace()
    torchvision.utils.save_image(grid,os.path.join(image_savepath, "train_"+str(i)+".png"))
    if i > 10:
        break
for i, item in enumerate(val_dataloader):
    images =  model(item.to(model.device))
    grid = torchvision.utils.make_grid(images) 
    # import ipdb; ipdb.set_trace()
    torchvision.utils.save_image(grid,os.path.join(image_savepath,"val_"+str(i)+".png"))
    if i > 10:
        break