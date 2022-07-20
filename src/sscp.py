from cvlg import CrossVLGenerator
from utils import load_yaml
from cvlgdata import CVLGDataModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchvision
from torchvision.transforms.functional import to_pil_image, five_crop, to_tensor
import argparse
import os
import torch
from math import exp
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="/mnt/default/runs/cvlg/lightning_logs/version_1/checkpoints/epoch=983-step=516599.ckpt")
parser.add_argument("--cfg_path", type=str)
parser.add_argument("--gen", action="store_true")
parser.add_argument("--origin", action="store_true")
parser.add_argument("--override", action="store_true")
parser.add_argument("--tokens", type=int, default=256)
parser.add_argument("--trecs", action="store_true")
parser.add_argument("--recon", action="store_true")
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


import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def preprocess_in_batch(img_tensor):
    img_list = []
    for img in img_tensor:
        img = to_pil_image(img.squeeze())
        img = preprocess(img)
        img_list.append(img)
    return torch.stack(img_list, dim=0)

def segment_five_crop(img_tensor, sentences, crop_size, img_size=256):
    """
    Args:
        img_tensor: (N, 3, img_size, img_size) which are N samples for a single caption
        sentences: [sentence] sentences for an single image
    """
    # import ipdb; ipdb.set_trace()
    # tl, tr, bl, br, center
    i_list = [0, img_size - crop_size, 0, img_size - crop_size, img_size // 2 - crop_size // 2]
    j_list = [0, 0, img_size - crop_size, img_size - crop_size, img_size // 2 - crop_size // 2]
    h, w = crop_size, crop_size

    croped_sentences = []
    for i, j in zip(i_list, j_list):
        processed_list = []
        for sentence in sentences:
            processed = process_sentence(sentence, img_size, img_size, i, j, h, w)
            if processed:
                processed_list.append(processed)
        croped_sentences.append(merge_caption(processed_list))

    imgs = torch.stack(five_crop(img_tensor, crop_size), dim=1).view(-1,3, crop_size, crop_size)
    return imgs, croped_sentences
        

def merge_caption(caption_list):

    utterances = [t["utterance"] for t in caption_list]

    caption = " ".join(utterances) if len(utterances) else "None"

    caption = caption if len(caption.split()) < 60 else " ".join(caption.split()[:60])
    return caption 

def process_sentence(sentence, img_width, img_height, i, j, h, w, pad_h=0, pad_w=0, threshold=0.2):
    x_series = [item["x"] for item in sentence["traces"]]
    y_series = [item["y"] for item in sentence["traces"]]
    series = torch.tensor([[item["x"], item["y"], item["t"]] for item in sentence["traces"]], dtype=torch.float).T
    
    x_lower = j / (img_width + 2 * pad_w)
    x_upper = (j + w) / (img_width + 2 * pad_w)
    y_lower = i / (img_height + 2 * pad_h)
    y_upper = (i + h) / (img_height + 2 * pad_h)

    series[0] = (series[0] * img_width + pad_w) / (img_width + 2 * pad_w)
    series[1] = (series[1] * img_height + pad_h) / (img_height + 2 * pad_h)
    
    x_mask = (series[0] > x_lower) & (series[0] < x_upper)
    y_mask = (series[1] > y_lower) & (series[1] < y_upper)
    
    mask = x_mask & y_mask
    exist_rate = mask.sum() / mask.size(0)
    
    if exist_rate > threshold:
        series = series.T[mask]
        series = series.T
        series[0] = (series[0] - x_lower) / (w / (img_width+2*pad_w))
        series[1] = (series[1] - y_lower) / (h / (img_height+2*pad_h))
        series = series.T
        traces = [{"x":item[0], "y":item[1], "t":item[2]} for item in series.tolist()]
        # sentence["aug_traces"] = series
        sentence["traces"] = traces
        return sentence



# model = model.load_from_checkpoint(checkpoint_path, strict=False)
image_savepath = os.path.join("/mnt/default/runs/vis",*checkpoint_path.split("/")[-5:])
os.makedirs(image_savepath, exist_ok=True)
dm = CVLGDataModule(dataset_config, batch_size=1)
dm.setup(None)
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()

crop_size = 100
def trecs_score():
    base_url = "/mnt/default/trecs_images/ln_coco/"
    with torch.no_grad():
        scores = []
        gt_scores = []
        sscp_scores = []
        for i, item in enumerate(val_dataloader):
            try:
                # images =  model(item.to(model.device))
                gt_image = item.image
                image_id = int(item.image_id[0])
                images = [to_tensor(Image.open(os.path.join(base_url,file))) for file in os.listdir(base_url) if file.startswith(f"{image_id}_")]
                if len(images)==0:
                    continue
                images.append(gt_image.squeeze())
                # import ipdb; ipdb.set_trace()
                text = item["text"]
                images = torch.stack(images, dim=0)
                num_samples = images.shape[0]

                # images_crops = torch.stack(five_crop(images, crop_size), dim=1).view(-1,3, crop_size, crop_size)
                # import ipdb; ipdb.set_trace()
                images_crop, texts = segment_five_crop(images, item["sentences"][0], 100)
                image_clip_input = preprocess_in_batch(images_crop).to(device)
                text_clip_input = clip.tokenize(texts).to(device)
                # image_features = clip_model.encode_image(image_clip_input)
                # text_features = clip_model.encode_text(text_clip_input)
                # import ipdb; ipdb.set_trace()
                logits_per_image, logits_per_text = clip_model(image_clip_input, text_clip_input)
                probs = logits_per_text.reshape(5, num_samples, 5).transpose(0,1)
                probs = torch.diagonal(probs, offset=0, dim1=1, dim2=2)
                # import ipdb; ipdb.set_trace()
                gt_score = probs[-1:].mean(dim=-1).mean()
                score = probs[:-1].mean(dim=-1).mean()
                # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                scores.append(score)
                gt_scores.append(gt_score)
                sscp_scores.append(exp(score)/(exp(gt_score) + exp(score)))
                # print(f"GT Score: {gt_score}, Score: {score}, SSCP: {score/(gt_score + score)}")
                print(f"{i}: Avg score: {(sum(scores) / len(scores)).item()} \n Avg GT score {(sum(gt_scores) / len(gt_scores)).item()} \n Avg SSCP score {(sum(sscp_scores) / len(sscp_scores))}")
                grid = torchvision.utils.make_grid(images) 
                # import ipdb; ipdb.set_trace()
                torchvision.utils.save_image(grid,os.path.join(image_savepath,"val_"+str(i)+"_"+text[0][:150]+".png"))
            except Exception as e:
                # raise 
                pass
if args.trecs:
    trecs_score()
elif args.recon:
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            # p.grad.data = p.grad.data.float() 
    convert_models_to_fp32(model)
    convert_models_to_fp32(clip_model)
    with torch.no_grad():
        scores = []
        gt_scores = []
        sscp_scores = []
        for i, item in enumerate(val_dataloader):
            try:
                images, _ =  model(item.to(model.device))
                num_samples = 2
                gt_gt_img = item["image"]
                # import ipdb; ipdb.set_trace()
                text = item["text"]
                images = torch.cat([images[-1:], gt_gt_img], dim=0)

                # images_crops = torch.stack(five_crop(images, crop_size), dim=1).view(-1,3, crop_size, crop_size)
                # import ipdb; ipdb.set_trace()
                images_crop, texts = segment_five_crop(images, item["sentences"][0], 100)
                image_clip_input = preprocess_in_batch(images_crop).to(device)
                text_clip_input = clip.tokenize(texts).to(device)
                # image_features = clip_model.encode_image(image_clip_input)
                # text_features = clip_model.encode_text(text_clip_input)
                # import ipdb; ipdb.set_trace()
                logits_per_image, logits_per_text = clip_model(image_clip_input, text_clip_input)
                probs = logits_per_text.reshape(5, num_samples, 5).transpose(0,1)
                probs = torch.diagonal(probs, offset=0, dim1=1, dim2=2)
                # import ipdb; ipdb.set_trace()
                gt_score = probs[-1:].mean(dim=-1).mean()
                score = probs[:-1].mean(dim=-1).mean()
                # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                scores.append(score)
                gt_scores.append(gt_score)
                sscp_scores.append(exp(score)/(exp(gt_score) + exp(score)))
                # print(f"GT Score: {gt_score}, Score: {score}, SSCP: {score/(gt_score + score)}")
                print(f"{i}: Avg score: {(sum(scores) / len(scores)).item()} \n Avg GT score {(sum(gt_scores) / len(gt_scores)).item()} \n Avg SSCP score {(sum(sscp_scores) / len(sscp_scores))}")
                grid = torchvision.utils.make_grid(images) 
                # import ipdb; ipdb.set_trace()
                # torchvision.utils.save_image(grid,os.path.join(image_savepath,"val_2_"+str(i)+"_"+text[0][:150]+".png"))
            except Exception as e:
                raise e
                pass
        # if i > 1:
        #     break    
else:
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            # p.grad.data = p.grad.data.float() 
    convert_models_to_fp32(model)
    convert_models_to_fp32(clip_model)
    with torch.no_grad():
        scores = []
        gt_scores = []
        sscp_scores = []
        for i, item in enumerate(val_dataloader):
            try:
                images, _ =  model(item.to(model.device))
                num_samples = images.shape[0]
                text = item["text"]

                # images_crops = torch.stack(five_crop(images, crop_size), dim=1).view(-1,3, crop_size, crop_size)
                # import ipdb; ipdb.set_trace()
                images_crop, texts = segment_five_crop(images, item["sentences"][0], 100)
                image_clip_input = preprocess_in_batch(images_crop).to(device)
                text_clip_input = clip.tokenize(texts).to(device)
                # image_features = clip_model.encode_image(image_clip_input)
                # text_features = clip_model.encode_text(text_clip_input)
                # import ipdb; ipdb.set_trace()
                logits_per_image, logits_per_text = clip_model(image_clip_input, text_clip_input)
                probs = logits_per_text.reshape(5, num_samples, 5).transpose(0,1)
                probs = torch.diagonal(probs, offset=0, dim1=1, dim2=2)
                # import ipdb; ipdb.set_trace()
                gt_score = probs[-1:].mean(dim=-1).mean()
                score = probs[:-1].mean(dim=-1).mean()
                # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                scores.append(score)
                gt_scores.append(gt_score)
                sscp_scores.append(exp(score)/(exp(gt_score) + exp(score)))
                # print(f"GT Score: {gt_score}, Score: {score}, SSCP: {score/(gt_score + score)}")
                print(f"{i}: Avg score: {(sum(scores) / len(scores)).item()} \n Avg GT score {(sum(gt_scores) / len(gt_scores)).item()} \n Avg SSCP score {(sum(sscp_scores) / len(sscp_scores))}")
                grid = torchvision.utils.make_grid(images) 
                # import ipdb; ipdb.set_trace()
                torchvision.utils.save_image(grid,os.path.join(image_savepath,"val_2_"+str(i)+"_"+text[0][:150]+".png"))
            except Exception as e:
                # raise e
                pass
        # if i > 1:
        #     break
