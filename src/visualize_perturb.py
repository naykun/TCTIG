from cvlg import CrossVLGenerator
from utils import load_yaml
from cvlgdata import CVLGDataModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchvision
import torch
import copy

def trace_perturbation(item):
    item = copy.deepcopy(item)
    trace_boxes = item["trace_boxes"] # b * l * [x1, y1, x2, y2, area]
    # import ipdb; ipdb.set_trace()
    offsets = torch.rand_like(trace_boxes) - 0.5
    x_offset = offsets[:,0,0]
    y_offset = offsets[:,0,1]
    offsets[:,:,0] = x_offset
    offsets[:,:,1] = y_offset    
    offsets[:,:,2] = x_offset
    offsets[:,:,3] = y_offset
    offsets[:,:,4] = 0
    trace_boxes = torch.clamp(trace_boxes + offsets,0,1)
    item["trace_boxes"] = trace_boxes
    return item, x_offset.item(), y_offset.item()
    

PERT_NUM = 5    

if __name__ == "__main__":
    config_path = "config/visualize_256tk.yaml"
    # checkpoint_path = "/home/v-kunyan/kvlb/CVLG/amlt/8gpu_12layer_8bs_axial/cvlg_8gpu_8layer_8bs/lightning_logs/version_0/checkpoints/epoch=39-step=83919.ckpt"
    checkpoint_path = "../lightning_logs/version_9/version_1/checkpoints/epoch=983-step=516599.ckpt"
    visualize_config = load_yaml(config_path).model_config.cvlg
    # checkpoint_config = load_yaml("config/image_coco2017_nucleus_sampling_12l_axial.yaml").model_config.cvlg
    checkpoint_config = load_yaml("config/image_coco2017_nucleus_sampling_12l_axial_256tk.yaml")

    dataset_config = checkpoint_config.dataset_config
    checkpoint_config = checkpoint_config.model_config.cvlg
    checkpoint_config.inference = visualize_config.inference
    # import ipdb; ipdb.set_trace()
    model = CrossVLGenerator(checkpoint_config)
    checkpoint = pl_load(checkpoint_path)
    keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(keys)
    model = model.cuda()
    model.eval()
    model.freeze()
    # model = model.load_from_checkpoint(checkpoint_path, strict=False)
    dm = CVLGDataModule(dataset_config.cvlg_coco2017, batch_size=1, local=True)
    dm.setup(None)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    with torch.no_grad():
        # for i, item in enumerate(train_dataloader):
        #     images =  model(item.to(model.device))
        #     grid = torchvision.utils.make_grid(images) 
        #     # import ipdb; ipdb.set_trace()
        #     torchvision.utils.save_image(grid,checkpoint_path+"_train_"+str(i)+".png")
        #     if i > 5:
        #         break
        for i, item in enumerate(val_dataloader):
            print(item["image_id"],item["text"])
            for j in range(PERT_NUM):
                pert_item, x_offset, y_offset = trace_perturbation(item)
                images =  model(pert_item.to(model.device))
                # import ipdb; ipdb.set_trace()
                images_withbox = []
                images_cpu = images.cpu() * 255
                for img in images_cpu:
                    img_withbox = torchvision.utils.draw_bounding_boxes(img.type(torch.uint8), pert_item["trace_boxes"][0,:,:-1]*255)
                    images_withbox.append(img_withbox)
                images_withbox = torch.stack(images_withbox, dim=0).float() / 255
                img_withbox_grid = torchvision.utils.make_grid(images_withbox)
                grid = torchvision.utils.make_grid(images) 
                # import ipdb; ipdb.set_trace()
                torchvision.utils.save_image(img_withbox_grid, checkpoint_path+f"_val_{str(i)}_{str(j)}_{x_offset:.2}_{y_offset:.2}_bbox.png")                
                torchvision.utils.save_image(grid,checkpoint_path+f"_val_{str(i)}_{str(j)}_{x_offset:.2}_{y_offset:.2}.png")
            
            images =  model(item.to(model.device))
            images_withbox = []
            images_cpu = images.cpu() * 255
            for img in images_cpu:
                img_withbox = torchvision.utils.draw_bounding_boxes(img.type(torch.uint8), item["trace_boxes"][0,:,:-1]*255)
                images_withbox.append(img_withbox)
            images_withbox = torch.stack(images_withbox, dim=0).float() / 255
            img_withbox_grid = torchvision.utils.make_grid(images_withbox)
            grid = torchvision.utils.make_grid(images) 
            # import ipdb; ipdb.set_trace()
            torchvision.utils.save_image(img_withbox_grid, checkpoint_path+f"_val_{str(i)}_bbox.png")                
            torchvision.utils.save_image(grid,checkpoint_path+f"_val_{str(i)}.png")
            if i > 20:
                break