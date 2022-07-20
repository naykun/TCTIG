from cvlg import CrossVLGenerator
from utils import load_yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from cvlgdata import CVLGDataModule
import pytorch_lightning as pl

import argparse
import json
import os
import shutil

def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--num_gpus_per_node',
        type=int,
        default=3
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get('AMLT_OUTPUT_DIR', "../image_gen_runs")
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/image_coco2017_nucleus_sampling.yaml"
    )
    parser.add_argument(
        "--dist",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--deepspeed",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str
    )
    args = parser.parse_args()

    return args


def train(config, args, dry_run=False):
    model = CrossVLGenerator(config.model_config.cvlg)
    dm = CVLGDataModule(config.dataset_config.cvlg_coco2017, batch_size=args["batch_size"], num_workers=4)
    # trainer = pl.Trainer(
    #     gpus=1, 
    #     accelerator="horovod", 
    #     fast_dev_run=dry_run, 
    #     default_root_dir=args["save_dir"], 
    #     plugins=DDPPlugin(num_nodes=2, find_unused_parameters=True),
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor="cross_entropy_epoch",
        filename="ckpt-{epoch:02d}-{cross_entropy:.2f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    if args.get("dist"):
        from itp import ITPEnvironment
        trainer = pl.Trainer(
            gpus=4, 
            num_nodes=2,
            accelerator="ddp", 
            fast_dev_run=dry_run, 
            default_root_dir=args["save_dir"], 
            # plugins=[DDPPlugin(parallel_devices=list(range(4)),num_nodes=2, find_unused_parameters=True, cluster_environment=ITPEnvironment()),],
            plugins=[ITPEnvironment()],
        )
    else:
        if args.get("deepspeed"):
            from pytorch_lightning.plugins import DeepSpeedPlugin
            trainer = pl.Trainer(
                gpus=8, 
                num_nodes=1,
                accelerator="ddp", 
                fast_dev_run=dry_run, 
                default_root_dir=args["save_dir"], 
                plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
                resume_from_checkpoint=args.get("resume",None),
                precision=16,
                callbacks=[checkpoint_callback, lr_monitor]
            )
        else:
            trainer = pl.Trainer(
                gpus=8, 
                num_nodes=1,
                accelerator="ddp", 
                fast_dev_run=dry_run, 
                default_root_dir=args["save_dir"], 
                plugins=[DDPPlugin(find_unused_parameters=True,gradient_as_bucket_view=True),"ddp_sharded",],
            )
    trainer.fit(model,datamodule=dm)
    return model, dm

def test(model, dm, config, dry_run=False):
    trainer = pl.Trainer(gpus=1, fast_dev_run=dry_run)
    test_loader = dm.test_dataloader()
    result = trainer.predict(model, test_loader)

if __name__ == "__main__":
    # config_path="config/image_coco2017_nucleus_sampling.yaml"
    args = vars(parse_opt())
    config = load_yaml(args["config_path"])
    if args["mode"] == "train":
        args["save_dir"] = os.path.join(args["save_dir"], os.path.splitext(os.path.basename(args["config_path"]))[0])
        os.makedirs(args["save_dir"], exist_ok=True)
        shutil.copy(args["config_path"], args["save_dir"])
        model, dm = train(config, args)
        # test(model, dm, config)
    else:
        model = CrossVLGenerator.load_from_checkpoint(args["checkpoint_path"])
        print("model loaded")
        trainer =  pl.Trainer(gpus=1)
        dm = CVLGDataModule({}, batch_size=32)
        dm.setup("val")
        print("data loaded")
        result = trainer.predict(model, dm.test_dataloader())
        json.dump(result,open("result.json","w"))
