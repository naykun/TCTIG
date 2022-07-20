from cvlg import CrossVLGenerator
from utils import load_yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from cvlgdata import CVLGDataModule
import pytorch_lightning as pl

import argparse
import json

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
    args = parser.parse_args()

    return args


def train(config, dry_run=False):
    model = CrossVLGenerator(config.model_config.cvlg)
    dm = CVLGDataModule({}, batch_size=8)
    trainer = pl.Trainer(
        gpus=3, 
        accelerator="ddp", 
        fast_dev_run=dry_run, 
        default_root_dir="./runs", 
        callbacks=[
            EarlyStopping(monitor='val_bleu4', mode="max", patience=5),
            ModelCheckpoint(monitor='val_bleu4', mode="max")
        ],
        plugins=DDPPlugin(find_unused_parameters=True),
    )
    trainer.fit(model,datamodule=dm)
    return model, dm

def test(model, dm, config, dry_run=False):
    trainer = pl.Trainer(gpus=1, fast_dev_run=dry_run)
    test_loader = dm.test_dataloader()
    result = trainer.predict(model, test_loader)

if __name__ == "__main__":
    config_path="./config/caption_coco2017.yaml"
    config = load_yaml(config_path)
    args = vars(parse_opt())
    if args["mode"] == "train":
        model, dm = train(config)
        test(model, dm, config)
    else:
        model = CrossVLGenerator.load_from_checkpoint(args["checkpoint_path"])
        print("model loaded")
        trainer =  pl.Trainer(gpus=1)
        dm = CVLGDataModule({}, batch_size=32)
        dm.setup("val")
        print("data loaded")
        result = trainer.predict(model, dm.test_dataloader())
        json.dump(result,open("result.json","w"))
