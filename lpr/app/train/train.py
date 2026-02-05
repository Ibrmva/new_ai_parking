import math
import sys
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights

DEPLOY_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/lpr/ocr_model.pt"

def _annealing_cos(start, end, pct):
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out

def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    total_steps = 1000 
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)

@hydra.main(config_path='../../configs', config_name='main', version_base='1.2')
def main(config: DictConfig):

    trainer_strategy = 'auto'
    with open_dict(config):
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        cpu = config.trainer.get('accelerator') == 'cpu'
        devices = config.trainer.get('devices', 0)
        if cpu:
            config.trainer.precision = '16-mixed'
        if cpu and devices > 1:
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    model: BaseSystem = hydra.utils.instantiate(config.model)

    evaluated_model_path = Path(hydra.utils.to_absolute_path("lpr/models/lpr/evaluated_ocr_model.pt"))
    if evaluated_model_path.exists():
        print(f"Loading evaluated model from {evaluated_model_path} for fine-tuning.")
        checkpoint = torch.load(evaluated_model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint.state_dict())
    elif config.get('pretrained') is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        m.load_state_dict(get_pretrained_weights(config.pretrained))

    print("Model summary:")
    print(summarize(model, max_depth=2))

    datamodule = SceneTextDataModule(
        root_dir=config.data.root_dir,
        train_dir=config.data.train_dir,
        val_dir=config.data.val_dir,
        test_dir=config.data.test_dir,
        img_size=config.data.img_size,
        charset_train=config.data.charset_train,
        charset_test=config.data.charset_test,
        max_label_length=config.data.max_label_length,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        augment=config.data.augment,
        remove_whitespace=config.data.remove_whitespace,
        normalize_unicode=config.data.normalize_unicode,
    )

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,
        save_last=True,
        dirpath=str(DEPLOY_MODEL_PATH.parent),
        filename='best',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(HydraConfig.get().runtime.output_dir, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa],
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.get('ckpt_path'))

    best_model_path = checkpoint.best_model_path
    if best_model_path:
        print(f"Saving deployment-ready model to {DEPLOY_MODEL_PATH}")
        best_model = torch.load(best_model_path, map_location="cpu", weights_only=False)
        torch.save(best_model, DEPLOY_MODEL_PATH)


if __name__ == '__main__':
    main()
