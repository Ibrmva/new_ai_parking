import logging
import math
import os
import shutil
import sys
from pathlib import Path

import hydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from hydra.utils import HydraConfig
from omegaconf import DictConfig, open_dict
from ray import train, tune
from ray.tune import CLIReporter, RunConfig
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.basic_variant import BasicVariantGenerator

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem

log = logging.getLogger(__name__)

class MetricTracker(tune.Stopper):
    def __init__(self, metric, max_t, patience=3, window=3):
        super().__init__()
        self.metric = metric
        self.trial_history = {}
        self.max_t = max_t
        self.training_iteration = 0
        self.eps = 0.01
        self.patience = patience
        self.kernel = self.gaussian_pdf(np.arange(window) - window // 2, sigma=0.6)
        self.buffer = 2 * (len(self.kernel) // 2) + 2

    @staticmethod
    def gaussian_pdf(x, sigma=1.0):
        return np.exp(-((x / sigma) ** 2) / 2) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def moving_average(x, k):
        return np.convolve(x, k, 'valid') / k.sum()

    def __call__(self, trial_id, result):
        self.training_iteration = result.get('epoch', 0)
        if np.isnan(result.get('loss', float('nan'))) or self.training_iteration >= self.max_t:
            try: del self.trial_history[trial_id]
            except KeyError: pass
            return True
        history = self.trial_history.get(trial_id, [])
        history = history[-(self.patience + self.buffer - 1):] + [result.get(self.metric, 0)]
        if len(history) == self.patience + self.buffer and sum(history) > 0:
            smooth_grad = np.gradient(self.moving_average(history, self.kernel))[1:-1]
            if (smooth_grad < self.eps).all():
                try: del self.trial_history[trial_id]
                except KeyError: pass
                return True
        self.trial_history[trial_id] = history
        return False

    def stop_all(self):
        return False

class TuneReportCheckpointPruneCallback(TuneReportCallback):
    def _handle(self, trainer: Trainer, pl_module: LightningModule):

        callback_metrics = trainer.callback_metrics

        if all(metric in callback_metrics for metric in self._metrics.values()):
            super()._handle(trainer, pl_module)
        trial_dir = train.get_context().get_trial_dir()

        for old in sorted(Path(trial_dir).glob('checkpoint_epoch=*-step=*'), key=os.path.getmtime)[:-1]:
            log.info(f'Deleting old checkpoint: {old}')
            shutil.rmtree(old)

def trainable(hparams, config):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    with open_dict(config):
        config.model.lr = hparams['lr']

    model: BaseSystem = hydra.utils.instantiate(config.model)
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    ckpt_dir = "/Users/gulaiymibraimova/Desktop/ai_app/parseq/outputs/parseq"
    ckpt_path = None
    if os.path.exists(ckpt_dir):
        checkpoints = sorted(Path(ckpt_dir).rglob("*.ckpt"), key=os.path.getmtime)
        if checkpoints:
            ckpt_path = str(checkpoints[-1])
            log.info(f"Using checkpoint: {ckpt_path}")

    tune_callback = TuneReportCheckpointPruneCallback({
        'loss': 'val_loss',
        'NED': 'val_NED',
        'accuracy': 'val_accuracy',
    })

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        accelerator='mps',
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=TensorBoardLogger(save_dir=tune.get_context().get_trial_dir(), name='', version='.'),
        callbacks=[tune_callback],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

@hydra.main(config_path=os.path.join(os.path.dirname(__file__), '../../configs'), config_name='tune', version_base='1.2')
def main(config: DictConfig):
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0

    with open_dict(config):
        if config.trainer.get('gpus', 0):
            config.trainer.precision = 16

        if not os.path.isabs(config.data.root_dir):
            config.data.root_dir = os.path.join(os.getcwd(), config.data.root_dir)
        else:
            config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)

    hparams = {'lr': tune.loguniform(config.tune.lr.min, config.tune.lr.max)}

    steps_per_epoch = len(hydra.utils.instantiate(config.data).train_dataloader())
    val_steps = steps_per_epoch * config.trainer.max_epochs / getattr(config.trainer, 'val_check_interval', 1)
    max_t = round(0.75 * config.trainer.max_epochs)
    warmup_t = round(config.model.warmup_pct * config.trainer.max_epochs)
    scheduler = MedianStoppingRule(time_attr='epoch', grace_period=warmup_t)

    lr = hparams['lr']
    start = np.log10(lr.lower)
    stop = np.log10(lr.upper)
    num = math.ceil(stop - start) + 1
    initial_points = [{'lr': np.clip(x, lr.lower, lr.upper).item()} for x in reversed(np.logspace(start, stop, num))]

    search_alg = BasicVariantGenerator()

    reporter = CLIReporter(parameter_columns=['lr'], metric_columns=['loss', 'accuracy', 'epoch'])
    out_dir = Path(HydraConfig.get().runtime.output_dir if config.tune.resume_dir is None else config.tune.resume_dir)

    resources_per_trial = {'cpu': 1, 'gpu': config.tune.gpus_per_trial}
    wrapped_trainable = tune.with_parameters(tune.with_resources(trainable, resources_per_trial), config=config)

    if config.tune.resume_dir is None:
        tuner = tune.Tuner(
            wrapped_trainable,
            param_space=hparams,
            tune_config=tune.TuneConfig(
                mode='max',
                metric='NED',
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=config.tune.num_samples,
            ),
            run_config=RunConfig(
                name=out_dir.name,
                stop=MetricTracker('NED', max_t),
                progress_reporter=reporter,
                storage_path=str(out_dir.parent.absolute()),
            ),
        )
    else:
        tuner = tune.Tuner.restore(config.tune.resume_dir, wrapped_trainable)

    results = tuner.fit()
    best_result = results.get_best_result()
    print('Best hyperparameters found were:', best_result.config)
    print('with result:\n', best_result)

if __name__ == '__main__':
    main()
