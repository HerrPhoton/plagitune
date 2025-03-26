import argparse
from typing import Any
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torchmetrics import MeanSquaredError
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.nn.models.MelodyNet import MelodyNet
from src.data.loaders.melody_loader import get_dataloader
from src.data.datasets.melody_dataset import MelodyDataset
from src.data.pipelines.configs.slice_config import SliceConfig

CUR_PATH = Path(__file__).resolve().parent
torch.set_float32_matmul_precision('high')


class PLMelodyNet(L.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.lr = float(kwargs['lr'])

        self.freqs_weight = float(kwargs['freqs_weight'])
        self.durations_weight = float(kwargs['durations_weight'])
        self.seq_len_weight = float(kwargs['seq_len_weight'])

        self.scheduler_type = kwargs['scheduler_type']
        self.reduce_on_plateau_params = kwargs.get('reduce_on_plateau_params', {})
        self.cosine_annealing_params = kwargs.get('cosine_annealing_params', {})
        self.one_cycle_params = kwargs.get('one_cycle_params', {})

        self.optimizer_type = kwargs['optimizer_type']
        self.adam_params = kwargs.get('adam_params', {})
        self.adamw_params = kwargs.get('adamw_params', {})
        self.sgd_params = kwargs.get('sgd_params', {})

        self.model = MelodyNet()

        self.masked_loss_fn = nn.MSELoss(reduction='none')
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.train_metrics = nn.ModuleDict({
            'mse_freqs_hz': MeanSquaredError(),
            'mse_durations_beats': MeanSquaredError(),
            'mse_seq_len': MeanSquaredError(),
        })

        self.val_metrics = nn.ModuleDict({
            'mse_freqs_hz': MeanSquaredError(),
            'mse_durations_beats': MeanSquaredError(),
            'mse_seq_len': MeanSquaredError(),
        })

        self.test_metrics = nn.ModuleDict({
            'mse_freqs_hz': MeanSquaredError(),
            'mse_durations_beats': MeanSquaredError(),
            'mse_seq_len': MeanSquaredError(),
        })

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        spectrograms = batch[0]
        targets = batch[1:]

        preds = self.forward(spectrograms)
        losses = self._compute_losses(preds, targets)

        loss, loss_freqs, loss_durations, loss_len_seq = losses

        self._compute_denormalized_metrics(preds, targets, self.train_metrics)

        self.log('train_loss_freqs', loss_freqs, on_step=False, on_epoch=True)
        self.log('train_loss_durations', loss_durations, on_step=False, on_epoch=True)
        self.log('train_loss_len_seq', loss_len_seq, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:

        metrics = {
            'train_mse_freqs_hz': self.train_metrics['mse_freqs_hz'].compute(),
            'train_mse_durations_beats': self.train_metrics['mse_durations_beats'].compute(),
            'train_mse_seq_len': self.train_metrics['mse_seq_len'].compute(),
        }

        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.log(
            'train_score',
            metrics['train_mse_freqs_hz'] + metrics['train_mse_durations_beats'] + metrics['train_mse_seq_len'],
            on_step=False,
            on_epoch=True
        )

        for metric in self.train_metrics.values():
            metric.reset()

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        spectrograms = batch[0]
        targets = batch[1:]

        preds = self.forward(spectrograms)
        losses = self._compute_losses(preds, targets)

        loss, loss_freqs, loss_durations, loss_len_seq = losses

        self._compute_denormalized_metrics(preds, targets, self.val_metrics)

        self.log('val_loss_freqs', loss_freqs, on_step=False, on_epoch=True)
        self.log('val_loss_durations', loss_durations, on_step=False, on_epoch=True)
        self.log('val_loss_len_seq', loss_len_seq, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:

        metrics = {
            'val_mse_freqs_hz': self.val_metrics['mse_freqs_hz'].compute(),
            'val_mse_durations_beats': self.val_metrics['mse_durations_beats'].compute(),
            'val_mse_seq_len': self.val_metrics['mse_seq_len'].compute(),
        }

        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.log(
            'val_score',
            metrics['val_mse_freqs_hz'] + metrics['val_mse_durations_beats'] + metrics['val_mse_seq_len'],
            on_step=False,
            on_epoch=True
        )

        for metric in self.val_metrics.values():
            metric.reset()

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:

        spectrograms = batch[0]
        targets = batch[1:]

        preds = self.forward(spectrograms)
        losses = self._compute_losses(preds, targets)

        loss = losses[0]
        loss_freqs = losses[1]
        loss_durations = losses[2]
        loss_len_seq = losses[3]

        self._compute_denormalized_metrics(preds, targets, self.test_metrics)

        self.log('test_loss_freqs', loss_freqs, on_step=False, on_epoch=True)
        self.log('test_loss_durations', loss_durations, on_step=False, on_epoch=True)
        self.log('test_loss_len_seq', loss_len_seq, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_mse_freqs_hz': self.test_metrics['mse_freqs_hz'].compute(),
            'test_mse_durations_beats': self.test_metrics['mse_durations_beats'].compute(),
            'test_mse_seq_len': self.test_metrics['mse_seq_len'].compute(),
        }

        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.log(
            'test_score',
            metrics['test_mse_freqs_hz'] + metrics['test_mse_durations_beats'] + metrics['test_mse_seq_len'],
            on_step=False,
            on_epoch=True
        )

        for metric in self.test_metrics.values():
            metric.reset()

    def predict_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.predict(batch)

    def configure_optimizers(self) -> dict[str, Any]:

        if self.optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=float(self.adamw_params['weight_decay'])
            )

        elif self.optimizer_type == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=float(self.sgd_params['momentum']),
                weight_decay=float(self.sgd_params['weight_decay'])
            )

        elif self.optimizer_type == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=float(self.adam_params['weight_decay'])
            )

        if self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                patience = int(self.reduce_on_plateau_params['patience']),
                min_lr = float(self.reduce_on_plateau_params['min_lr']),
                threshold = float(self.reduce_on_plateau_params['threshold'])
            )

        elif self.scheduler_type == 'cosine_annealing':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max = int(self.cosine_annealing_params['T_max']),
                eta_min = float(self.cosine_annealing_params['eta_min'])
            )

        elif self.scheduler_type == 'one_cycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr = float(self.one_cycle_params['max_lr']),
                steps_per_epoch = int(self.one_cycle_params['steps_per_epoch']),
                epochs = int(self.one_cycle_params['epochs'])
            )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss"
        }

    def _compute_losses(
        self,
        preds: tuple[Tensor, Tensor, Tensor],
        targets: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Вычисляет потери для предсказанных и целевых значений.

        :param Tuple[Tensor, ...] preds: Предсказанные значения
        :param Tuple[Tensor, ...] targets: Целевые значения
        :return Tuple[Tensor, ...]: Общая потеря, потеря по частотам, потеря по классам, потеря по смещениям, потеря по длительностям
        """
        predicted_freqs = preds[0].squeeze(1)
        predicted_durations = preds[1].squeeze(1)
        predicted_seq_len = preds[2]

        target_freqs = targets[0]
        target_durations = targets[1]
        target_seq_len = targets[2]

        freqs_loss = self._masked_mse(predicted_freqs, target_freqs)
        durations_loss = self._masked_mse(predicted_durations, target_durations)
        seq_len_loss = self.loss_fn(predicted_seq_len, target_seq_len)

        loss = (
            self.freqs_weight * freqs_loss +
            self.durations_weight * durations_loss +
            self.seq_len_weight * seq_len_loss
        )

        return loss, freqs_loss, durations_loss, seq_len_loss

    def _masked_mse(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Вычисляет MSE с учетом паддинга.

        :param Tensor preds: Предсказанные значения [batch_size, seq_len]
        :param Tensor targets: Целевые значения [batch_size, seq_len]
        :return Tensor: Среднеквадратичная ошибка (MSE)
        """
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, :targets.shape[1]]

        mask = (targets != SliceConfig.label_pad_value).float()

        loss = self.masked_loss_fn(preds, targets)
        loss *= mask

        return loss.sum() / mask.sum()

    def _compute_denormalized_metrics(self, preds: tuple[Tensor, Tensor, Tensor], targets: tuple[Tensor, Tensor, Tensor], metrics_dict: dict):
        """Вычисляет метрики для денормализованных значений.

        :param preds: Предсказания модели
        :param targets: Целевые значения
        :param metrics_dict: Словарь с метриками
        """
        preds_freqs = preds[0][:, :targets[0].shape[1]]
        preds_durations = preds[1][:, :targets[1].shape[1]]
        preds_seq_len = preds[2]

        freqs_mask = (targets[0] != SliceConfig.label_pad_value)
        durations_mask = (targets[1] != SliceConfig.label_pad_value)

        denorm_preds_freqs, denorm_preds_durations, denorm_preds_seq_len = self.model.label_normalizer.inverse_transform(
            preds_freqs, preds_durations, preds_seq_len
        )
        denorm_target_freqs, denorm_target_durations, denorm_target_seq_len = self.model.label_normalizer.inverse_transform(
            targets[0], targets[1], targets[2]
        )

        metrics_dict['mse_freqs_hz'](
            denorm_preds_freqs[freqs_mask],
            denorm_target_freqs[freqs_mask]
        )
        metrics_dict['mse_durations_beats'](
            denorm_preds_durations[durations_mask],
            denorm_target_durations[durations_mask]
        )
        metrics_dict['mse_seq_len'](
            denorm_preds_seq_len,
            denorm_target_seq_len
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "-t", "--task",
        choices = ['train', 'test'],
        help = "Режим работы модели: train, test",
        default = 'train',
    )
    parser.add_argument(
        "-n", "--name",
        type = str,
        help = "Имя версии обучения",
    )
    parser.add_argument(
        "-m", "--model",
        type = str,
        help = "Путь до модели",
    )
    args = parser.parse_args()

    CUR_PATH = Path(__file__).parent

    with open(CUR_PATH / "../configs/train.yaml") as stream:
        config = yaml.safe_load(stream)

        batch_size = int(config['batch_size'])
        num_workers = int(config['num_workers'])
        epochs = int(config['epochs'])
        log_every_n_steps = int(config['log_every_n_steps'])

    match args.task:

        case 'train':

            model = PLMelodyNet(**config)

            logger = TensorBoardLogger(save_dir = CUR_PATH / "..", name="logs", version=args.name)
            lr_monitor = LearningRateMonitor(logging_interval = 'epoch')
            best_checkpoint = ModelCheckpoint(filename = "best", monitor = 'val_loss', mode = "min")
            last_checkpoint = ModelCheckpoint(filename = "last", monitor = 'epoch', mode = "max")

            train_dataset = MelodyDataset.from_path(CUR_PATH / config['train_dataset_path'])
            val_dataset = MelodyDataset.from_path(CUR_PATH / config['val_dataset_path'])

            train_loader = get_dataloader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True
            )

            val_loader = get_dataloader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                persistent_workers=True
            )

            trainer = L.Trainer(
                max_epochs = epochs,
                logger = logger,
                callbacks = [lr_monitor, best_checkpoint, last_checkpoint],
                log_every_n_steps=log_every_n_steps
            )

            trainer.fit(model, train_loader, val_loader)

        case 'test':

            model = PLMelodyNet.load_from_checkpoint(args.model, **config)
            model.eval()

            test_dataset = MelodyDataset.from_path(CUR_PATH / config['test_dataset_path'])
            test_loader = get_dataloader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                persistent_workers=True
            )

            trainer = L.Trainer(logger = False)
            trainer.test(model, test_loader)
