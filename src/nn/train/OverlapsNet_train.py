import argparse
from typing import Any
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import BinaryAUROC

from src.data.loaders.overlaps import get_overlaps_dataloader
from src.nn.models.OverlapsNet import OverlapsNet
from src.data.datasets.overlaps import OverlapsDataset

CUR_PATH = Path(__file__).resolve().parent
torch.set_float32_matmul_precision('high')


class PLOverlapsNet(L.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.lr = float(kwargs['lr'])

        self.scheduler_type = kwargs['scheduler_type']
        self.reduce_on_plateau_params = kwargs.get('reduce_on_plateau_params', {})
        self.cosine_annealing_params = kwargs.get('cosine_annealing_params', {})
        self.one_cycle_params = kwargs.get('one_cycle_params', {})

        self.optimizer_type = kwargs['optimizer_type']
        self.adam_params = kwargs.get('adam_params', {})
        self.adamw_params = kwargs.get('adamw_params', {})
        self.sgd_params = kwargs.get('sgd_params', {})

        self.model = OverlapsNet(
            in_features = int(kwargs['in_features']),
            dropout = float(kwargs['dropout'])
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(kwargs['pos_weight'])]))

        self.train_metrics = BinaryAUROC()
        self.val_metrics = BinaryAUROC()
        self.test_metrics = BinaryAUROC()

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        features, targets = batch

        logits = self.forward(features)
        loss = self.loss_fn(logits, targets)

        probs = torch.sigmoid(logits)
        self.train_metrics.update(probs, targets.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:

        self.log(
            'train_auroc',
            self.train_metrics.compute(),
            on_step=False,
            on_epoch=True
        )
        self.train_metrics.reset()


    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        features, targets = batch

        logits = self.forward(features)
        loss = self.loss_fn(logits, targets)

        probs = torch.sigmoid(logits)
        self.val_metrics.update(probs, targets.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:

        self.log(
            'val_auroc',
            self.val_metrics.compute(),
            on_step=False,
            on_epoch=True
        )
        self.val_metrics.reset()


    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        features, targets = batch

        logits = self.forward(features)
        loss = self.loss_fn(logits, targets)

        probs = torch.sigmoid(logits)
        self.test_metrics.update(probs, targets.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log(
            'test_auroc',
            self.test_metrics.compute(),
            on_step=False,
            on_epoch=True
        )
        self.test_metrics.reset()

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> tuple[Tensor, Tensor]:
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
                patience=int(self.reduce_on_plateau_params['patience']),
                min_lr=float(self.reduce_on_plateau_params['min_lr']),
                threshold=float(self.reduce_on_plateau_params['threshold'])
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }

        elif self.scheduler_type == 'cosine_annealing':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.cosine_annealing_params['T_max']),
                eta_min=float(self.cosine_annealing_params['eta_min'])
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }

        elif self.scheduler_type == 'one_cycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=float(self.one_cycle_params['max_lr']),
                steps_per_epoch=int(self.one_cycle_params['steps_per_epoch']),
                epochs=int(self.one_cycle_params['epochs'])
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }


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

    with open(CUR_PATH / "../configs/OverlapsNet_train.yaml") as stream:
        config = yaml.safe_load(stream)

        batch_size = int(config['batch_size'])
        num_workers = int(config['num_workers'])
        epochs = int(config['epochs'])
        log_every_n_steps = int(config['log_every_n_steps'])

    match args.task:

        case 'train':

            model = PLOverlapsNet(**config)

            logger = TensorBoardLogger(save_dir = CUR_PATH / ".." / "logs", name="OverlapsNet", version=args.name)
            lr_monitor = LearningRateMonitor(logging_interval = 'epoch')
            best_checkpoint = ModelCheckpoint(filename = "best", monitor = 'val_loss', mode = "min")
            last_checkpoint = ModelCheckpoint(filename = "last", monitor = 'epoch', mode = "max")

            train_dataset = OverlapsDataset.from_path(CUR_PATH / config['train_dataset_path'], split = 'train')
            val_dataset = OverlapsDataset.from_path(CUR_PATH / config['val_dataset_path'], split = 'val')

            train_loader = get_overlaps_dataloader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True
            )

            val_loader = get_overlaps_dataloader(
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

            model = PLOverlapsNet.load_from_checkpoint(args.model, **config)
            model.eval()

            test_dataset = OverlapsDataset.from_path(CUR_PATH / config['test_dataset_path'], split = 'test')
            test_loader = get_overlaps_dataloader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                persistent_workers=True
            )

            trainer = L.Trainer(logger = False)
            trainer.test(model, test_loader)
