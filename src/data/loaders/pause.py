import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.datasets.pause import PauseDataset


def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Собирает батч из нарезанных семплов.

    :param batch: Список кортежей (спектрограмма, указание пауз).
    :return: Батч подготовленных данных.
    """
    spectrograms, classes = zip(*batch)

    spectrograms_batch = torch.stack(spectrograms)
    classes_batch = torch.stack(classes)

    return (
        spectrograms_batch,
        classes_batch
    )


def get_pause_dataloader(
    dataset: PauseDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    persistent_workers: bool = False,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers
    )
