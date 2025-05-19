import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.datasets.overlaps import OverlapsDataset


def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Собирает батч из нарезанных семплов.

    :param batch: Список кортежей (признаки, классы).
    :return: Батч подготовленных данных.
    """
    features, targets = zip(*batch)

    features_tensor = torch.stack(features)
    targets_tensor = torch.cat(targets)

    return features_tensor, targets_tensor

def get_overlaps_dataloader(
    dataset: OverlapsDataset,
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
        persistent_workers=persistent_workers,
        collate_fn=collate_fn
    )
