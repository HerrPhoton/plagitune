from typing import List

import torch
from torch.utils.data import DataLoader
from torch import Tensor

from src.data.datasets.audio_dataset import AudioDataset


def collate_fn(batch: List[List[Tensor]]) -> Tensor:
    """Собирает батч из нарезанных семплов.
    
    :param List[List[Tensor]] batch: Список списков спектрограмм для каждого аудио
    :return Tensor: Батч подготовленных данных
    """
    return torch.stack(batch)


def get_dataloader(
    dataset: AudioDataset, 
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
