import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.datasets.audio_dataset import AudioDataset


def collate_fn(batch: list[Tensor]) -> Tensor:
    """Собирает батч из окон аудиофайлов.

    :param List[Tensor] batch: Список спектрограмм.
    :return Tensor: Батч спектрограмм.
    """
    return torch.stack(batch)


def get_audio_dataloader(
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
