import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.data.datasets.melody_dataset import MelodyDataset
from src.data.pipelines.configs.slice_config import SliceConfig


def collate_fn(batch: list[tuple[list[Tensor], list[tuple[Tensor, ...]]]]) -> tuple[Tensor, ...]:
    """Собирает батч из нарезанных семплов.

    :param batch: Список кортежей (audio_windows, melody_windows), где каждое окно содержит
                 (input_spec, freqs, classes, offsets, target_masks, durations)
    :return: Батч подготовленных данных
    """
    spectrograms, offsets, durations, seq_lens = zip(*batch)

    def prepare_sequences(sequences: list[Tensor]) -> Tensor:
        return pad_sequence(
            [s.squeeze() if s.dim() > 1 else s for s in sequences],
            batch_first=True,
            padding_value=SliceConfig.label_pad_value
        )

    spectrograms_batch = torch.stack(spectrograms)
    offsets_batch = prepare_sequences(offsets)
    durations_batch = prepare_sequences(durations)
    seq_lengths_batch = torch.stack(seq_lens)

    return (
        spectrograms_batch,
        offsets_batch,
        durations_batch,
        seq_lengths_batch
    )


def get_dataloader(
    dataset: MelodyDataset,
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
