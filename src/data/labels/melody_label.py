from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.structures.melody import Melody


@dataclass
class MelodyLabel:

    intervals: Tensor
    durations: Tensor
    seq_len: Tensor

    @classmethod
    def from_melody(cls, melody: Melody) -> 'MelodyLabel':
        """Создает разметку из объекта Melody.

        :param Melody melody: Объект мелодии
        :return: MelodyLabel: Объект с различными представлениями мелодии
        """
        intervals = torch.tensor(melody.get_intervals(), dtype=torch.float32)
        durations = torch.tensor(melody.get_durations(), dtype=torch.float32)
        seq_len = torch.tensor([len(intervals)], dtype=torch.float32)

        return cls(
            intervals=intervals,
            durations=durations,
            seq_len=seq_len
        )
