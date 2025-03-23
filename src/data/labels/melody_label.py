from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.structures.melody import Melody


@dataclass
class MelodyLabel:

    offsets: Tensor
    durations: Tensor
    seq_len: Tensor

    @classmethod
    def from_melody(cls, melody: Melody) -> 'MelodyLabel':
        """Создает разметку из объекта Melody.

        :param Melody melody: Объект мелодии
        :return: MelodyLabel: Объект с различными представлениями мелодии
        """
        offsets = torch.tensor(melody.get_offsets(), dtype=torch.float32)
        durations = torch.tensor(melody.get_durations(), dtype=torch.float32)
        seq_len = torch.tensor([len(offsets)], dtype=torch.float32)

        return cls(
            offsets=offsets,
            durations=durations,
            seq_len=seq_len
        )
