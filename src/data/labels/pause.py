from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.configs.slicer import SlicerConfig
from src.data.structures.melody import Melody


@dataclass
class PauseLabel:

    classes: Tensor

    @classmethod
    def from_melody(cls, melody: Melody) -> 'PauseLabel':
        """Создает разметку из объекта Melody, разделяя на четвертные доли.

        :param Melody melody: Объект мелодии
        :return PauseLabel: Объект с указанием пауз для каждой четвертной доли
        """
        total_quarters = SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice * 4
        classes = torch.zeros(total_quarters, dtype=torch.float32)

        current_position = 0.0
        quarter_idx = 0

        for note in melody.notes:
            quarters_in_note = note.duration * 4

            quarters_count = round(quarters_in_note)

            for _ in range(quarters_count):
                if quarter_idx < total_quarters:
                    classes[quarter_idx] = float(note.is_rest)
                    quarter_idx += 1

            current_position += note.duration

        return cls(classes=classes)
