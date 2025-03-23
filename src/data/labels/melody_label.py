from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.structures.melody import Melody
from src.data.structures.spectrogram import Spectrogram


@dataclass
class MelodyLabel:

    offsets: Tensor
    durations: Tensor
    seq_len: Tensor

    @classmethod
    def from_melody(
        cls,
        melody: Melody,
        spectrogram: Spectrogram,
        threshold: float,
    ) -> 'MelodyLabel':
        """Создает разметку из объекта Melody.

        :param Melody melody: Объект мелодии
        :param Spectrogram spectrogram: Спектрограмма аудиофайла
        :param float threshold: Порог для бинаризации спектрограммы мелодии
        :return: MelodyLabel: Объект с различными представлениями мелодии
        """
        notes = melody._notes

        # Смещения нот относительной самой низкой ноты
        lowest_midi = float('inf')

        for note in notes:
            if not note.is_rest:
                lowest_midi = min(lowest_midi, note.midi_number)

        if lowest_midi == float('inf'):
            lowest_midi = 0

        offsets = torch.tensor([
                (note.midi_number - lowest_midi + 1)
                if not note.is_rest else 0.0 for note in notes
            ],
            dtype=torch.float32,
        )

        # Длительности нот в долях
        durations = torch.tensor([note._duration for note in notes], dtype=torch.float32)

        # Длина последовательности
        seq_len = torch.tensor([len(notes)], dtype=torch.float32)

        return cls(
            offsets=offsets,
            durations=durations,
            seq_len=seq_len
        )
