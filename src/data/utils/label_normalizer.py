from typing import List

import torch

from src.data.labels.melody_label import MelodyLabel


class LabelNormalizer:

    def __init__(
        self, 
        f_min: float | None = None, 
        f_max: float | None = None,
        offset_min: float | None = None,
        offset_max: float | None = None,
        dur_min: float | None = None,
        dur_max: float | None = None,
        seq_len_min: float | None = None,
        seq_len_max: float | None = None,
    ):
        """Инициализация нормализатора.

        :param f_min: Минимальная частота ноты.
        :param f_max: Максимальная частота ноты.
        :param offset_min: Минимальное смещение (отсноительное нижней ноты).
        :param offset_max: Максимальное смещение (отсноительное нижней ноты).
        :param dur_min: Минимальная длительность ноты (в долях).
        :param dur_max: Максимальная длительность ноты (в долях).        
        """
        self.f_min = f_min
        self.f_max = f_max
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.dur_min = dur_min
        self.dur_max = dur_max
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max

        self.eps = 1e-8
    
    def fit(self, labels: List[MelodyLabel]) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param List[MelodyLabel] labels: Список меток мелодий.
        """
        #all_freqs = torch.cat([label.freqs for label in labels])
        all_offsets = torch.cat([label.offsets for label in labels])
        all_durations = torch.cat([label.durations for label in labels])
        all_seq_lens = torch.tensor([len(label.offsets) for label in labels])
        
        self.seq_len_min = all_seq_lens.min().item()
        self.seq_len_max = all_seq_lens.max().item()

        # self.f_min = all_freqs.min().item()
        # self.f_max = all_freqs.max().item()
        
        self.offset_min = all_offsets.min().item()
        self.offset_max = all_offsets.max().item()

        self.dur_min = all_durations.min().item()
        self.dur_max = all_durations.max().item()

    def transform(self, label: MelodyLabel) -> MelodyLabel:
        """Нормализует метку мелодии.

        :param MelodyLabel label: Метка мелодии.
        :return MelodyLabel: Нормализованная метка.
        """
        offsets = (label.offsets - self.offset_min) / (self.offset_max - self.offset_min + self.eps)
        durations = (label.durations - self.dur_min) / (self.dur_max - self.dur_min + self.eps)
        seq_len = (label.seq_len - self.seq_len_min) / (self.seq_len_max - self.seq_len_min + self.eps)

        return MelodyLabel(
            offsets=offsets,
            durations=durations,
            seq_len=seq_len
        )

    def inverse_transform(self, label: MelodyLabel) -> MelodyLabel:
        """Денормализует метку мелодии.

        :param MelodyLabel label: Нормализованная метка мелодии.
        :return MelodyLabel: Исходная метка.
        """
        offsets = label.offsets * (self.offset_max - self.offset_min) + self.offset_min
        durations = label.durations * (self.dur_max - self.dur_min) + self.dur_min
        seq_len = label.seq_len * (self.seq_len_max - self.seq_len_min) + self.seq_len_min

        return MelodyLabel(
            offsets=offsets,
            durations=durations,
            seq_len=seq_len
        )




