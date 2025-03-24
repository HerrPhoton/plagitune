import torch
from torch import Tensor

from src.data.labels.melody_label import MelodyLabel


class LabelNormalizer:

    NOTE_TO_REST = -0.1
    REST_TO_NOTE = 1.1

    def __init__(
        self,
        interval_min: float | None = None,
        interval_max: float | None = None,
        dur_min: float | None = None,
        dur_max: float | None = None,
        seq_len_min: float | None = None,
        seq_len_max: float | None = None,
    ):
        """Инициализация нормализатора.

        :param interval_min: Минимальное расстояние между нотами.
        :param interval_max: Максимальное расстояние между нотами.
        :param dur_min: Минимальная длительность ноты (в долях).
        :param dur_max: Максимальная длительность ноты (в долях).
        """
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.dur_min = dur_min
        self.dur_max = dur_max
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max

        self.eps = 1e-8

    def fit(self, intervals: Tensor, durations: Tensor, seq_len: Tensor) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param Tensor intervals: Расстояния между нотами.
        :param Tensor durations: Длительности нот.
        :param Tensor seq_len: Длина последовательности.
        """
        finite_mask = torch.isfinite(intervals)
        finite_intervals = intervals[finite_mask]

        self.interval_min = finite_intervals.min().item()
        self.interval_max = finite_intervals.max().item()

        self.dur_min = durations.min().item()
        self.dur_max = durations.max().item()

        self.seq_len_min = seq_len.min().item()
        self.seq_len_max = seq_len.max().item()

    def fit_from_labels(self, labels: list[MelodyLabel]) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param List[MelodyLabel] labels: Список меток мелодий.
        """
        all_intervals = torch.cat([label.intervals for label in labels])
        all_durations = torch.cat([label.durations for label in labels])
        all_seq_lens = torch.tensor([len(label.intervals) for label in labels])

        self.fit(all_intervals, all_durations, all_seq_lens)

    def transform(self, intervals: Tensor, durations: Tensor, seq_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Нормализует метку мелодии.

        :param MelodyLabel label: Метка мелодии.
        :return MelodyLabel: Нормализованная метка.
        """
        inf_mask = torch.isinf(intervals)
        neg_inf_mask = intervals == float('-inf')
        pos_inf_mask = intervals == float('inf')

        normalized_intervals = intervals.clone()

        finite_mask = ~inf_mask
        normalized_intervals[finite_mask] = (
            (intervals[finite_mask] - self.interval_min) /
            (self.interval_max - self.interval_min + self.eps)
        )

        normalized_intervals[neg_inf_mask] = self.NOTE_TO_REST
        normalized_intervals[pos_inf_mask] = self.REST_TO_NOTE

        durations = (durations - self.dur_min) / (self.dur_max - self.dur_min + self.eps)
        seq_len = (seq_len - self.seq_len_min) / (self.seq_len_max - self.seq_len_min + self.eps)

        return normalized_intervals, durations, seq_len

    def transform_label(self, label: MelodyLabel) -> MelodyLabel:
        """Нормализует метку мелодии.

        :param MelodyLabel label: Метка мелодии.
        :return MelodyLabel: Нормализованная метка.
        """
        return MelodyLabel(*self.transform(label.intervals, label.durations, label.seq_len))

    def inverse_transform(self, intervals: Tensor, durations: Tensor, seq_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Денормализует метку мелодии.

        :param MelodyLabel label: Нормализованная метка мелодии.
        :return MelodyLabel: Исходная метка.
        """
        note_to_rest_mask = (intervals - self.NOTE_TO_REST).abs() < 0.1
        rest_to_note_mask = (intervals - self.REST_TO_NOTE).abs() < 0.1
        special_mask = note_to_rest_mask | rest_to_note_mask

        denormalized_intervals = intervals.clone()

        normal_mask = ~special_mask
        denormalized_intervals[normal_mask] = (
            intervals[normal_mask] * (self.interval_max - self.interval_min) +
            self.interval_min
        )

        denormalized_intervals[note_to_rest_mask] = float('-inf')
        denormalized_intervals[rest_to_note_mask] = float('inf')

        durations = durations * (self.dur_max - self.dur_min) + self.dur_min
        seq_len = seq_len * (self.seq_len_max - self.seq_len_min) + self.seq_len_min

        return denormalized_intervals, durations, seq_len

    def inverse_transform_label(self, label: MelodyLabel) -> MelodyLabel:
        """Денормализует метку мелодии.

        :param MelodyLabel label: Нормализованная метка мелодии.
        :return MelodyLabel: Исходная метка.
        """
        return MelodyLabel(*self.inverse_transform(label.intervals, label.durations, label.seq_len))
