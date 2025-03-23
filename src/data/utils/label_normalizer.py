import torch
from torch import Tensor

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

    def fit(self, offsets: Tensor, durations: Tensor, seq_len: Tensor) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param Tensor offsets: Смещения нот.
        :param Tensor durations: Длительности нот.
        :param Tensor seq_len: Длина последовательности.
        """
        self.offset_min = offsets.min().item()
        self.offset_max = offsets.max().item()

        self.dur_min = durations.min().item()
        self.dur_max = durations.max().item()

        self.seq_len_min = seq_len.min().item()
        self.seq_len_max = seq_len.max().item()

    def fit_from_labels(self, labels: list[MelodyLabel]) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param List[MelodyLabel] labels: Список меток мелодий.
        """
        all_offsets = torch.cat([label.offsets for label in labels])
        all_durations = torch.cat([label.durations for label in labels])
        all_seq_lens = torch.tensor([len(label.offsets) for label in labels])

        self.fit(all_offsets, all_durations, all_seq_lens)

    def transform(self, offsets: Tensor, durations: Tensor, seq_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Нормализует метку мелодии.

        :param MelodyLabel label: Метка мелодии.
        :return MelodyLabel: Нормализованная метка.
        """
        offsets = (offsets - self.offset_min) / (self.offset_max - self.offset_min + self.eps)
        durations = (durations - self.dur_min) / (self.dur_max - self.dur_min + self.eps)
        seq_len = (seq_len - self.seq_len_min) / (self.seq_len_max - self.seq_len_min + self.eps)

        return offsets, durations, seq_len

    def transform_label(self, label: MelodyLabel) -> MelodyLabel:
        """Нормализует метку мелодии.

        :param MelodyLabel label: Метка мелодии.
        :return MelodyLabel: Нормализованная метка.
        """
        return MelodyLabel(*self.transform(label.offsets, label.durations, label.seq_len))

    def inverse_transform(self, offsets: Tensor, durations: Tensor, seq_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Денормализует метку мелодии.

        :param MelodyLabel label: Нормализованная метка мелодии.
        :return MelodyLabel: Исходная метка.
        """
        offsets = offsets * (self.offset_max - self.offset_min) + self.offset_min
        durations = durations * (self.dur_max - self.dur_min) + self.dur_min
        seq_len = seq_len * (self.seq_len_max - self.seq_len_min) + self.seq_len_min

        return offsets, durations, seq_len

    def inverse_transform_label(self, label: MelodyLabel) -> MelodyLabel:
        """Денормализует метку мелодии.

        :param MelodyLabel label: Нормализованная метка мелодии.
        :return MelodyLabel: Исходная метка.
        """
        return MelodyLabel(*self.inverse_transform(label.offsets, label.durations, label.seq_len))
