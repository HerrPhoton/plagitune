import torch
from torch import Tensor

from src.data.labels.melody_label import MelodyLabel


class LabelNormalizer:

    def __init__(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        dur_min: float | None = None,
        dur_max: float | None = None,
        seq_len_min: float | None = None,
        seq_len_max: float | None = None,
    ):
        """Инициализация нормализатора.

        :param freq_min: Минимальная частота.
        :param freq_max: Максимальная частота.
        :param dur_min: Минимальная длительность ноты (в долях).
        :param dur_max: Максимальная длительность ноты (в долях).
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.dur_min = dur_min
        self.dur_max = dur_max
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max

        self.eps = 1e-8

    def fit(self, freqs: Tensor, durations: Tensor, seq_len: Tensor) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param Tensor freqs: Частоты нот.
        :param Tensor durations: Длительности нот.
        :param Tensor seq_len: Длина последовательности.
        """
        self.freq_min = freqs.min().item()
        self.freq_max = freqs.max().item()

        self.dur_min = durations.min().item()
        self.dur_max = durations.max().item()

        self.seq_len_min = seq_len.min().item()
        self.seq_len_max = seq_len.max().item()

    def fit_from_labels(self, labels: list[MelodyLabel]) -> None:
        """Находит минимальные и максимальные значения для нормализации.

        :param List[MelodyLabel] labels: Список меток мелодий.
        """
        all_freqs = torch.cat([label.freqs for label in labels])
        all_durations = torch.cat([label.durations for label in labels])
        all_seq_lens = torch.tensor([len(label.freqs) for label in labels])

        self.fit(all_freqs, all_durations, all_seq_lens)

    def transform(self, freqs: Tensor, durations: Tensor, seq_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Нормализует метку мелодии.

        :param Tensor freqs: Частоты нот
        :param Tensor durations: Длительности нот
        :param Tensor seq_len: Длина последовательности
        :return: Кортеж нормализованных значений (частоты, длительности, длина)
        """
        # Логарифмическая нормализация частот в диапазон [0, 1]
        log_freqs = torch.log2(freqs + 1)
        log_freq_min = torch.log2(torch.tensor(self.freq_min + 1))
        log_freq_max = torch.log2(torch.tensor(self.freq_max + 1))
        freqs = (log_freqs - log_freq_min) / (log_freq_max - log_freq_min)

        durations = (durations - self.dur_min) / (self.dur_max - self.dur_min + self.eps)
        seq_len = (seq_len - self.seq_len_min) / (self.seq_len_max - self.seq_len_min + self.eps)

        return freqs, durations, seq_len

    def transform_label(self, label: MelodyLabel) -> MelodyLabel:
        """Нормализует метку мелодии.

        :param MelodyLabel label: Метка мелодии.
        :return MelodyLabel: Нормализованная метка.
        """
        return MelodyLabel(*self.transform(label.freqs, label.durations, label.seq_len))

    def inverse_transform(self, freqs: Tensor, durations: Tensor, seq_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Денормализует метку мелодии.

        :param Tensor freqs: Нормализованные частоты
        :param Tensor durations: Нормализованные длительности
        :param Tensor seq_len: Нормализованная длина последовательности
        :return: Кортеж денормализованных значений
        """
        # Обратное преобразование логарифмической нормализации
        log_freq_min = torch.log2(torch.tensor(self.freq_min + 1))
        log_freq_max = torch.log2(torch.tensor(self.freq_max + 1))
        log_freqs = freqs * (log_freq_max - log_freq_min) + log_freq_min
        freqs = torch.pow(2, log_freqs) - 1

        durations = durations * (self.dur_max - self.dur_min) + self.dur_min
        seq_len = seq_len * (self.seq_len_max - self.seq_len_min) + self.seq_len_min

        return freqs, durations, seq_len

    def inverse_transform_label(self, label: MelodyLabel) -> MelodyLabel:
        """Денормализует метку мелодии.

        :param MelodyLabel label: Нормализованная метка мелодии.
        :return MelodyLabel: Исходная метка.
        """
        return MelodyLabel(*self.inverse_transform(label.freqs, label.durations, label.seq_len))
