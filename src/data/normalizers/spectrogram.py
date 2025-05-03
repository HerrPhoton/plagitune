import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class SpectrogramNormalizer:

    def __init__(self, mean: float | Tensor | None = None, std: float | Tensor | None = None):

        if mean is not None:
            self.mean = mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=torch.float32)

        if std is not None:
            self.std = std.clone().detach() if isinstance(std, torch.Tensor) else torch.tensor(std, dtype=torch.float32)

    def fit(self, spectrograms: list[Tensor], batch_size: int = 32, num_workers: int = 4) -> None:
        """Рассчитывает среднее и стандартное отклонение по спектрограммам.

        :param List[Tensor] spectrograms: Список спектрограмм
        :param int batch_size: Размер батча для обработки
        :param int num_workers: Количество рабочих процессов для загрузки данных
        """
        count = 0
        mean = 0.0
        M2 = 0.0

        dataset = TensorDataset(torch.arange(len(spectrograms)))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        for batch_indices in tqdm(loader, desc="Calculating mean and std"):

            batch_specs = [spectrograms[i.item()] for i in batch_indices[0]]

            max_freq_bins = 0
            max_time_steps = 0

            for spec in batch_specs:

                if spec.dim() == 3 and spec.size(0) == 1:
                    spec = spec.squeeze(0)

                max_freq_bins = max(max_freq_bins, spec.size(0))
                max_time_steps = max(max_time_steps, spec.size(1))

            for spectrogram in batch_specs:

                if spectrogram.dim() == 3 and spectrogram.size(0) == 1:
                    spectrogram = spectrogram.squeeze(0)

                batch_count = spectrogram.numel()
                batch_mean = spectrogram.mean().item()
                batch_var = spectrogram.var(unbiased=False).item()

                delta = batch_mean - mean
                new_count = count + batch_count
                mean = mean + delta * batch_count / new_count
                M2 = M2 + batch_var * batch_count + delta**2 * count * batch_count / new_count
                count = new_count

        self.std = torch.tensor(np.sqrt(M2 / (count - 1)))
        self.mean = torch.tensor(mean)

    def transform(self, spectrogram: Tensor) -> Tensor:
        """Нормализует спектрограмму.

        :param Tensor spectrogram: Спектрограмма
        :return Tensor: Нормализованная спектрограмма
        """
        return (spectrogram - self.mean) / (self.std + 1e-8)
