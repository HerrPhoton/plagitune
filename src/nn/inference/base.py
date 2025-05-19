from abc import abstractmethod
from typing import Any, TypeVar

import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset

T = TypeVar('T')


class BaseInference:

    def __init__(self, model_path: str):
        """
        :param str model_path: Путь к весам модели
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def _load_model(self, model_path: str) -> Any:
        """Загрузка модели из чекпоинта.

        :param str model_path: Путь к весам модели
        :return Any: Загруженная модель
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Получение предсказаний модели.

        :param Any input_data: Входные данные
        :return Any: Предсказания модели
        """
        pass

    def bootstrap_inference(
        self,
        dataset: Dataset,
        dataloader_fn: callable,
        n_bootstraps: int = 1000,
        num_workers: int = 4,
        confidence_level: float = 0.95,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Бутстрап-инференс для оценки неопределенности модели.

        :param Dataset dataset: Датасет для оценки
        :param callable dataloader_fn: Функция для создания даталоадера
        :param int n_bootstraps: Количество бутстрап-итераций
        :param int num_workers: Количество воркеров для загрузки данных
        :param float confidence_level: Уровень доверия для интервалов
        :param int | None seed: Сид для генератора случайных чисел
        :return dict[str, float]: Результаты бутстрапа (среднее, std, границы доверительного интервала)
        """
        rng = np.random.default_rng(seed)

        loader = dataloader_fn(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False
        )

        all_samples = []
        all_targets = []

        for batch in tqdm(loader, desc="Loading test samples"):
            all_samples.append(batch)
            all_targets.append(batch[1].item())

        all_targets = np.array(all_targets)
        positive_indices = np.where(all_targets == 1)[0]
        negative_indices = np.where(all_targets == 0)[0]

        metrics = []

        for _ in tqdm(range(n_bootstraps), desc="Calculating CI"):
            self.model.test_metrics.reset()

            pos_bootstrap = rng.choice(positive_indices, size=len(positive_indices), replace=True)
            neg_bootstrap = rng.choice(negative_indices, size=len(negative_indices), replace=True)

            bootstrap_indices = np.concatenate([pos_bootstrap, neg_bootstrap])

            rng.shuffle(bootstrap_indices)

            for idx in bootstrap_indices:
                batch = all_samples[idx]
                batch = [x.to(self.device) if isinstance(x, Tensor) else x for x in batch]
                features, targets = batch

                with torch.no_grad():
                    probs = self.model.forward(features).sigmoid()
                    self.model.test_metrics.update(probs, targets.int())

            metric = self.model.test_metrics.compute().item()
            metrics.append(metric)

        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        results = {
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics)),
            'lower_bound': float(np.percentile(metrics, lower_percentile)),
            'upper_bound': float(np.percentile(metrics, upper_percentile)),
        }

        return results
