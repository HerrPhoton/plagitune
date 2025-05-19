import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Dataset

from src.nn.inference.base import BaseInference
from src.data.structures.melody import Melody
from src.nn.train.OverlapsNet_train import PLOverlapsNet


class OverlapsNetInference(BaseInference):

    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(self, melody1: Melody, melody2: Melody) -> bool:
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

        fpr_grid = np.linspace(0, 1, 100)
        tpr_curves = []
        aucs = []

        metrics = {
            "auc": {"mean": None, "lower": None, "upper": None},
            "roc": {"fpr": [], "tpr_mean": [], "tpr_lower": [], "tpr_upper": []}
        }

        for _ in tqdm(range(n_bootstraps), desc="Calculating CI"):
            pos_bootstrap = rng.choice(positive_indices, size=len(positive_indices), replace=True)
            neg_bootstrap = rng.choice(negative_indices, size=len(negative_indices), replace=True)

            bootstrap_indices = np.concatenate([pos_bootstrap, neg_bootstrap])

            rng.shuffle(bootstrap_indices)

            y_true = []
            y_pred = []

            for idx in bootstrap_indices:
                batch = all_samples[idx]
                batch = [x.to(self.device) if isinstance(x, Tensor) else x for x in batch]
                features, targets = batch

                with torch.no_grad():
                    probs = self.model.forward(features).sigmoid()
                    y_true.append(targets.cpu().numpy())
                    y_pred.append(probs.cpu().numpy())

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            fpr, tpr, _ = roc_curve(y_true, y_pred)

            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_curves.append(tpr_interp)

            aucs.append(auc(fpr, tpr))

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        auc_lower = np.percentile(aucs, lower_percentile)
        auc_upper = np.percentile(aucs, upper_percentile)

        tpr_curves = np.array(tpr_curves)
        tpr_mean = np.mean(tpr_curves, axis=0)
        tpr_lower = np.percentile(tpr_curves, lower_percentile, axis=0)
        tpr_upper = np.percentile(tpr_curves, upper_percentile, axis=0)

        return {
            "auc": {
                "mean": float(auc_mean),
                "std": float(auc_std),
                "lower": float(auc_lower),
                "upper": float(auc_upper)
            },
            "roc": {
                "fpr": fpr_grid,
                "tpr_mean": tpr_mean,
                "tpr_lower": tpr_lower,
                "tpr_upper": tpr_upper
            }
        }

    def _load_model(self, model_path: str) -> PLOverlapsNet:
        return PLOverlapsNet.load_from_checkpoint(model_path)
