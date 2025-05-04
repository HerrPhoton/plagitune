import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.threshold_classifiers.base import BaseThresholdClassifier


class LogRegThresholdClassifier(BaseThresholdClassifier):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)

    def get_feature_importance(self) -> dict[str, float]:
        """Возвращает важность признаков.

        :return dict[str, float]: Словарь с важностью признаков
        """
        if self.model is None or self.feature_names is None:
            return {}

        importance = np.abs(self.model.coef_[0])
        feature_importance = dict(zip(self.feature_names, importance))

        return {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
