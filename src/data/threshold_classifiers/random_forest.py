from sklearn.ensemble import RandomForestClassifier

from src.data.threshold_classifiers.base import BaseThresholdClassifier


class RandomForestThresholdClassifier(BaseThresholdClassifier):

    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(class_weight='balanced', max_depth=100)

    def get_feature_importance(self) -> dict[str, float]:
        """Возвращает важность признаков.

        :return dict[str, float]: Словарь с важностью признаков
        """
        if self.model is None or self.feature_names is None:
            return {}

        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))

        return {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
