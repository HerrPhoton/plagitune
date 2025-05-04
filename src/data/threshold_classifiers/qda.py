from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from src.data.threshold_classifiers.base import BaseThresholdClassifier


class QDAThresholdClassifier(BaseThresholdClassifier):

    def __init__(self):
        super().__init__()
        self.model = QDA()
