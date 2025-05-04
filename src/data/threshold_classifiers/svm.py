from sklearn import svm

from src.data.threshold_classifiers.base import BaseThresholdClassifier


class SVMThresholdClassifier(BaseThresholdClassifier):

    def __init__(self):
        super().__init__()
        self.model = svm.SVC(kernel='poly', class_weight='balanced', probability=True)
