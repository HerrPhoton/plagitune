import torch
from torch import Tensor

from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.labels.pause_label import PauseLabel
from src.data.pipelines.audio_pipeline import AudioPipeline


class PausePipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.audio_pipeline = AudioPipeline()

    def forward(self, audio: Audio, melody: Melody) -> tuple[Tensor, Tensor]:

        label = self._get_label(melody)
        spectrogram = self.audio_pipeline.forward(audio)

        return (
            spectrogram,
            label.classes
        )

    def _get_label(self, melody: Melody) -> PauseLabel:
        """Возвращает разметку для мелодии

        :param Melody melody: Экземпляр мелодии
        :return PauseLabel: Разметка мелодии
        """
        return PauseLabel.from_melody(melody)
