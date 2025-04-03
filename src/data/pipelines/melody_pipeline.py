import torch
from torch import Tensor

from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.labels.melody_label import MelodyLabel
from src.data.utils.label_normalizer import LabelNormalizer
from src.data.pipelines.audio_pipeline import AudioPipeline
from src.data.configs.melody_pipeline_config import MelodyPipelineConfig


class MelodyPipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.audio_pipeline = AudioPipeline()

        self.label_normalizer = LabelNormalizer(
            freq_min=MelodyPipelineConfig.f_min,
            freq_max=MelodyPipelineConfig.f_max,
            dur_min=MelodyPipelineConfig.dur_min,
            dur_max=MelodyPipelineConfig.dur_max,
            seq_len_min=MelodyPipelineConfig.seq_len_min,
            seq_len_max=MelodyPipelineConfig.seq_len_max,
        )

    def forward(self, audio: Audio, melody: Melody) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        label = self._get_label(melody)

        spectrogram = self.audio_pipeline.forward(audio)
        label = self.label_normalizer.transform_label(label)

        return (
            spectrogram,
            label.freqs,
            label.durations,
            label.seq_len
        )

    def _get_label(self, melody: Melody) -> MelodyLabel:
        """Возвращает разметку для мелодии

        :param Melody melody: Экземпляр мелодии
        :return MelodyLabel: Разметка мелодии
        """
        return MelodyLabel.from_melody(melody)
