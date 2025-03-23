import torch
from torch import Tensor

from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.labels.melody_label import MelodyLabel
from src.data.structures.spectrogram import Spectrogram
from src.data.utils.label_normalizer import LabelNormalizer
from src.data.pipelines.audio_pipeline import AudioPipeline
from src.data.pipelines.configs.melody_config import (
    MelodyConfig,)
from src.data.pipelines.configs.pipeline_config import (
    PipelineConfig,)
from src.data.pipelines.configs.spectrogram_config import (
    SpectrogramConfig,)


class MelodyPipeline(torch.nn.Module):

    def __init__(
        self,
        melody_config: MelodyConfig,
        spectrogram_config: SpectrogramConfig,
        pipeline_config: PipelineConfig
    ):
        """Инициализация пайплайна для экстракции мелодии.

        :param MelodyConfig melody_config: Параметры мелодии
        :param SpectrogramConfig spectrogram_config: Параметры спектрограммы
        :param PipelineConfig pipeline_config: Параметры пайплайна
        """
        super().__init__()

        self.melody_config = melody_config
        self.spectrogram_config = spectrogram_config
        self.pipeline_config = pipeline_config

        self.audio_pipeline = AudioPipeline(
            spectrogram_config=self.spectrogram_config,
            pipeline_config=self.pipeline_config
        )

        self.label_normalizer = LabelNormalizer(
            f_min=pipeline_config.f_min,
            f_max=pipeline_config.f_max,
            offset_min=pipeline_config.offset_min,
            offset_max=pipeline_config.offset_max,
            dur_min=pipeline_config.dur_min,
            dur_max=pipeline_config.dur_max,
            seq_len_min=pipeline_config.seq_len_min,
            seq_len_max=pipeline_config.seq_len_max,
        )

    def forward(self, audio: Audio, melody: Melody) -> tuple[list[Tensor], ...]:

        spectrogram = self.audio_pipeline._get_spectrogram(audio)
        label = self._get_label(spectrogram, melody)

        spectrogram = self.audio_pipeline.forward(audio)
        label = self.label_normalizer.transform(label)

        return (
            spectrogram,
            label.offsets,
            label.durations,
            label.seq_len
        )

    def _get_label(self, spectrogram: Spectrogram, melody: Melody) -> MelodyLabel:
        """Возвращает метки для мелодии

        :param Melody melody: Экземпляр мелодии
        :return MelodyLabel: Разметка мелодии
        """
        return MelodyLabel.from_melody(
            melody=melody,
            spectrogram=spectrogram,
            threshold=self.melody_config.threshold
        )
