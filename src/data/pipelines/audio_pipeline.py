import torch
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB

from src.data.structures.audio import Audio
from src.data.structures.spectrogram import Spectrogram
from src.data.utils.spectrogram_normalizer import SpectrogramNormalizer
from src.data.pipelines.configs.pipeline_config import PipelineConfig
from src.data.pipelines.configs.spectrogram_config import SpectrogramConfig


class AudioPipeline(torch.nn.Module):

    def __init__(self, spectrogram_config: SpectrogramConfig, pipeline_config: PipelineConfig):
        super().__init__()

        self.spectrogram_config = spectrogram_config

        self.spec_normalizer = SpectrogramNormalizer(
            mean=pipeline_config.mean,
            std=pipeline_config.std
        )

        self.amplitude_to_db = AmplitudeToDB()

    def forward(self, audio: Audio) -> Tensor:

        self._preprocess_audio(audio)
        spectrogram = self._get_spectrogram(audio)

        spectrogram = self.amplitude_to_db(spectrogram.spectrogram)

        spectrogram = torch.nn.functional.interpolate(
            spectrogram.unsqueeze(0),
            size=(128, 256),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        spectrogram = self.spec_normalizer.transform(spectrogram)

        return spectrogram

    def _preprocess_audio(self, audio: Audio) -> Audio:
        """Предобрабатывает аудио.

        :param Audio audio: Аудио.
        """
        audio.resample(self.spectrogram_config.sample_rate)
        audio.to_mono()

        return audio

    def _get_spectrogram(self, audio: Audio) -> Spectrogram:
        """Возвращает экземпялр спектрограммы из аудио

        :param Audio audio: Экземпялр аудио
        :return Spectrogram: Спектрограмма
        """
        return Spectrogram.from_audio(
            audio=audio,
            n_mels=self.spectrogram_config.n_mels,
            hop_length=self.spectrogram_config.hop_length,
            n_fft=self.spectrogram_config.n_fft,
            f_min=self.spectrogram_config.f_min,
            f_max=self.spectrogram_config.f_max
        )
