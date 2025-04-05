from copy import deepcopy

import torch
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB

from src.data.structures.audio import Audio
from src.data.configs.audio_config import AudioConfig
from src.data.structures.spectrogram import Spectrogram
from src.data.configs.spectrogram_config import SpectrogramConfig
from src.data.configs.audio_pipeline_config import AudioPipelineConfig
from src.data.normalizers.spectrogram_normalizer import SpectrogramNormalizer


class AudioPipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.spec_normalizer = SpectrogramNormalizer(
            mean=AudioPipelineConfig.mean,
            std=AudioPipelineConfig.std
        )
        self.amplitude_to_db = AmplitudeToDB()

    def forward(self, audio: Audio) -> Tensor:

        audio_copy = deepcopy(audio)
        audio_copy = self._preprocess_audio(audio_copy)

        spectrogram = self._get_spectrogram(audio_copy)
        spectrogram = self.amplitude_to_db(spectrogram.spectrogram)
        spectrogram = torch.nn.functional.interpolate(
            spectrogram.unsqueeze(0),
            size=AudioPipelineConfig.interpolate_size,
            mode=AudioPipelineConfig.interpolate_mode,
            align_corners=AudioPipelineConfig.interpolate_align_corners
        ).squeeze(0)
        spectrogram = self.spec_normalizer.transform(spectrogram)

        return spectrogram

    def _preprocess_audio(self, audio: Audio) -> Audio:
        """Предобрабатывает аудио.

        :param Audio audio: Аудио.
        :return Audio: Обработанное аудио
        """
        audio.to_mono()
        audio.resample(SpectrogramConfig.sample_rate)
        audio.denoise(AudioConfig.prop_decrease, SpectrogramConfig.n_fft)
        audio.normalize()

        return audio

    def _get_spectrogram(self, audio: Audio) -> Spectrogram:
        """Возвращает экземпялр спектрограммы из аудио

        :param Audio audio: Экземпялр аудио
        :return Spectrogram: Спектрограмма
        """
        return Spectrogram.from_audio(
            audio=audio,
            n_mels=SpectrogramConfig.n_mels,
            hop_length=SpectrogramConfig.hop_length,
            n_fft=SpectrogramConfig.n_fft,
            f_min=SpectrogramConfig.f_min,
            f_max=SpectrogramConfig.f_max
        )
