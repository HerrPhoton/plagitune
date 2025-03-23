from pathlib import Path
from typing import List, Tuple
from copy import deepcopy

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.structures.audio import Audio
from src.data.pipelines.audio_pipeline import AudioPipeline
from src.data.pipelines.configs.slice_config import SliceConfig
from src.data.pipelines.configs.spectrogram_config import SpectrogramConfig
from src.data.pipelines.configs.pipeline_config import PipelineConfig
from src.data.utils.slicer import Slicer


class AudioDataset(Dataset):

    def __init__(self, audio: List[Audio], **kwargs):
        """
        :param List[Audio] audio: Аудиофайлы.
        :param **kwargs: Параметры пайплайна
        """
        super().__init__()

        self.audio = audio

        self.slice_config = SliceConfig(
            slice_size=kwargs.get('slice_size', SliceConfig.slice_size),
            hop_size=kwargs.get('hop_size', SliceConfig.hop_size),
            audio_pad_value=kwargs.get('spec_pad_value', SliceConfig.audio_pad_value),
            label_pad_value=kwargs.get('label_pad_value', SliceConfig.label_pad_value),
        )
        self.spectrogram_config = SpectrogramConfig(
            sample_rate=kwargs.get('sample_rate', SpectrogramConfig.sample_rate),
            win_length=kwargs.get('win_length', SpectrogramConfig.win_length),
            hop_length=kwargs.get('hop_length', SpectrogramConfig.hop_length),
            n_fft=kwargs.get('n_fft', SpectrogramConfig.n_fft),
            n_mels=kwargs.get('n_mels', SpectrogramConfig.n_mels),
            f_min=kwargs.get('f_min', SpectrogramConfig.f_min),
            f_max=kwargs.get('f_max', SpectrogramConfig.f_max),
        )
        self.pipeline_config = PipelineConfig(
            mean=kwargs.get('mean', PipelineConfig.mean),
            std=kwargs.get('std', PipelineConfig.std),
            f_min=kwargs.get('f_min', PipelineConfig.f_min),
            f_max=kwargs.get('f_max', PipelineConfig.f_max),
            offset_min=kwargs.get('offset_min', PipelineConfig.offset_min),
            offset_max=kwargs.get('offset_max', PipelineConfig.offset_max),
            dur_min=kwargs.get('dur_min', PipelineConfig.dur_min),
            dur_max=kwargs.get('dur_max', PipelineConfig.dur_max),
        )
        
        self.slicer = Slicer(
            slice_config=self.slice_config,
            spectrogram_config=self.spectrogram_config,
        )
        self.pipeline = AudioPipeline(
            spectrogram_config=self.spectrogram_config,
            pipeline_config=self.pipeline_config
        )
        
        self.sliced_audio = self.slice_audio(self.audio)

    def __getitem__(self, idx: int) -> List[Tensor]:
        """Возвращает элемент датасета.

        :param int idx: Индекс элемента
        :return List[Tensor]: Список спектрограмм окон аудиофайла
        """
        return self.pipeline.forward(self.sliced_audio[idx])

    def __len__(self) -> int:
        return len(self.sliced_audio)

    @classmethod
    def from_path(cls, audio_path: str | Path, **kwargs) -> 'AudioDataset':
        """Создает датасет из пути к аудиофайлу или директории.
        
        :param str | Path audio_path: Путь к аудиофайлу или директории с аудиофайлами
        :param **kwargs: Параметры пайплайна
        :return AudioDataset: Датасет
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        if audio_path.is_dir():
            audio_files = sorted(list(audio_path.rglob("*.wav")))

        else:
            audio_files = [audio_path]
            
        audio = [Audio(file) for file in audio_files]
        return cls(audio, **kwargs)

    def slice_audio(self, audio: List[Audio]) -> List[List[Audio]]:
        """Нарезает аудиофайлы на окна фиксированного размера.
        Выполняет паддинг там, где это необходимо, добавляя паузы.

        :param List[Audio] audio: Аудиофайлы.
        :return List[List[Audio]]: Нарезанные аудиофайлы.
        """
        sliced_audio = []

        for a in tqdm(audio, total=len(audio), desc="Slicing audio"):
            
            a.trim_silence()

            audio_slices = self.slicer.slice_audio(a)
            sliced_audio.extend(audio_slices)
        
        return sliced_audio

    

