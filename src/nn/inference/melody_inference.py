from pathlib import Path

import torch
from torch import Tensor

from src.data.utils.slicer import Slicer
from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.nn.train.MelodyNet_train import PLMelodyNet
from src.data.loaders.audio_loader import get_dataloader
from src.data.datasets.audio_dataset import AudioDataset
from src.data.utils.label_normalizer import LabelNormalizer
from src.data.pipelines.audio_pipeline import AudioPipeline
from src.data.pipelines.configs.slice_config import SliceConfig
from src.data.pipelines.configs.melody_config import MelodyConfig
from src.data.pipelines.configs.pipeline_config import PipelineConfig
from src.data.pipelines.configs.spectrogram_config import SpectrogramConfig


class MelodyInference:

    def __init__(self, model_path: str, **kwargs):

        self.slice_config = SliceConfig(
            slice_size=kwargs.get('slice_size', SliceConfig.slice_size),
            hop_size=kwargs.get('hop_size', SliceConfig.slice_size),
            audio_pad_value=kwargs.get('spec_pad_value', SliceConfig.audio_pad_value),
            label_pad_value=kwargs.get('label_pad_value', SliceConfig.label_pad_value),
        )
        self.melody_config = MelodyConfig(
            threshold=kwargs.get('threshold', MelodyConfig.threshold),
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
            dur_min=kwargs.get('dur_min', PipelineConfig.dur_min),
            dur_max=kwargs.get('dur_max', PipelineConfig.dur_max),
            seq_len_min=kwargs.get('seq_len_min', PipelineConfig.seq_len_min),
            seq_len_max=kwargs.get('seq_len_max', PipelineConfig.seq_len_max),
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = PLMelodyNet.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.slicer = Slicer(
            slice_config=self.slice_config,
            spectrogram_config=self.spectrogram_config
        )
        self.audio_pipeline = AudioPipeline(
            spectrogram_config=self.spectrogram_config,
            pipeline_config=self.pipeline_config
        )
        self.label_normalizer = LabelNormalizer(
            freq_min=self.pipeline_config.f_min,
            freq_max=self.pipeline_config.f_max,
            dur_min=self.pipeline_config.dur_min,
            dur_max=self.pipeline_config.dur_max,
            seq_len_min=self.pipeline_config.seq_len_min,
            seq_len_max=self.pipeline_config.seq_len_max,
        )

    def extract_melody(self, audio: str | Path, tempo: int | None = None) -> Melody:
        """Извлечение мелодии из аудиофайла.

        :param str | Path | Audio audio: Аудиофайл или путь к аудиофайлу
        :return Melody: Извлеченная мелодия
        """
        audio = Audio(audio)

        if tempo is None:
            tempo = audio.get_tempo()

        dataset = AudioDataset([audio], [tempo])

        dataloader = get_dataloader(
            dataset,
            shuffle=False
        )

        all_freqs = []
        all_durations = []

        with torch.no_grad():
            for i, spectrograms in enumerate(dataloader):
                spectrograms = spectrograms.to(self.device)
                freqs, durations = self.model.predict_step(spectrograms, i)

                for j in range(freqs.size(0)):
                    mask = (freqs[j] != SliceConfig.label_pad_value)
                    all_freqs.append(freqs[j][mask])
                    all_durations.append(durations[j][mask])

                break

        merged_freqs = torch.cat(all_freqs, dim=0)
        merged_durations = torch.cat(all_durations, dim=0)

        merged_predictions = (merged_freqs, merged_durations)
        return self._predictions_to_melody(merged_predictions, tempo)

    def _predictions_to_melody(self, predictions: tuple[Tensor, ...], tempo: int) -> Melody:
        """Преобразование предсказаний в объект Melody.

        :param Tuple[Tensor, ...] predictions: Кортеж предсказаний (частоты, классы, смещения, длительности)
        :return Melody: Объект Melody
        """
        freqs, durations = predictions

        freqs = freqs.cpu().numpy()
        durations = durations.cpu().numpy()

        notes = []

        for i in range(len(freqs)):
            duration = float(durations[i])
            freq = float(freqs[i])

            if freq < 20:
                note = Note(None, duration)

            else:
                note = Note(freq, duration)

            notes.append(note)

        return Melody(notes, tempo=tempo)
