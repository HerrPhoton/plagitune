from pathlib import Path

import torch
from torch import Tensor

from src.nn.inference.base import BaseInference
from src.data.loaders.audio import get_audio_dataloader
from src.data.configs.slicer import SlicerConfig
from src.data.datasets.audio import AudioDataset
from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.normalizers.label import LabelNormalizer
from src.data.structures.melody import Melody
from src.nn.train.MelodyNet_train import PLMelodyNet
from src.data.configs.melody_pipeline import MelodyPipelineConfig


class MelodyNetInference(BaseInference):

    def __init__(self, model_path: str):
        super().__init__(model_path)

        self.label_normalizer = LabelNormalizer(
            freq_min=MelodyPipelineConfig.f_min,
            freq_max=MelodyPipelineConfig.f_max,
            dur_min=MelodyPipelineConfig.dur_min,
            dur_max=MelodyPipelineConfig.dur_max,
            seq_len_min=MelodyPipelineConfig.seq_len_min,
            seq_len_max=MelodyPipelineConfig.seq_len_max,
        )

    def predict(self, audio: str | Path, tempo: int | None = None) -> Melody:
        """Извлечение мелодии из аудиофайла.

        :param str | Path | Audio audio: Аудиофайл или путь к аудиофайлу
        :return Melody: Извлеченная мелодия
        """
        audio = Audio(audio)

        if tempo is None:
            tempo = audio.get_tempo()

        dataset = AudioDataset([audio], hop_beats=SlicerConfig.measures_per_slice)

        dataloader = get_audio_dataloader(
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
                    mask = (freqs[j] != SlicerConfig.label_pad_value)
                    all_freqs.append(freqs[j][mask])
                    all_durations.append(durations[j][mask])

                break

        merged_freqs = torch.cat(all_freqs, dim=0)
        merged_durations = torch.cat(all_durations, dim=0)

        merged_predictions = (merged_freqs, merged_durations)
        return self._predictions_to_melody(merged_predictions, tempo)

    def _load_model(self, model_path: str) -> PLMelodyNet:
        return PLMelodyNet.load_from_checkpoint(model_path)

    def _predictions_to_melody(self, predictions: tuple[Tensor, Tensor], tempo: int) -> Melody:
        """Преобразование предсказаний в объект Melody.

        :param Tuple[Tensor, Tensor] predictions: Кортеж предсказаний (частоты, длительности)
        :return Melody: Объект Melody
        """
        freqs, durations = predictions

        freqs = freqs.cpu().numpy()
        durations = durations.cpu().numpy()

        notes = []

        for i in range(len(freqs)):
            duration = float(durations[i])
            freq = float(freqs[i])

            if freq <= 47.5:
                note = Note(None, duration)

            else:
                note = Note(freq, duration)

            notes.append(note)

        return Melody(notes, tempo=tempo)
