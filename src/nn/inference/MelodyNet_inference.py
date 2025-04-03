from pathlib import Path

import torch
from torch import Tensor

from src.data.utils.slicer import Slicer
from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.nn.train.MelodyNet_train import PLMelodyNet
from src.data.loaders.audio_loader import get_audio_dataloader
from src.data.configs.slicer_config import SlicerConfig
from src.data.datasets.audio_dataset import AudioDataset
from src.data.utils.label_normalizer import LabelNormalizer
from src.data.configs.melody_pipeline_config import MelodyPipelineConfig


class MelodyInference:

    def __init__(self, model_path: str):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = PLMelodyNet.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.slicer = Slicer(hop_beats=SlicerConfig.beats_per_measure)
        self.label_normalizer = LabelNormalizer(
            freq_min=MelodyPipelineConfig.f_min,
            freq_max=MelodyPipelineConfig.f_max,
            dur_min=MelodyPipelineConfig.dur_min,
            dur_max=MelodyPipelineConfig.dur_max,
            seq_len_min=MelodyPipelineConfig.seq_len_min,
            seq_len_max=MelodyPipelineConfig.seq_len_max,
        )

    def extract_melody(self, audio: str | Path, tempo: int | None = None) -> Melody:
        """Извлечение мелодии из аудиофайла.

        :param str | Path | Audio audio: Аудиофайл или путь к аудиофайлу
        :return Melody: Извлеченная мелодия
        """
        audio = Audio(audio)

        if tempo is None:
            tempo = audio.get_tempo()

        dataset = AudioDataset([audio])
        dataset.sliced_audio = self.slicer.slice_audio_by_measure(audio, tempo)
        dataset.preprocessed_data = dataset._preprocess_data(dataset.sliced_audio)

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

            if freq < 20:
                note = Note(None, duration)

            else:
                note = Note(freq, duration)

            notes.append(note)

        return Melody(notes, tempo=tempo)
