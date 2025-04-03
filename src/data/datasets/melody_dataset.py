from copy import deepcopy
from pathlib import Path

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset

from src.data.utils.slicer import Slicer
from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.pipelines.melody_pipeline import MelodyPipeline


class MelodyDataset(Dataset):

    def __init__(self, audio: list[Audio], melody: list[Melody]):
        """
        :param List[Audio] audio: Аудиофайлы.
        :param List[Melody] melody: Мелодии.
        """
        super().__init__()

        self.audio = audio
        self.melody = melody

        self.slicer = Slicer()
        self.pipeline = MelodyPipeline()

        self.sliced_audio, self.sliced_melody = self.slice_data(self.audio, self.melody)
        self.preprocessed_data = self._preprocess_data(self.sliced_audio, self.sliced_melody)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Возвращает элемент датасета.

        :param int idx: Индекс элемента
        :return Tuple[Tensor, Tensor, Tensor, Tensor]: Cпектрограмма, частоты нот, длительности нот, длина последовательности.
        """
        return self.preprocessed_data[idx]

    def __len__(self) -> int:
        return len(self.sliced_audio)

    @classmethod
    def from_path(cls, dataset_path: str | Path) -> 'MelodyDataset':
        """Загружает сэмплы из датасета.

        :param str | Path dataset_path: Путь к директории с датасетом
        :return MelodyDataset: Датасет
        """
        AUDIO_DIR = Path("audio")
        LABELS_DIR = Path("labels")

        dataset_path = Path(dataset_path)
        audio_dir = dataset_path / AUDIO_DIR
        labels_dir = dataset_path / LABELS_DIR

        audio_pathes = sorted(list(audio_dir.rglob("*.wav")))
        midi_pathes = sorted(list(labels_dir.rglob("*.mid")))

        audio = [Audio(audio_file) for audio_file in audio_pathes]
        melody = [Melody.from_midi(midi_file) for midi_file in midi_pathes]

        return cls(audio, melody)

    def slice_data(self, audio: list[Audio], melody: list[Melody]) -> tuple[list[Audio], list[Melody]]:
        """Нарезает аудиофайлы и мелодии на окна по 4 такта.
        Выполняет паддинг там, где это необходимо, добавляя паузы в мелодию.

        :param List[Audio] audio: Аудиофайлы.
        :param List[Melody] melody: Мелодии.
        :return Tuple[List[Audio], List[Melody]]: Нарезанные аудиофайлы и мелодии.
        """
        sliced_audio = []
        sliced_melody = []

        for a, m in tqdm(zip(audio, melody), total=len(audio), desc="Slicing audio and melody"):

            audio_file = deepcopy(a)
            melody_file = deepcopy(m)

            audio_file.trim_silence()

            target_duration = min(audio_file.duration, melody_file.duration)

            if audio_file.duration > target_duration:
                samples_to_keep = int(target_duration * audio_file.sample_rate)
                audio_file.waveform = audio_file.waveform[:, :samples_to_keep]

            if melody_file.duration > target_duration:
                total_beats = melody_file._seconds_to_beats(target_duration)
                accumulated_beats = 0.0
                new_notes = []

                for note in melody_file.notes:
                    remaining_beats = total_beats - accumulated_beats

                    if remaining_beats <= 0:
                        break

                    if accumulated_beats + note.duration <= total_beats:
                        new_notes.append(note)
                        accumulated_beats += note.duration

                    else:
                        trimmed_note = Note(note.note, remaining_beats)
                        new_notes.append(trimmed_note)
                        break

                melody_file.notes = new_notes

            audio_windows, melody_windows = self.slicer.slice_data(audio_file, melody_file)

            sliced_audio.extend(audio_windows)
            sliced_melody.extend(melody_windows)

        return sliced_audio, sliced_melody

    def _preprocess_data(self, audio: list[Audio], melody: list[Melody]) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Предобрабатывает аудиофайлы и мелодии.

        :param List[Audio] audio: Аудиофайлы.
        :param List[Melody] melody: Мелодии.
        :return List[Tuple[Tensor, Tensor, Tensor, Tensor]]: Спектрограмма, частоты нот, длительности нот, длина последовательности.
        """
        preprocessed_data = []

        for a, m in tqdm(zip(audio, melody), total=len(audio), desc="Preprocessing data"):
            preprocessed_data.append(self.pipeline.forward(a, m))

        return preprocessed_data
