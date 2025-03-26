from pathlib import Path

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset

from src.data.utils.slicer import Slicer
from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.pipelines.melody_pipeline import MelodyPipeline
from src.data.pipelines.configs.slice_config import SliceConfig
from src.data.pipelines.configs.melody_config import MelodyConfig
from src.data.pipelines.configs.pipeline_config import PipelineConfig
from src.data.pipelines.configs.spectrogram_config import SpectrogramConfig


class MelodyDataset(Dataset):

    def __init__(self, audio: list[Audio], melody: list[Melody], **kwargs):
        """
        :param List[Audio] audio: Аудиофайлы.
        :param List[Melody] melody: Мелодии.
        :param **kwargs: Параметры пайплайна
        """
        super().__init__()

        self.audio = audio
        self.melody = melody

        self.slice_config = SliceConfig(
            slice_size=kwargs.get('slice_size', SliceConfig.slice_size),
            hop_size=kwargs.get('hop_size', SliceConfig.hop_size),
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

        self.slicer = Slicer(
            slice_config=self.slice_config,
            spectrogram_config=self.spectrogram_config,
        )
        self.pipeline = MelodyPipeline(
            melody_config=self.melody_config,
            spectrogram_config=self.spectrogram_config,
            pipeline_config=self.pipeline_config
        )

        self.sliced_audio, self.sliced_melody = self.slice_data(self.audio, self.melody)

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        """Возвращает элемент датасета.

        :param int idx: Индекс элемента
        :return Tuple[Tensor, ...]:
            спектрограмма,
            частоты нот,
            классы нот,
            относительные смещения нот,
            маска нот,
            длительности нот
        """
        return self.pipeline.forward(self.sliced_audio[idx], self.sliced_melody[idx])

    def __len__(self) -> int:
        return len(self.sliced_audio)

    @classmethod
    def from_path(cls, dataset_path: Path, **kwargs) -> 'MelodyDataset':
        """Загружает сэмплы из датасета.

        :param Path dataset_path: Путь к директории с датасетом
        :param **kwargs: Параметры пайплайна
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

        return cls(audio, melody, **kwargs)

    def slice_data(self, audio: list[Audio], melody: list[Melody]) -> tuple[list[Audio], list[Melody]]:
        """Нарезает аудиофайлы и мелодии на окна фиксированного размера.
        Выполняет паддинг там, где это необходимо, добавляя паузы в мелодию.

        :param List[Audio] audio: Аудиофайлы.
        :param List[Melody] melody: Мелодии.
        :return Tuple[List[Audio], List[Melody]]: Нарезанные аудиофайлы и мелодии.
        """
        sliced_audio = []
        sliced_melody = []

        for a, m in tqdm(zip(audio, melody), total=len(audio), desc="Slicing audio and melody"):

            a.trim_silence()

            target_duration = min(a.duration, m.duration)

            if a.duration > target_duration:
                samples_to_keep = int(target_duration * a.sample_rate)
                a.waveform = a.waveform[:, :samples_to_keep]

            if m.duration > target_duration:
                total_beats = m._seconds_to_beats(target_duration)
                accumulated_beats = 0.0
                new_notes = []

                for note in m._notes:
                    remaining_beats = total_beats - accumulated_beats

                    if remaining_beats <= 0:
                        break

                    if accumulated_beats + note._duration <= total_beats:
                        # Нота помещается целиком
                        new_notes.append(note)
                        accumulated_beats += note._duration
                    else:
                        # Обрезаем последнюю ноту
                        trimmed_note = Note(note._note, max(0.25, remaining_beats))
                        new_notes.append(trimmed_note)
                        break

                m._notes = new_notes

            #audio_windows = self.slicer.slice_audio_by_measure(a, m._tempo)
            #melody_windows = self.slicer.slice_melody_by_measure(m)

            audio_windows, melody_windows = self.slicer.slice_data(a, m)

            sliced_audio.extend(audio_windows)
            sliced_melody.extend(melody_windows)

        return sliced_audio, sliced_melody
