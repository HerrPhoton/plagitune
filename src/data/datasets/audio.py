from pathlib import Path

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset

from src.data.utils.slicer import Slicer
from src.data.configs.slicer import SlicerConfig
from src.data.pipelines.audio import AudioPipeline
from src.data.structures.audio import Audio


class AudioDataset(Dataset):

    def __init__(self, audio: list[Audio], **kwargs):
        """
        :param List[Audio] audio: Аудиофайлы.
        """
        super().__init__()

        self.audio = audio

        self.slicer = Slicer(hop_beats=kwargs.get('hop_beats') or SlicerConfig.hop_beats)
        self.pipeline = AudioPipeline()

        self.sliced_audio = self.slice_audio(self.audio)
        self.preprocessed_data = self._preprocess_data(self.sliced_audio)


    def __getitem__(self, idx: int) -> Tensor:
        """Возвращает предобработанную спектрограмму.

        :param int idx: Индекс элемента
        :return Tensor: Спектрограмма
        """
        return self.preprocessed_data[idx]

    def __len__(self) -> int:
        return len(self.sliced_audio)

    @classmethod
    def from_path(cls, audio_path: str | Path) -> 'AudioDataset':
        """Создает датасет из пути к аудиофайлу или директории.

        :param str | Path audio_path: Путь к аудиофайлу или директории с аудиофайлами
        :return AudioDataset: Датасет
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        if audio_path.is_dir():
            audio_files = sorted(list(audio_path.rglob("*.wav")))

        else:
            audio_files = [audio_path]

        audio = [Audio(file) for file in audio_files]
        return cls(audio)

    def slice_audio(self, audio: list[Audio]) -> list[Audio]:
        """Нарезает аудиофайлы на окна по 4 такта.
        Выполняет паддинг там, где это необходимо, добавляя паузы.

        :param List[Audio] audio: Аудиофайлы.
        :return List[Audio]: Нарезанные аудиофайлы.
        """
        sliced_audio = []

        for a in tqdm(audio, total=len(audio), desc="Slicing audio"):

            a.trim_silence()

            audio_slices = self.slicer.slice_audio_by_measure(a, a.get_tempo())
            sliced_audio.extend(audio_slices)

        return sliced_audio

    def _preprocess_data(self, audio: list[Audio]) -> list[Tensor]:
        """Предобрабатывает аудиофайлы.

        :param List[Audio] audio: Аудиофайлы.
        :return List[Tensor]: Предобработанные аудиофайлы.
        """
        preprocessed_data = []

        for a in tqdm(audio, total=len(audio), desc="Preprocessing audio"):
            preprocessed_data.append(self.pipeline.forward(a))

        return preprocessed_data
