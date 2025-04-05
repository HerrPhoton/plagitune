from pathlib import Path

import torch
from torch import Tensor

from src.data.structures.audio import Audio
from src.nn.train.PauseNet_train import PLPauseNet
from src.data.loaders.audio_loader import get_audio_dataloader
from src.data.configs.slicer_config import SlicerConfig
from src.data.datasets.audio_dataset import AudioDataset


class PauseNetInference:

    def __init__(self, model_path: str):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = PLPauseNet.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_pause(self, audio: str | Path, tempo: int | None = None) -> Tensor:
        """Предсказание пауз в музыкальном произведении.

        :param str | Path | Audio audio: Аудиофайл или путь к аудиофайлу
        :return Tensor: Предсказанные паузы
        """
        audio = Audio(audio)

        if tempo is None:
            tempo = audio.get_tempo()

        dataset = AudioDataset([audio], hop_beats=SlicerConfig.measures_per_slice)

        dataloader = get_audio_dataloader(
            dataset,
            shuffle=False
        )

        all_pauses = []

        with torch.no_grad():
            for i, spectrograms in enumerate(dataloader):
                spectrograms = spectrograms.to(self.device)
                pauses = self.model.predict_step(spectrograms, i)

                all_pauses.append(pauses.cpu())

        merged_pauses = torch.cat(all_pauses, dim=0)
        merged_pauses = merged_pauses.view(-1)

        return merged_pauses
