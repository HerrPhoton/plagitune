from pathlib import Path

import numpy as np
import torch
import librosa
import torchaudio
import noisereduce as nr
import matplotlib.pyplot as plt

from src.core.styles.waveform_style import WaveformStyle


class Audio:

    def __init__(self, audio_path: str | Path):

        self.audio_path = Path(audio_path)
        self.waveform, self.sample_rate = torchaudio.load(self.audio_path)

    def trim_silence(self, threshold_db: float = 60.0) -> None:
        """Обрезает тишину в начале и конце аудио.

        :param float threshold_db: Пороговое значение в децибелах
        """
        waveform_numpy = self.waveform.numpy()
        trimmed_audio, _ = librosa.effects.trim(waveform_numpy, top_db=threshold_db)
        self.waveform = torch.from_numpy(trimmed_audio)

    def resample(self, target_sample_rate: int) -> None:
        """Изменяет частоту дискретизации аудио.

        :param int target_sample_rate: Целевая частота дискретизации
        """
        if self.sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=target_sample_rate)
            self.waveform = resampler(self.waveform)
            self.sample_rate = target_sample_rate

    def to_mono(self) -> None:
        """Преобразует стерео аудио в моно."""
        if self.waveform.shape[0] > 1:
            self.waveform = torch.mean(self.waveform, dim=0, keepdim=True)

    def normalize(self) -> None:
        """Нормализует амплитуду аудиосигнала в диапазон [-1, 1]."""
        max_val = torch.max(torch.abs(self.waveform))

        if max_val > 0:
            self.waveform = self.waveform / max_val

    def denoise(self, prop_decrease: float = 0.2) -> None:
        """Применяет шумоподавление на основе спектрального вычитания.

        :param float prop_decrease: Степень фильтрации от 0 до 1
        """
        waveform_numpy = self.waveform.numpy()
        waveform_numpy = np.where(waveform_numpy == 0, 1e-6, waveform_numpy)

        reduced_noise = nr.reduce_noise(
            y=waveform_numpy,
            sr=self.sample_rate,
            prop_decrease=prop_decrease,
            n_fft=2048,
            use_torch=True
        )
        self.waveform = torch.from_numpy(reduced_noise)

    def get_tempo(self) -> int:
        tempo, _ = librosa.beat.beat_track(y=self.waveform.squeeze(0).numpy(), sr=self.sample_rate)
        return round(tempo[0])

    def visualize(self, ax: plt.Axes | None = None, **style_kwargs) -> plt.Axes:
        """Визуализирует волновую форму аудиосигнала.

        :param ax: Axes для отрисовки. Если None, создается новая фигура
        :param style_kwargs: Дополнительные параметры визуализации
        :return: Axes с отрисованной волновой формой
        """
        style = WaveformStyle(**style_kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=style.figsize)
            fig.patch.set_facecolor(style.background_color)

        ax.set_facecolor(style.background_color)

        time_axis = np.linspace(0, self.duration, self.waveform.shape[1])

        ax.plot(
            time_axis,
            self.waveform[0].numpy(),
            color=style.color,
            alpha=style.alpha,
            linewidth=style.linewidth,
        )

        if style.grid_visible:
            ax.grid(
                True,
                linestyle=style.grid_linestyle,
                alpha=style.grid_alpha,
                color=style.grid_color
            )

        if style.x_label:
            ax.set_xlabel(style.x_label, color=style.text_color, size=style.labels_fontsize)
        if style.y_label:
            ax.set_ylabel(style.y_label, color=style.text_color, size=style.labels_fontsize)

        if style.xlim:
            ax.set_xlim(style.xlim)
        if style.ylim:
            ax.set_ylim(style.ylim)

        ax.tick_params(colors=style.text_color, labelsize=style.ticks_fontsize)

        for spine in ax.spines.values():
            spine.set_color(style.grid_color)

        if style.title:
            ax.set_title(
                style.title,
                color=style.text_color,
                fontsize=style.title_fontsize,
                pad=style.title_pad
            )

        return ax

    @property
    def duration(self) -> float:
        """Возвращает длительность аудио в секундах.

        :return float: Длительность аудио в секундах
        """
        return self.waveform.shape[1] / self.sample_rate
