from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import librosa
import torchaudio
import matplotlib.pyplot as plt
from torch import Tensor

from src.data.structures.audio import Audio
from src.core.styles.spectrogram import SpectrogramStyle


class Spectrogram:

    def __init__(
        self,
        waveform: Tensor,
        sample_rate: int,
        n_fft: int = 2048,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 20.0,
        f_max: float = 20_000.0,
        n_mels: int = 128,
        window_fn: torch.nn.Module = torch.hann_window,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = 'reflect',
    ):
        """Инициализация мел-спектрограммы для входного аудиосигнала

        :param torch.Tensor waveform: Исходный аудиосигнал
        :param int sample_rate: Частота дискретизации аудио
        :param int n_fft: Размер FFT
        :param int win_length: Размер окна
        :param int hop_length: Размер перекрытия окон
        :param float f_min: Минимальная частота
        :param float f_max: Максимальная частота
        :param int n_mels: Количество мел-фильтров
        :param float power: Степень для преобразования амплитуды в мощность
        :param torch.nn.Module window_fn: Функция окна
        :param bool center: Центрировать ли окна
        :param str pad_mode: Способ заполнения аудио сигнала
        """
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.window_fn = window_fn
        self.power = power
        self.center = center
        self.pad_mode = pad_mode

        if self.f_max is None:
            self.f_max = self.sample_rate // 2

        if self.win_length is None:
            self.win_length = self.n_fft

        if self.hop_length is None:
            self.hop_length = self.win_length // 2

        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            window_fn=window_fn,
            center=center,
            pad_mode=pad_mode,
            f_min=f_min,
            f_max=f_max
        )

        self.spectrogram = transform(self.waveform)
        self.n_frames = self.spectrogram.shape[2]

    def apply_threshold(self, threshold: float) -> 'Spectrogram':
        """Применяет пороговую фильтрацию.

        :param float threshold: Пороговое значение
        :return Spectrogram: Экземпляр Spectrogram с отфильтрованными значениями
        """
        filtered_spec = deepcopy(self)
        filtered_spec.spectrogram[filtered_spec.spectrogram < threshold] = 0

        return filtered_spec

    def visualize(self, ax: plt.Axes | None = None, **style_kwargs) -> plt.Axes:
        """Визуализирует спектрограмму.

        :param ax: Axes для отрисовки. Если None, создается новая фигура
        :param style_kwargs: Дополнительные параметры визуализации
        :return: Axes с отрисованной спектрограммой
        """
        style = SpectrogramStyle(**style_kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=style.figsize)
            fig.patch.set_facecolor(style.background_color)

        ax.set_facecolor(style.background_color)

        spec_db = librosa.power_to_db(
            self.spectrogram.squeeze().cpu().numpy(),
            ref=np.max,
            top_db=80.0
        )

        img = librosa.display.specshow(
            data=spec_db,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='mel',
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.f_min,
            fmax=self.f_max,
            ax=ax,
        )

        if style.x_label:
            ax.set_xlabel(style.x_label, color=style.text_color, size=style.labels_fontsize)

        if style.y_label:
            ax.set_ylabel(style.y_label, color=style.text_color, size=style.labels_fontsize)

        if style.xlim:
            ax.set_xlim(style.xlim)

        if style.ylim:
            ax.set_ylim(style.ylim)

        if style.title:
            ax.set_title(
                style.title,
                color=style.text_color,
                pad=style.title_pad,
                fontsize=style.title_fontsize
            )

        if style.grid_visible:
            ax.grid(
                True,
                linestyle=style.grid_linestyle,
                alpha=style.grid_alpha,
                color=style.grid_color
            )

        ax.tick_params(colors=style.text_color, labelsize=style.ticks_fontsize)

        for spine in ax.spines.values():
            spine.set_color(style.grid_color)

        if style.color_bar:
            fig = ax.figure
            cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.ax.tick_params(colors=style.text_color, labelsize=style.ticks_fontsize)
            if style.color_bar_label:
                cbar.set_label(style.color_bar_label, color=style.text_color, size=style.labels_fontsize)

        return ax

    @classmethod
    def from_audio(
        cls,
        audio: str | Path | Audio,
        n_fft: int = 2048,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 20.0,
        f_max: float = 20_000.0,
        n_mels: int = 128,
        window_fn: torch.nn.Module = torch.hann_window,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = 'reflect',
    ) -> 'Spectrogram':
        """Создает спектрограмму из аудиофайла.

        :param torch.Tensor waveform: Исходный аудиосигнал
        :param int sample_rate: Частота дискретизации аудио
        :param int n_fft: Размер FFT
        :param int win_length: Размер окна
        :param int hop_length: Размер перекрытия окон
        :param float f_min: Минимальная частота
        :param float f_max: Максимальная частота
        :param int n_mels: Количество мел-фильтров
        :param float power: Степень для преобразования амплитуды в мощность
        :param torch.nn.Module window_fn: Функция окна
        :param bool center: Центрировать ли окна
        :param str pad_mode: Способ заполнения аудио сигнала
        """
        if isinstance(audio, (str, Path)):
            audio = Audio(audio)

        return cls(
            waveform=audio.waveform,
            sample_rate=audio.sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            power=power,
            window_fn=window_fn,
            center=center,
            pad_mode=pad_mode,
            f_min=f_min,
            f_max=f_max
        )
