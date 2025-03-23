from pathlib import Path

import numpy as np
import torch
import librosa
import torchaudio
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

    def visualize(self, **style_kwargs) -> None:
        """Визуализирует волновую форму аудиосигнала.

        :param style_kwargs: Дополнительные параметры визуализации
        """
        style = WaveformStyle(**style_kwargs)
        plt.style.use('dark_background')

        fig, axes = plt.subplots(
            self.num_channels,
            1,
            figsize=(style.figsize[0], style.figsize[1] * self.num_channels),
            facecolor=style.background_color
        )

        if self.num_channels == 1:
            axes = [axes]

        time_axis = np.linspace(0, self.duration, self.waveform.shape[1])

        for channel, ax in enumerate(axes):

            ax.plot(
                time_axis,
                self.waveform[channel].numpy(),
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

            ax.set_facecolor(style.background_color)

            ax.set_xlabel(style.x_label, color=style.text_color, size=style.labels_fontsize)
            ax.set_ylabel(style.y_label, color=style.text_color, size=style.labels_fontsize)

            ax.set_xlim(style.xlim)
            ax.set_ylim(style.ylim)

            ax.tick_params(colors=style.text_color, size=style.ticks_fontsize, labelsize=style.ticks_fontsize)

            for spine in ax.spines.values():
                spine.set_color(style.grid_color)

            if self.num_channels > 1:
                ax.set_title(f'Канал {channel + 1}', pad=style.title_pad, color=style.text_color)

        if style.title:
            fig.suptitle(
                style.title,
                color=style.text_color,
                fontsize=style.title_fontsize
            )

        plt.tight_layout()
        plt.show()

    @property
    def duration(self) -> float:
        """Возвращает длительность аудио в секундах.

        :return float: Длительность аудио в секундах
        """
        return self.waveform.shape[1] / self.sample_rate

    @property
    def num_channels(self) -> int:
        """Возвращает количество каналов аудио.

        :return int: Количество каналов
        """
        return self.waveform.shape[0]
