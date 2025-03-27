import matplotlib.pyplot as plt

from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.core.styles.sample_style import SampleStyle
from src.data.structures.spectrogram import Spectrogram


class Sample:

    def __init__(self, audio: Audio, melody: Melody):
        self.audio = audio
        self.melody = melody

    def visualize_with_waveform(self, ax: plt.Axes | None = None, **style_kwargs) -> plt.Axes:
        """Визуализация образца.

        :param ax: Axes для отрисовки. Если None, создается новая фигура
        :param style_kwargs: Дополнительные параметры визуализации
        :return: Axes с отрисованным образцом
        """
        style = SampleStyle(**style_kwargs)

        match style.direction:

            case 'horizontal':
                fig, ax = plt.subplots(2, 1, figsize=style.figsize)

            case 'vertical':
                fig, ax = plt.subplots(1, 2, figsize=style.figsize)

        fig.patch.set_facecolor(style.background_color)

        self.audio.visualize(ax=ax[0])
        self.melody.visualize(ax=ax[1])

        return ax

    def visualize_with_spectrogram(self, ax: plt.Axes | None = None, **style_kwargs) -> plt.Axes:
        """Визуализация образца с спектрограммой.

        :param ax: Axes для отрисовки. Если None, создается новая фигура
        :param style_kwargs: Дополнительные параметры визуализации
        :return: Axes с отрисованным образцом
        """
        style = SampleStyle(**style_kwargs)

        spectrogram = Spectrogram.from_audio(self.audio)

        fig, ax = plt.subplots(2, 1, figsize=style.figsize)
        fig.patch.set_facecolor(style.background_color)

        spectrogram.visualize(ax=ax[0], **style_kwargs)
        self.melody.visualize(ax=ax[1], **style_kwargs)

        return ax
