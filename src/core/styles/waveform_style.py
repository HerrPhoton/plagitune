from typing import Tuple
from dataclasses import dataclass

from src.core.styles.base_style import BaseStyle


@dataclass
class WaveformStyle(BaseStyle):
    figsize: Tuple[int, int] = (12, 5)
    color: str = 'royalblue'
    alpha: float = 0.7
    linewidth: float = 1.0
    x_label: str | None = 'Время, с'
    y_label: str | None = 'Амплитуда'
    ylim: Tuple[float, float] | None = (-1.1, 1.1)