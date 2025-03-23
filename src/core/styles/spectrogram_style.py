from dataclasses import dataclass

import roseus.mpl as rs

from src.core.styles.base_style import BaseStyle


@dataclass
class SpectrogramStyle(BaseStyle):
    vmin: float = -80.0
    vmax: float = 0.0
    color_bar: bool = True
    cmap: str = rs.roseus
    grid_visible: bool = False
    color_bar: bool = True
    x_label: str | None = 'Время, с'
    y_label: str | None = 'Частота, Гц'
    color_bar_label: str | None = 'Амплитуда, дБ'
