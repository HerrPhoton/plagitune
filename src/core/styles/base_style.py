from dataclasses import dataclass


@dataclass
class BaseStyle:

    figsize: tuple[int, int] = (15, 8)
    background_color: str = '#1A1A1A'
    text_color: str = '#FFFFFF'
    grid_visible: bool = True
    grid_color: str = '#2F2F2F'
    grid_alpha: float = 1.0
    grid_linestyle: str = '--'
    title: str | None = None
    title_fontsize: int = 14
    title_pad: int = 10
    x_label: str | None = None
    y_label: str | None = None
    ylim: tuple[float, float] | None = None
    xlim: tuple[float, float] | None = None
    ticks_fontsize: int | None = 10
    labels_fontsize: int | None = 10
