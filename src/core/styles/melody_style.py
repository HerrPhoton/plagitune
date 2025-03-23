from dataclasses import field, dataclass

from src.core.styles.base_style import BaseStyle


@dataclass
class MelodyStyle(BaseStyle):
    figsize: tuple[int, int] = (12, 6)
    grid_alpha: float = 0.2
    x_label: str | None = 'Время, с'
    x_ticks_fontsize: int | None = 10
    y_ticks_fontsize: int | None = 8
    lines_color: str = '#2F2F2F'
    lines_alpha: float = 0.3
    lines_linewidth: float = 0.5
    lines_linestyle: str = '-'
    note_edge_color: str = '#000000'
    note_linewidth: float = 1
    note_alpha: float = 0.7
    note_gradient: dict[str, str] = field(default_factory=lambda: {
        "C": '#FF6B6B',
        "C♯": '#FF9F40',
        "D": '#FFD93D',
        "D♯": '#6BCB77',
        "E": '#4D96FF',
        "F": '#9B72AA',
        "F♯": '#FF6B6B',
        "G": '#FF9F40',
        "G♯": '#FFD93D',
        "A": '#6BCB77',
        "A♯": '#4D96FF',
        "B": '#9B72AA',
    })
