from dataclasses import dataclass

from src.core.styles.melody import MelodyStyle


@dataclass
class MatcherStyle(MelodyStyle):

    figsize: tuple[int, int] = (15, 10)

    non_match_color: str | None = '#808080'

    match_color: str = '#FF4444'
    match_edge_color: str = '#000000'
    match_alpha: float = 0.8

    subplot_titles: tuple[str, str] | None = None
    subplot_title_fontsize: int = 12

    suptitle: str | None = None
    suptitle_fontsize: int = 14
