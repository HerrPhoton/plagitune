from typing import Literal
from dataclasses import dataclass

from src.core.styles.base_style import BaseStyle


@dataclass
class SampleStyle(BaseStyle):
    figsize: tuple[int, int] = (12, 10)
    background_color: str = '#1A1A1A'
    direction: Literal['horizontal', 'vertical'] = 'horizontal'
