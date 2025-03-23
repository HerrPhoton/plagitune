from dataclasses import dataclass
from typing import Literal


@dataclass
class MelodyConfig:
    threshold: float = 0.5