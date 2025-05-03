from dataclasses import dataclass


@dataclass
class AudioConfig:
    threshold_db: float = 60.0
    prop_decrease: float = 0.5
