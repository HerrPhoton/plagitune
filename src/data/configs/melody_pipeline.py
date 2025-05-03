from dataclasses import dataclass


@dataclass
class MelodyPipelineConfig:
    f_min: float = 0.0
    f_max: float = 20_000.0
    dur_min: float = 0.25
    dur_max: float = 16.0
    seq_len_min: float = 0
    seq_len_max: float = 64
