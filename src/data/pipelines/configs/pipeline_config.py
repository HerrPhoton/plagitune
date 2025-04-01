from dataclasses import dataclass


@dataclass
class PipelineConfig:
    mean: float = 8.240909576416016
    std: float = 15.86678615808288


    f_min: float = 0.0
    f_max: float = 20_000
    dur_min: float = 0.25
    dur_max: float = 16.0
    seq_len_min: float = 0
    seq_len_max: float = 64



    offset_min: float = 0.0
    offset_max: float = 41.0
    interval_min: float = -22.0
    interval_max: float = 26.0
