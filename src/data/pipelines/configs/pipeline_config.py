from dataclasses import dataclass


@dataclass
class PipelineConfig:
    mean: float = -1.0866488218307495
    std: float = 17.541074344830257
    f_min: float = 0.0
    f_max: float = 1567.981689453125
    offset_min: float = 0.0
    offset_max: float = 41.0
    interval_min: float = -22.0
    interval_max: float = 26.0
    dur_min: float = 7.856235606595874e-05
    dur_max: float = 9.999990463256836
    seq_len_min: float = 2.0
    seq_len_max: float = 49.0
