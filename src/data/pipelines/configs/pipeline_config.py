from dataclasses import dataclass


@dataclass
class PipelineConfig:
    mean: float = -1.0866488218307495
    std: float = 17.541074344830257
    f_min: float = 0.0
    f_max: float = 1567.981689453125
    offset_min: float = 0.0
    offset_max: float = 41.0
    dur_min: float = 0.010317468084394932
    dur_max: float = 9.999992370605469
    seq_len_min: float = 2.0
    seq_len_max: float = 50.0
