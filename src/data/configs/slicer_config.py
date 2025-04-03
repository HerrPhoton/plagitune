from dataclasses import dataclass


@dataclass
class SlicerConfig:
    beats_per_measure: int = 4
    measures_per_slice: int = 4
    audio_pad_value: float = 0
    label_pad_value: float = -100
    hop_beats: int = 1
