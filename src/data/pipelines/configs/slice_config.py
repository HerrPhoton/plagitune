from dataclasses import dataclass


@dataclass
class SliceConfig:
    slice_size: int = 256
    hop_size: int = slice_size // 2
    audio_pad_value: float = 0
    label_pad_value: float = -100
