from dataclasses import dataclass


@dataclass
class AudioPipelineConfig:
    mean: float = 6.772914886474609
    std: float = 16.016413179469225
    interpolate_size: tuple[int, int] = (128, 256)
    interpolate_mode: str = 'bilinear'
    interpolate_align_corners: bool = True
