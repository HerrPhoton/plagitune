from dataclasses import dataclass


@dataclass
class SpectrogramConfig:
    sample_rate: int = 44100
    win_length: int | None = 2048
    hop_length: int | None = 1024
    n_fft: int = 2048
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 20_000
