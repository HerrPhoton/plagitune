import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from src.data.utils.label_normalizer import LabelNormalizer
from src.data.pipelines.configs.slice_config import SliceConfig
from src.data.pipelines.configs.pipeline_config import PipelineConfig


class MelodyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.slice_size = SliceConfig.slice_size

        self.label_normalizer = LabelNormalizer(
            freq_min=PipelineConfig.f_min,
            freq_max=PipelineConfig.f_max,
            dur_min=PipelineConfig.dur_min,
            dur_max=PipelineConfig.dur_max,
            seq_len_min=PipelineConfig.seq_len_min,
            seq_len_max=PipelineConfig.seq_len_max,
        )

        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            1,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False if self.resnet.conv1.bias is None else True
        )

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.freqs_head = nn.Sequential(
            nn.Linear(2048, self.slice_size),
            nn.Sigmoid()
        )
        self.duration_head = nn.Sequential(
            nn.Linear(2048, self.slice_size),
            nn.Sigmoid()
        )
        self.seq_len_head = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        features = self.resnet(x)

        features = self.avg_pool(features)
        features = features.flatten(1)

        freqs = self.freqs_head(features)
        durations = self.duration_head(features)
        seq_len = self.seq_len_head(features)

        return freqs, durations, seq_len

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        self.eval()

        freqs, durations, seq_len = self.forward(x)
        freqs, durations, seq_len = self.label_normalizer.inverse_transform(freqs, durations, seq_len)

        seq_lengths = seq_len.squeeze(-1).round().long()

        batch_size = freqs.size(0)
        truncated_freqs = []
        truncated_durations = []

        for i in range(batch_size):
            length = seq_lengths[i]
            truncated_freqs.append(freqs[i, :length])
            truncated_durations.append(durations[i, :length])

        max_len = max(len(seq) for seq in truncated_freqs)

        padded_freqs = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SliceConfig.label_pad_value)
            for seq in truncated_freqs
        ])
        padded_durations = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SliceConfig.label_pad_value)
            for seq in truncated_durations
        ])

        return padded_freqs, padded_durations
