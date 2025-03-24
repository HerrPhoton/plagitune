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
            interval_min=PipelineConfig.interval_min,
            interval_max=PipelineConfig.interval_max,
            dur_min=PipelineConfig.dur_min,
            dur_max=PipelineConfig.dur_max,
            seq_len_min=PipelineConfig.seq_len_min,
            seq_len_max=PipelineConfig.seq_len_max,
        )

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )

        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(
                original_conv.weight.data.mean(dim=1, keepdim=True)
            )

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.interval_head = nn.Linear(2048, self.slice_size)
        self.duration_head = nn.Linear(2048, self.slice_size)
        self.seq_len_head = nn.Linear(2048, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        features = self.resnet(x)

        features = self.avg_pool(features)
        features = features.flatten(1)

        intervals = self.interval_head(features)
        durations = self.duration_head(features)
        seq_len = self.seq_len_head(features)

        return intervals, durations, seq_len

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        intervals, durations, seq_len = self.forward(x)
        intervals, durations, seq_len = self.label_normalizer.inverse_transform(intervals, durations, seq_len)

        seq_lengths = seq_len.squeeze(-1).round().long()

        batch_size = intervals.size(0)
        truncated_intervals = []
        truncated_durations = []

        for i in range(batch_size):
            length = seq_lengths[i]
            truncated_intervals.append(intervals[i, :length])
            truncated_durations.append(durations[i, :length])

        max_len = max(len(seq) for seq in truncated_intervals)

        padded_intervals = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SliceConfig.label_pad_value)
            for seq in truncated_intervals
        ])
        padded_durations = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SliceConfig.label_pad_value)
            for seq in truncated_durations
        ])

        return padded_intervals, padded_durations
