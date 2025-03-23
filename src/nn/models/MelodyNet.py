import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from src.data.utils.label_normalizer import LabelNormalizer
from src.data.pipelines.configs.slice_config import SliceConfig
from src.data.pipelines.configs.pipeline_config import PipelineConfig


class MelodyNet(nn.Module):

    def __init__(
        self,
        hidden_size: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """Модель для транскрибации мелодии из спектрограммы.

        :param int hidden_size: Размер скрытого состояния LSTM
        :param int num_lstm_layers: Количество слоев LSTM
        :param float dropout: Вероятность dropout
        :param bool bidirectional: Использовать ли двунаправленный LSTM
        """
        super().__init__()

        self.slice_size = SliceConfig.slice_size

        self.label_normalizer = LabelNormalizer(
            f_min=PipelineConfig.f_min,
            f_max=PipelineConfig.f_max,
            offset_min=PipelineConfig.offset_min,
            offset_max=PipelineConfig.offset_max,
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

        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None))
        )

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # self.frequency_head = nn.Sequential(
        #     #nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 1)
        # )

        # self.classes_head = nn.Sequential(
        #     #nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 13)
        # )

        self.offset_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.slice_size)
        )

        self.duration_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.slice_size)
        )

        self.seq_len_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:

        features = self.resnet(x)

        features = self.adapter(features)
        features = features.squeeze(2)
        features = features.transpose(1, 2)

        self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]

        offsets = self.offset_head(lstm_out)
        durations = self.duration_head(lstm_out)
        seq_len = self.seq_len_head(lstm_out)

        return offsets, durations, seq_len

    def predict(self, x: Tensor) -> tuple[Tensor, ...]:

        offsets, durations, seq_len = self.forward(x)
        offsets, durations, seq_len = self.label_normalizer.inverse_transform(offsets, durations, seq_len)

        seq_lengths = seq_len.squeeze(-1).round().long()

        batch_size = offsets.size(0)
        truncated_offsets = []
        truncated_durations = []

        for i in range(batch_size):
            length = seq_lengths[i]
            truncated_offsets.append(offsets[i, :length])
            truncated_durations.append(durations[i, :length])

        max_len = max(len(seq) for seq in truncated_offsets)
        padded_offsets = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SliceConfig.label_pad_value)
            for seq in truncated_offsets
        ])
        padded_durations = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SliceConfig.label_pad_value)
            for seq in truncated_durations
        ])

        return padded_offsets, padded_durations
