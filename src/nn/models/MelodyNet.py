import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50

from src.data.configs.slicer_config import SlicerConfig
from src.data.normalizers.label_normalizer import LabelNormalizer
from src.data.configs.melody_pipeline_config import MelodyPipelineConfig


class MelodyNet(nn.Module):

    def __init__(
        self,
        hidden_size: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        :param int hidden_size: Размер скрытого состояния LSTM
        :param int num_lstm_layers: Количество слоев LSTM
        :param float dropout: Вероятность dropout
        :param bool bidirectional: Использовать ли двунаправленный LSTM
        """
        super().__init__()

        self.label_normalizer = LabelNormalizer(
            freq_min=MelodyPipelineConfig.f_min,
            freq_max=MelodyPipelineConfig.f_max,
            dur_min=MelodyPipelineConfig.dur_min,
            dur_max=MelodyPipelineConfig.dur_max,
            seq_len_min=MelodyPipelineConfig.seq_len_min,
            seq_len_max=MelodyPipelineConfig.seq_len_max,
        )

        self.resnet = resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            1,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False if self.resnet.conv1.bias is None else True
        )
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        heads_output_size = SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice * 4

        self.freqs_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, heads_output_size),
            nn.Sigmoid()
        )

        self.duration_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, heads_output_size),
            nn.Sigmoid()
        )

        self.seq_len_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        features = self.resnet(x)

        features = self.avg_pool(features)
        features = features.squeeze(2)
        features = features.transpose(1, 2)

        self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]

        freqs = self.freqs_head(lstm_out)
        durations = self.duration_head(lstm_out)
        seq_len = self.seq_len_head(lstm_out)

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
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SlicerConfig.label_pad_value)
            for seq in truncated_freqs
        ])
        padded_durations = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=SlicerConfig.label_pad_value)
            for seq in truncated_durations
        ])

        return padded_freqs, padded_durations
