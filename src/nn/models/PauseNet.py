import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50

from src.data.configs.slicer_config import SlicerConfig


class PauseNet(nn.Module):

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
        self.sequence_length = SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice * 4

        self.pauses_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.sequence_length),
        )

    def forward(self, x: Tensor) -> Tensor:

        features = self.resnet(x)

        features = self.avg_pool(features)
        features = features.squeeze(2)
        features = features.transpose(1, 2)

        self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]

        pauses = self.pauses_head(lstm_out)

        return pauses

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        pauses = self.forward(x)

        return (torch.sigmoid(pauses) > 0.5).float()
