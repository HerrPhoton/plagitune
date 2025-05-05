import torch
import torch.nn as nn
from torch import Tensor


class OverlapsNet(nn.Module):

    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        return (self.forward(x).sigmoid() >= 0.5).float()
