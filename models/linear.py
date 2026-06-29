import torch
import torch.nn as nn

from globals import (
    INPUT_SIZE,
    NUM_CLASSES,
)


class LinearModel(nn.Module):

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)
