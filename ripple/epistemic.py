from typing import Tuple

import torch

from ripple.base import Base
from ripple.model import RippleModel


class DropoutEpistemic(Base):
    """Uses multiple forward passes with dropout \\ enabled to get the mean and std of \
    prediction.

    Args:
        model (RippleModel): RippleModel
        T (int): Number of iterations
    """

    T: int

    def __init__(self, model: RippleModel, T: int) -> None:
        super().__init__(model)
        self.T = T

    def forward(self, x: torch.Tensor, training: bool=True) -> \
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = []
        for _ in range(self.T):
            outputs.append(self.model(x, True))
        stacked_outputs = torch.stack(outputs)
        mean, std = stacked_outputs.mean(), stacked_outputs.std()

        return mean, (mean, std)
