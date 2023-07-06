import torch
import torch.nn as nn


class Conv1dDropout(nn.Module):

    def __init__(self, conv_module: nn.Conv1d, dropout_rate: float=0.5) -> None:
        super().__init__()

        self.layers = [conv_module, nn.Dropout1d(dropout_rate)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Conv2dDropout(nn.Module):

    def __init__(self, conv_module: nn.Conv2d, dropout_rate: float=0.5) -> None:
        super().__init__()

        self.layers = [conv_module, nn.Dropout2d(dropout_rate)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Conv3dDropout(nn.Module):

    def __init__(self, conv_module: nn.Conv3d, dropout_rate: float=0.5) -> None:
        super().__init__()

        self.layers = [conv_module, nn.Dropout3d(dropout_rate)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LinearDropout(nn.Module):

    def __init__(self, conv_module: nn.Linear, dropout_rate: float=0.5) -> None:
        super().__init__()

        self.layers = [conv_module, nn.Dropout2d(dropout_rate)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
