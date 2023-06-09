from typing import List

import torch
import torch.nn as nn


class SimpleFF(nn.Module):
    def __init__(self, input_dim: int, layer_config: List[int]) -> None:
        """Simple Feedforward network.

        Args:
            input_dim (int): Input 1 dimension shape for the model
            layer_config (List[int]): List of integers that add a \
                Linear layer + ReLU to the model(except for the last \
                    layer, where only a Linear Layer is added)
        """
        super().__init__()

        assert len(layer_config) > 0
        self.layer_config = layer_config
        self.input_dim = input_dim

        self.layers = []
        self.feature_extractor = []
        self.output_layer = []
        if len(layer_config) > 1:
            for i in range(len(layer_config) - 1):
                self.feature_extractor.append(nn.Linear(input_dim, layer_config[i]))
                self.feature_extractor.append(nn.ReLU())
                input_dim = layer_config[i]

        self.output_layer.append(nn.Linear(layer_config[i], layer_config[-1]))
        # self.output_layer.append(nn.Softmax())

        self.layers = self.feature_extractor + self.output_layer

        self.module = nn.Sequential(*self.layers)
        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self.output_layer = nn.Sequential(*self.output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
