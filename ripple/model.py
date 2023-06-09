from typing import Tuple

import torch
import torch.nn as nn


class RippleModel(nn.Module):
    """Ripple Model wrapper.

    This class enables wrapping generic PyTorch models to be compatible \
        with Ripple's functionalities.

    Args:
        feature_extractor (nn.Module): Feature extractor of the model
        output_layer (nn.Module): Final output module of the model
        input_shape (Tuple): Shape of the input tensor
        is_classification (bool): If this model is a classification model

    Raises:
        AssertionError: Multiple dimension outputs are not supported
    """

    feature_extractor: nn.Module
    output_layer: nn.Module
    input_shape: Tuple
    feature_shape: Tuple
    output_dim: int
    is_classification: bool

    def __init__(
        self,
        feature_extractor: nn.Module,
        output_layer: nn.Module,
        input_shape: Tuple,
        is_classification: bool,
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.output_layer = output_layer

        self.input_shape = input_shape
        self.is_classification = is_classification

        test_feature = self.feature_extractor(torch.randn((1, *self.input_shape)))
        self.feature_shape = test_feature.shape[1:]

        output_dim = self.output_layer(test_feature).shape[1:]

        if len(output_dim) > 1:
            raise AssertionError("More than one dim not supported")
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.feature_extractor(x))
