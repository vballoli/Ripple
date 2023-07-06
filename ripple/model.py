from typing import Tuple

import torch
import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    """Return's the torch activation module.

    Args:
        activation (str): Choices: ["relu", "softmax", "identity"]

    Returns:
        nn.Module: The corresponding activation module
    """
    activation = activation.lower()
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'softmax':
        return torch.nn.Softmax()
    elif activation == 'identity':
        return torch.nn.Identity()
    else:
        raise AssertionError(f"Unsupported activation name: {activation}")

class Conv1dActivationDropout(nn.Module):
    def __init__(self, conv: torch.nn.Conv1d, activation: str, dropout_rate: float) \
     -> None:
        super().__init__()

        self.conv = conv
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.conv(x)))


class Conv2dActivationDropout(nn.Module):
    def __init__(self, conv: torch.nn.Conv1d, activation: str, dropout_rate: float) \
    -> None:
        super().__init__()

        self.conv = conv
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.conv(x)))


class Conv3dActivationDropout(nn.Module):
    def __init__(self, conv: torch.nn.Conv1d, activation: str, dropout_rate: float) \
        -> None:
        super().__init__()

        self.conv = conv
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout3d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.conv(x)))


class LinearActivationDropout(nn.Module):
    def __init__(self, linear: torch.nn.Linear, activation: str, dropout_rate: float) \
        -> None:
        super().__init__()

        self.linear = linear
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


def replace_module(module: nn.Module, activation: str, dropout_rate: float) \
    -> torch.nn.Module:
    """Recursively replaces every module with module + dropout inplace.
    Source: https://github.com/vballoli/nfnets-pytorch/blob/main/nfnets/utils.py#LL9C1-L28C38

    Usage: replace_conv(model) #(In-line replacement)
    Args:
        module (nn.Module): target's model whose convolutions must be replaced.
        dropout_rate (float): rate of dropout for that module

    Returns:
        nn.Module: new module with added dropout layers
    """
    if isinstance(module, torch.nn.Conv1d):
        new_module = Conv1dActivationDropout(module, activation, dropout_rate)
        return new_module
    elif isinstance(module, torch.nn.Conv2d):
        new_module = Conv2dActivationDropout(module, activation, dropout_rate)
        return new_module
    elif isinstance(module, torch.nn.Conv3d):
        new_module = Conv3dActivationDropout(module, activation, dropout_rate)
        return new_module
    elif isinstance(module, torch.nn.Linear):
        new_module = LinearActivationDropout(module, activation, dropout_rate)
        return new_module

    for name, mod in module.named_children():
        if isinstance(mod, torch.nn.Dropout):
            new_module = torch.nn.Identity()
        if isinstance(mod, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Softmax)):
            new_module = torch.nn.Identity()

    named_children = list(module.named_children())
    for name, mod in named_children:
        if isinstance(mod, torch.nn.Conv1d):
            new_module = Conv1dActivationDropout(
                mod, activation, dropout_rate
            )
        elif isinstance(mod, torch.nn.Conv2d):
            new_module = Conv2dActivationDropout(
                mod, activation, dropout_rate
            )
        elif isinstance(mod, torch.nn.Conv3d):
            new_module = Conv3dActivationDropout(
                mod, activation, dropout_rate
            )
        elif isinstance(mod, torch.nn.Linear):
            new_module = LinearActivationDropout(
                mod, activation, dropout_rate
            )
        elif isinstance(mod, torch.nn.Dropout):
            new_module = torch.nn.Identity()
        elif isinstance(mod,
                        (Conv1dActivationDropout,
                         Conv2dActivationDropout,
                         Conv3dActivationDropout)):
            new_module = mod
        else:
            new_module = mod

        setattr(module, name, new_module)

    for name, mod in named_children:
        replace_module(mod, activation, dropout_rate)

    return module

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

    def forward(self, x: torch.Tensor, training: bool=True) -> torch.Tensor:
        self.feature_extractor.train(training)
        self.output_layer.train(training)
        return self.output_layer(self.feature_extractor(x))

    def replace_modules(self, feature_dropout_rate: float,
                        feature_activation: str,
                        output_dropout_rate: float,
                        output_activation: str) -> None:
        self.feature_extractor = replace_module(
            self.feature_extractor, feature_activation, feature_dropout_rate
        )
        self.output_layer = replace_module(
            self.output_layer, output_activation, output_dropout_rate
        )
