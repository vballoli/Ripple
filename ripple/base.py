from copy import deepcopy

import torch
import torch.nn as nn

from .constants import FEATURE_EXTRACTOR, OUTPUT_LAYER
from .model import RippleModel


class Base(nn.Module):
    def __init__(self, model: RippleModel) -> None:
        """Base Class for Ripple modules.

        Args:
            model (RippleModel): Ripple Model
        """
        super().__init__()

        self.model = model

    def copy_layer(self, name: str) -> nn.Module:
        """Provides a copy of a particular part of the model.

        Args:
            name (Literal[FEATURE_EXTRACTOR, OUTPUT_LAYER]): Supported \
                  layers that can be copied.

        Raises:
            NotImplementedError: If the input argument is not \
                in the supported choices.

        Returns:
            nn.Module: A copy of the requested module
        """
        assert name in [FEATURE_EXTRACTOR, OUTPUT_LAYER]
        if name == FEATURE_EXTRACTOR:
            layer = self.model.feature_extractor
        elif name == OUTPUT_LAYER:
            layer = self.model.output_layer[0]
        else:
            raise NotImplementedError("Unsupported layer name")

        assert isinstance(layer, nn.Module), (
            f"Expected nn.Module, received {type(layer)} for Name: {name} Model:"
            f" {self.model}"
        )
        state_dict = deepcopy(layer.state_dict())
        new_layer = deepcopy(layer)
        new_layer.load_state_dict(state_dict=state_dict)
        return new_layer

    @property
    def is_classification(self):
        return self.model.is_classification

    def get_features(self, input_tensor: torch.Tensor, training: bool) -> torch.Tensor:
        self.model.feature_extractor.train(training)
        return self.model.feature_extractor(input_tensor)

    def get_output(self, input_tensor: torch.Tensor, training: bool) -> torch.Tensor:
        self.model.feature_extractor.train(training)
        self.model.output_layer.train(training)
        return self.model(input_tensor)

    def get_output_from_features(
        self, input_features: torch.Tensor, training
    ) -> torch.Tensor:
        self.model.output_layer.train(training)
        return self.model.output_layer(input_features)
