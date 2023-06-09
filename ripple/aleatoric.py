from typing import Tuple

import torch
import torch.nn as nn

from ripple.base import Base
from ripple.constants import OUTPUT_LAYER
from ripple.model import RippleModel


def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample from mean and variance tensor.

    Args:
        mu (torch.Tensor): Input Mean.
        logvar (torch.Tensor): Input log(variance).

    Returns:
        torch.Tensor: Sampled tensor from a normal distribution \
            based on the input mean and variance.
    """
    assert mu.ndim == 2
    assert logvar.ndim == 2
    epsilon = torch.normal(mean=0, std=1.0, size=mu.shape).detach()
    return mu + torch.exp(0.5 * logvar) * epsilon


def neg_log_likelihood(
    y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Negative log likelihood.

    Args:
        y (torch.Tensor): Label(integer)
        mu (torch.Tensor): Input mean tensor
        logvar (torch.Tensor): Input log(variance) tensor

    Returns:
        torch.Tensor: Negative log likelihood loss wrt label.
    """
    var = torch.exp(logvar)
    loss = logvar + torch.pow(y - mu, 2) / var
    return loss


class MVE(Base):
    """Implementation of Mean and Variance Estimation:
    https://doi.org/10.1109/ICNN.1994.374138. This implementation utilizes cross entropy
    loss for classification models and negative log likelihood loss.

    Args:
        model (ripple.model.RippleModel): Base Ripple Model
    """

    mu: nn.Module
    logvar: nn.Module

    def __init__(self, model: RippleModel) -> None:
        super().__init__(model)

        self.mu = self.copy_layer(OUTPUT_LAYER)
        self.logvar = self.copy_layer(OUTPUT_LAYER)

    def forward(
        self, x: torch.Tensor, training: bool = True, return_risk: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.get_features(x, training)
        y_hat = self.get_output_from_features(features, training)

        mu = self.mu(features)
        logvar = self.logvar(features)

        if return_risk:
            var = torch.exp(logvar)
            y_hat.aleatoric = var
            return y_hat, (mu, logvar)

        return y_hat, (mu, logvar)

    def train_forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mve loss for input data and label.

        Args:
            x (torch.Tensor): Input training data
            y (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MVE Loss, prediction for input data.
        """
        y_hat, (mu, logvar) = self.forward(x)

        loss = self.get_loss(y, mu, logvar)
        return loss, y_hat

    def get_loss(
        self, y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Implementation of MVE loss.

        Args:
            y (torch.Tensor): Ground truth label
            mu (torch.Tensor): Mean tensor
            logvar (torch.Tensor): Log(variance) tensor

        Returns:
            torch.Tensor: MVE Loss
        """
        if self.is_classification:
            z = sample(mu, logvar)
            loss = nn.CrossEntropyLoss()(z, y.long())
        else:
            loss = neg_log_likelihood(y, mu, logvar)

        return loss
