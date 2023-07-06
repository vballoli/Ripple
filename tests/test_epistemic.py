import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ripple.epistemic import DropoutEpistemic
from ripple.model import RippleModel


def get_data():
    x = torch.from_numpy(np.random.uniform(-4, 4, (16384, 1)))
    x_val = torch.linspace(-6, 6, 2048).reshape(-1, 1)

    y = x ** 3 / 10
    y_val = x_val ** 3 / 10

    # add noise to y
    y += torch.normal(0, 0.2, (16384, 1))
    y_val += torch.normal(0, 0.2, (2048, 1))

    # add greater noise in the middle to reproduce that plot from
    # the 'Deep Evidential Regression' paper
    x = torch.vstack((x, torch.normal(1.5, 0.3, size=(4096, ))\
                      .unsqueeze(-1))).float()
    y = torch.vstack((y, torch.normal(1.5, 0.6, size=(4096, ))\
                      .unsqueeze(-1))).float()

    x_val = torch.vstack((x_val, torch.normal(1.5, 0.3, size=(256,))\
                          .unsqueeze(-1))).float()
    y_val = torch.vstack((y_val, torch.normal(1.5, 0.6, size=(256,))\
                          .unsqueeze(-1))).float()

    def _get_ds(x: torch.Tensor, y: torch.Tensor,
                batch_size: int, shuffle: bool=True):
        ds = TensorDataset(x, y)
        ds = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        return ds

    ds_train = _get_ds(x, y, 256, True)
    ds_val = _get_ds(x_val, y_val, 256, False)

    # return x, y, x_val, y_val as well to test models on
    # both batched and not batched inputs
    return ds_train, ds_val, x, y, x_val, y_val

def test_dropout_epistemic():
    feature_layers = [torch.nn.Linear(1, 16), torch.nn.Linear(16, 32),
                      torch.nn.Linear(32, 64), torch.nn.Linear(64, 32),
                      torch.nn.Linear(32, 16)]
    feature_model = torch.nn.Sequential(*feature_layers)

    output_layer = torch.nn.Linear(16, 1)

    model = RippleModel(feature_model, output_layer, (1, ), False)
    dropout_epistemic = DropoutEpistemic(model, 20)
    model.replace_modules(0.1, 'relu', 0.1, 'identity')
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    ds_train, ds_val, x, y, x_val, y_val = get_data()

    epochs = 30
    for epoch in range(epochs):
        avg_loss = []
        for i, data in enumerate(ds_train):
            opt.zero_grad()
            X_batch, y_batch = data
            y_hat = model(X_batch, True)

            loss = torch.nn.MSELoss()(y_hat, y_batch)
            loss.backward()
            avg_loss.append(loss.item())
            opt.step()

    for i, data in enumerate(ds_val):
        X_batch, y_batch = data
        _, (_, epistemic_uncertainty) = dropout_epistemic(X_batch, False)
