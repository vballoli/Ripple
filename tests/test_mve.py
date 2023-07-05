import numpy as np
import torch
from capsa.utils import generate_moon_data_classification
from torch.utils.data import DataLoader, TensorDataset

from ripple.aleatoric import MVE
from ripple.model import RippleModel
from ripple.models import SimpleFF


def test_mve():
    X, y = generate_moon_data_classification()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)

    model = SimpleFF(2, [8, 8, 8, 8, 2])

    model = RippleModel(model.feature_extractor, model.output_layer, (2,), True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    mve = MVE(model)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 20
    for epoch in range(epochs):
        running_loss = 0
        running_mse_loss = 0
        running_ce_loss = 0

        for i, data in enumerate(dataloader):
            opt.zero_grad()
            X_batch, y_batch = data
            mve_loss, y_hat = mve.train_forward(X_batch, y_batch)
            ce_loss = torch.nn.CrossEntropyLoss()(y_hat, y_batch.long())
            loss = mve_loss + ce_loss

            loss.backward()
            opt.step()
            running_loss += loss.item()
            running_mse_loss += mve_loss.item()
            running_ce_loss += ce_loss.item()

    max_points = torch.from_numpy(np.load("tests/data/mve_max_points.npy")).float()
    min_points = torch.from_numpy(np.load("tests/data/mve_min_points.npy")).float()

    max_results = mve(max_points, return_risk=True)[0].aleatoric.mean(-1)
    min_results = mve(min_points, return_risk=True)[0].aleatoric.mean(-1)

    assert max_results.mean() > min_results.mean()
