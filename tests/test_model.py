import torch

from ripple.model import RippleModel


def test_replace_dropout():
    feature_extractor = [torch.nn.Conv2d(3, 32, 3), torch.nn.Conv2d(32, 16, 3),
                         torch.nn.Conv2d(16, 8, 3), torch.nn.Flatten()]
    feature_model = torch.nn.Sequential(*feature_extractor)

    output_model = torch.nn.Linear(119072, 1)
    model = RippleModel(feature_model, output_model, (3, 128, 128), True)

    model.replace_modules(0.5, "relu", 0.5, "softmax")

    dropout_counter = 0
    relu_counter = 0
    for name, mod in model.feature_extractor.named_modules():
        if isinstance(mod, torch.nn.modules.dropout._DropoutNd):
            dropout_counter += 1
        if isinstance(mod, torch.nn.ReLU):
            relu_counter += 1

    assert dropout_counter == 3, feature_model
    assert relu_counter == 3, feature_model

    dropout_counter = 0
    for name, mod in model.output_layer.named_modules():
        if isinstance(mod, torch.nn.modules.dropout._DropoutNd):
            dropout_counter += 1

    assert dropout_counter == 1, output_model
