import torch
from torch import nn
from torchvision import models

####
def get_resnet50(n, weights_path=None):
    resnet = models.resnet50()
    if weights_path is not None:
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
    for parameter in resnet.parameters():
        parameter.requires_grad = False
    cin = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(cin, 1024, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(1024, n, bias=False)
    )

    return resnet
