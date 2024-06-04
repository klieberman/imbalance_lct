import torch.optim as optim
from optimizers.sam import SAM


_available_optimizers = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'sam': SAM
}


def available_optimizers():
    return _available_optimizers
