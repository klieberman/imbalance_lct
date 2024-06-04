import torchvision.datasets
import torchvision.transforms as transforms
from .imbalanced import ImbalancedCifar

NAME = "CIFAR10"
CLASS = torchvision.datasets.CIFAR10
IMBALANCED_CLASS = ImbalancedCifar

MEAN = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
STD = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

LABEL_MAP = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
