import torchvision.datasets
import torchvision.transforms as transforms
from .imbalanced import ImbalancedImageFolder

NAME = "PetFinder"
CLASS = torchvision.datasets.ImageFolder
IMBALANCED_CLASS = ImbalancedImageFolder

MEAN = (0.48832044, 0.45507638, 0.41696083)
STD = (0.22954217, 0.22502467, 0.22531743)

TRAIN_TRANSFORM = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

VAL_TRANSFORM = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

LABEL_MAP = {
    0: 'cat',
    1: 'dog'
}