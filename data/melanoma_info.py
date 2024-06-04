import torchvision
import torchvision.transforms as transforms

from .imbalanced import ImbalancedImageFolder
    
NAME = "melanoma"
CLASS = torchvision.datasets.ImageFolder
IMBALANCED_CLASS = ImbalancedImageFolder

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)]
    )

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)]
    )

# This is the original label (ie, the one from ImageFolder), NOT THE IMBALANCED ONE
LABEL_MAP = {
    0: 'melanoma',
    1: 'no-melanoma'
}
