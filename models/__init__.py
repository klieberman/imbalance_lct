from torchvision.models import resnet18, resnet34, resnet50, resnext50_32x4d
from .resnet_cifar import resnet32
from .lct_models import resnet32LCT_last, resnet32LCT_penultimate, resnet32LCT_both, resnext50_32x4d_lct

_available_models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet32': resnet32,
    'resnet32lct_last': resnet32LCT_last,
    'resnet32lct_penultimate': resnet32LCT_penultimate,
    'resnet32lct_both': resnet32LCT_both,
    'resNext50-32x4d': resnext50_32x4d,
    'resNext50-32x4d_lct': resnext50_32x4d_lct,
}

_lct_models = ['resnet32lct_last', 'resnet32lct_penultimate', 'resnet32lct_both', 'resNext50-32x4d_lct']


def available_models():
    return _available_models

def lct_models():
    return _lct_models