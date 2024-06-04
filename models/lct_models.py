from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torchvision.models.resnet import ResNet, ResNeXt50_32X4D_Weights, Bottleneck
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._api import register_model

from models.resnet_cifar import BasicBlockCifar, ResNet_s
from models.lct_layers import FiLMConvBlock, FiLMLinearBlock


class ResNet_s_LCT(ResNet_s):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False,
                 penultimate=False, last=False, n_lambdas=1):
        super().__init__(block, num_blocks, num_classes, use_norm)

        self.penultimate_film_layer = FiLMConvBlock(n_lambdas, 64) if penultimate else None
        self.last_film_layer = FiLMLinearBlock(n_lambdas, self.num_classes) if last else None

    def forward(self, x, lmbda):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        if self.penultimate_film_layer is not None:
            out = self.penultimate_film_layer(out, lmbda)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.last_film_layer is not None:
            out = self.last_film_layer(out, lmbda)
        return out
    

class ResNetLCT(ResNet):
    def __init__(self,
        block: Type[Union[BasicBlockCifar, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_hidden: int = 128,
        n_lambdas = 1
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.film_layer = FiLMLinearBlock(n_lambdas, num_classes, n_hidden=n_hidden)


    def _forward_impl(self, x: Tensor, lmbda: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.film_layer(x, lmbda)

        return x

    def forward(self, x: Tensor, lmbda: Tensor) -> Tensor:
        return self._forward_impl(x, lmbda)
    
    def load_common_params(self, state_dict):
        for name, param in state_dict.items():
            if name not in self.state_dict():
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            self.state_dict()[name].copy_(param)


def resnet32LCT_penultimate(num_classes=10, use_norm=False, n_lambdas=1):
    return ResNet_s_LCT(BasicBlockCifar, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, 
                        penultimate=True, last=False, n_lambdas=n_lambdas)

def resnet32LCT_last(num_classes=10, use_norm=False):
    return ResNet_s_LCT(BasicBlockCifar, [5, 5, 5], num_classes=num_classes, use_norm=use_norm,
                        penultimate=False, last=True)

def resnet32LCT_both(num_classes=10, use_norm=False):
    return ResNet_s_LCT(BasicBlockCifar, [5, 5, 5], num_classes=num_classes, use_norm=use_norm,
                        penultimate=True, last=True)


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d_lct(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetLCT(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if weights is not None:
        model.load_common_params(weights.get_state_dict(progress=progress)) #, check_hash=True
    
    return model
