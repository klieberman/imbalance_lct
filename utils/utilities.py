import os
import os.path as osp
import numpy as np

import torch
        

def get_device():
    cuda_available = torch.cuda.is_available()
    print(f"Cuda available? {cuda_available}.")
    if cuda_available:
        dev = "cuda:0"
    else:
        dev = "cpu"
    return dev


def makedirs_if_needed(path):
    if not osp.exists(path):
        os.makedirs(path)
    return path


def get_list_from_tuple_or_scalar(x):
    if isinstance(x, int):
        return [x]
    else:
        return list(x)
    
def print_model_param_nums(model=None):
    total = sum([param.nelement() for param in model.parameters()])
    print(f'Model has {total/1_000_000:.1f}M parameters')
    return total
    
        