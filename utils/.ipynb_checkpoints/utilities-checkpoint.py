import os
import os.path as osp

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
    
    
def label_rows_and_columns(axes, rows, cols):
    # Cite: https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    pad = 5 # in points
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')