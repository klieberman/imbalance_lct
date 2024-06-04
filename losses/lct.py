import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from losses.vs import get_omega_list, get_binary_omega_list, get_delta_list, get_iota_list

    
class VSLossLCT(nn.Module):    
    def __init__(self, class_dist, device, omega, gamma, tau):
        super().__init__()
        self.device = device
        self.class_dist = class_dist

        if omega is None:
            self.omega_list = None
        elif len(class_dist) == 2:
            self.omega_list = get_binary_omega_list(omega, device)
        else:
            print(f'Warning: Hyperparameter Omega is not being used since this is a"\
                  " multi-class dataset.')
            self.omega_list = get_omega_list(class_dist, device, k=1)
        self.delta_list = get_delta_list(self.class_dist, gamma, self.device) if gamma is not None else None
        self.iota_list = get_iota_list(self.class_dist, tau, self.device) if tau is not None else None
        

    def forward(self, x, target, hypers):
        i = 0
        if self.omega_list is not None:
            weight = self.omega_list
        else:
            if len(self.class_dist) == 2:
                weight = get_binary_omega_list(hypers[i].item(), self.device)
            else:
                print(f'Warning: Hyperparameter Omega is not being used since this is a"\
                    " multi-class dataset.')
                weight = get_omega_list(self.class_dist, self.device, k=1)
            i += 1
        if self.delta_list is not None:
            delta_list = self.delta_list
        else:
            delta_list = get_delta_list(self.class_dist, hypers[i].item(), self.device)
            i += 1
        if self.iota_list is not None:
            iota_list = self.iota_list
        else:
            iota_list = get_iota_list(self.class_dist, hypers[i].item(), self.device)
            i += 1

        output = x / delta_list + iota_list

        return F.cross_entropy(output, target, weight=weight)