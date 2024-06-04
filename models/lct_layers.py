import torch.nn as nn
import torch.nn.functional as F


class FiLMConvBlock(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=128, bias=True):
        super().__init__()
        self.mu_linear1 = nn.Linear(n_input, n_hidden, bias=bias)
        self.mu_linear2 = nn.Linear(n_hidden, n_output, bias=bias)
        self.sigma_linear1 = nn.Linear(n_input, n_hidden, bias=bias)
        self.sigma_linear2 = nn.Linear(n_hidden, n_output, bias=bias)

    def forward(self, x, lmbda):
        mu = self.mu_linear1(lmbda)
        mu = F.relu(mu)
        mu = self.mu_linear2(mu)

        sigma = self.sigma_linear1(lmbda)
        sigma = F.relu(sigma)
        sigma = self.sigma_linear2(sigma)

        return x * sigma[:, :, None, None] + mu[:, :, None, None]
    

class FiLMLinearBlock(FiLMConvBlock):
    def __init__(self, n_input, n_output, n_hidden=128, bias=True):
        super(FiLMLinearBlock, self).__init__(n_input, n_output, n_hidden, bias)

    def forward(self, x, lmbda):
        mu = self.mu_linear1(lmbda)
        mu = F.relu(mu)
        mu = self.mu_linear2(mu)

        sigma = self.sigma_linear1(lmbda)
        sigma = F.relu(sigma)
        sigma = self.sigma_linear2(sigma)

        return x * sigma + mu