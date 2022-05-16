from torch import nn
import torch as t


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x + t.rand_like(x) * self.sigma
        else:
            return x
