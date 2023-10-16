import os
import sys

import torch
from torch import nn, optim


class Model(nn.Module):
    def __init__(self, params):
        pass

    def _select_optimizer(self):
        if self.optim == 'adamw':
            model_optim = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim == 'adam':
            model_optim = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == 'sgd':
            model_optim = optim.SGD(self.parameters(), lr=self.lr)
        return model_optim

