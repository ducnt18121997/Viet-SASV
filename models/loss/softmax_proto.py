#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from models.loss.softmax import Softmax
from models.loss.angleproto import AngleProto


class SoftmaxProto(nn.Module):

    def __init__(self, **kwargs):
        super(SoftmaxProto, self).__init__()

        self.test_normalize = True

        self.softmax = Softmax(**kwargs)
        self.angleproto = AngleProto(**kwargs)

        print("Initialised SoftmaxPrototypical Loss")

    def forward(self, x: torch.Tensor, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.softmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(2)
        )

        nlossP, _ = self.angleproto(x, None)

        return (nlossS + nlossP), prec1
