# coding=UTF-8


# Multiple landmark detection in 3D ultrasound images of fetal head
# Network training
#
# Reference
# Fast Multiple Landmark Localisation Using a Patch-based Iterative Network
# https://arxiv.org/abs/1806.06987
#
# Code EDITED BY: Xingchen Xiao
# ==============================================================================


"""CNN model architecture in pytorch"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms

from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import truncnorm



class VisitNet(torch.nn.Module):

    """
    Args:
        input_x: an input tensor with the dimensions (N_examples, width, height, channel).
        w: weights
        b: bias
        input_dim: number of input feature maps
        output_dim: number of output feature maps
    """

    def __init__(self, clas_outputs, reg_outputs, keep_prob):
        super(VisitNet, self).__init__()

        # number of output neurons in the classification layer. (before softmax) (6 neurons)
        self.num_output_cla = clas_outputs

        self.keep_prob = torch.tensor(keep_prob)
        # number of output neurons in the regression layer. (1 neuron)
        self.num_output_reg = reg_outputs

        self.layer1 =   nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer2 =   nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer3 =   nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer4 =   nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer5 =   nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )


        self.layer_cla = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p = self.keep_prob),
            nn.Linear(1024,1024),
            nn.Dropout(p = self.keep_prob),
            nn.Linear(1024, self.num_output_cla)
        )

        self.layer_reg = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p = self.keep_prob),
            nn.Linear(1024, 1024),
            nn.Dropout(p = self.keep_prob),
            nn.Linear(1024, self.num_output_reg)
        )

        self.dropout = nn.Dropout(p=self.keep_prob)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_x):

        out = self.layer1(input_x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        b, c, h, w = out.size()

        fc_input_dim = c * h * w

        out_flat = out.view(b,-1)


        y_clas = self.layer_cla(out_flat)

        y_reg = self.layer_reg(out_flat)


        return y_clas, y_reg, self.keep_prob
