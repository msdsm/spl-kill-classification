"""
network 定義
"""

import sys
# sys.path.append("original_layers")
# from ECO_2D import ECO_2D
# from ECO_3D import ECO_3D

from eco import *

import torch
import torch.nn as nn

class myECO(nn.Module):
    def __init__(self):
        super(myECO, self).__init__()

        self.eco_2d = ECO_2D()
        self.eco_3d = ECO_3D()
        self.fc_final = nn.Linear(512, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        # print("start ECO_2d")
        x = self.eco_2d(x)
        x = x.view(-1, ns, 96, 22, 40)
        # print("start eco_3d")
        x = self.eco_3d(x)
        # print("start FC")
        x = self.fc_final(x)
        x = self.sigmoid(x)
        # print("end")
        return x


def main():
    x = torch.rand(10, 24, 3, 180, 320)
    net = myECO()
    output = net(x)
    print(output.size())
    print(output)
    print(output.dtype)

if __name__ == '__main__':
    main()
