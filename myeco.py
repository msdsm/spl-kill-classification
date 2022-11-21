"""
network 定義
"""

import sys
sys.path.append("layer")
from ECO_2D import ECO_2D
from ECO_3D import ECO_3D

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
        out = x.view(-1, c, h, w)
        out = self.eco_2d(out)
        out = out.view(-1, ns, 96, 22, 40)
        out = self.eco_3d(out)
        out = self.fc_final(out)
        out = self.sigmoid(out)
        return out


def main():
    x = torch.rand(10, 24, 3, 180, 320)
    net = myECO()
    output = net(x)
    print(output.size())
    print(output)
    print(output.dtype)

if __name__ == '__main__':
    main()
