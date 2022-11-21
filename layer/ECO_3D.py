"""
ECO_3D module
input :24, 96, 22, 40
output : 512
"""

import torch
import torch.nn as nn
from Resnet_3D_3 import Resnet_3D_3
from Resnet_3D_4 import Resnet_3D_4
from Resnet_3D_5 import Resnet_3D_5

class ECO_3D(nn.Module):
    def __init__(self):
        super(ECO_3D, self).__init__()

        # 3D Resnet module
        self.res_3d_3 = Resnet_3D_3()
        self.res_3d_4 = Resnet_3D_4()
        self.res_3d_5 = Resnet_3D_5()

        # Grobal Average Pooling
        self.global_pool = nn.AvgPool3d(kernel_size=(6, 6, 10), stride=1, padding=0)

    def forward(self, x):
        out = torch.transpose(x, 1, 2) # tensorの順番入れ替え
        out = self.res_3d_3(out)
        out = self.res_3d_4(out)
        out = self.res_3d_5(out)
        out = self.global_pool(out)

        # tensor size 変更
        # torch.Size([batch_num, 512, 1, 1]) -> torch.Size([batch_num, 512])
        out = out.view(out.size()[0], out.size()[1])

        return out

    def size_confirmation(self, x):
        print(x.size())
        out = torch.transpose(x, 1, 2) # tensorの順番入れ替え
        print(out.size())
        out = self.res_3d_3(out)
        print(out.size())
        out = self.res_3d_4(out)
        print(out.size())
        out = self.res_3d_5(out)
        print(out.size())
        out = self.global_pool(out)
        print(out.size())

        # tensor size 変更
        # torch.Size([batch_num, 512, 1, 1]) -> torch.Size([batch_num, 512])
        out = out.view(out.size()[0], out.size()[1])
        print(out.size())

        return out


def main(): # channnel,height,width,netを変える
    batch_size = 10
    channel = 96
    frame = 24
    height = 22
    width = 40
    net = ECO_3D() # network
    x = torch.rand(batch_size, frame, channel, height, width)
    y = net.size_confirmation(x)
    print(y.size())

if __name__ == '__main__':
    main()