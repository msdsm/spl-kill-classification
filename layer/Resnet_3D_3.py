"""
Resnet_3D_3 module
input : 96, f, h, w
output : 128, f, h, w
"""

import torch
import torch.nn as nn

class Resnet_3D_3(nn.Module):
    def __init__(self):
        super(Resnet_3D_3, self).__init__()

        self.res3a_2 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3a_bn = nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)

        self.res3b_2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res3b_bn = nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.res3a_2(x) # 残渣
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_1_relu(out)
        out = self.res3b_2(out)

        out += residual # 残渣接続
        
        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out

    def size_confirmation(self, x):
        print(x.size())
        residual = self.res3a_2(x) # 残渣
        print(residual.size())
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        print(out.size())
        out = self.res3b_1_bn(out)
        out = self.res3b_1_relu(out)
        out = self.res3b_2(out)
        print(out.size())

        out += residual # 残渣接続
        
        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out

def main(): # channnel,height,width,netを変える
    batch_size = 10
    channel = 96
    frame = 24
    height = 22
    width = 40
    net = Resnet_3D_3() # network
    x = torch.rand(batch_size, channel, frame, height, width) # batch_size, channel, frame, height, width
    y = net.size_confirmation(x)
    print(y.size())

if __name__ == '__main__':
    main()

