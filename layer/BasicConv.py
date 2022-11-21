"""
ECOの2Dnetモジュールの最初のモジュール
入力
tensor.Size(3, h, w)
出力
tensor.Size(192, h/8, w/8)
"""

import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace=True)
        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    
    def forward(self, x):
        out = self.conv1_7x7_s2(x)
        out = self.conv1_7x7_s2_bn(out)
        out = self.conv1_relu_7x7(out)
        out = self.pool1_3x3_s2(out)
        out = self.conv2_3x3_reduce(out)
        out = self.conv2_3x3_reduce_bn(out)
        out = self.conv2_relu_3x3_reduce(out)
        out = self.conv2_3x3(out)
        out = self.conv2_3x3_bn(out)
        out = self.conv2_relu_3x3(out)
        out = self.pool2_3x3_s2(out)
        return out

    def size_confirmation(self, x):
        print(x.size())
        out = self.conv1_7x7_s2(x)
        print(out.size())
        out = self.conv1_7x7_s2_bn(out)
        print(out.size())
        out = self.conv1_relu_7x7(out)
        print(out.size())
        out = self.pool1_3x3_s2(out)
        print(out.size())
        out = self.conv2_3x3_reduce(out)
        print(out.size())
        out = self.conv2_3x3_reduce_bn(out)
        print(out.size())
        out = self.conv2_relu_3x3_reduce(out)
        print(out.size())
        out = self.conv2_3x3(out)
        print(out.size())
        out = self.conv2_3x3_bn(out)
        print(out.size())
        out = self.conv2_relu_3x3(out)
        print(out.size())
        out = self.pool2_3x3_s2(out)
        print(out.size())
        return out

def main():
    batch_size = 100
    channel = 3
    height = 180
    width = 320
    net = BasicConv()
    x = torch.rand(batch_size, channel, height, width)
    y = net.size_confirmation(x)
    print(y.size())

if __name__ == '__main__':
    main()
