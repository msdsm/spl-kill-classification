"""
ECO 2dmodule
"""

import torch
import torch.nn as nn

from BasicConv import BasicConv
from InceptionA import InceptionA
from InceptionB import InceptionB
from InceptionC import InceptionC

class ECO_2D(nn.Module):
    def __init__(self):
        super(ECO_2D, self).__init__()

        # BasicConv module
        self.basic_conv = BasicConv()
        # Inception module
        self.inception_a = InceptionA()
        self.inception_b = InceptionB()
        self.inception_c = InceptionC()

    def forward(self, x):
        """
        input x : torch.Size([batch_num, 3, 224, 224])
        """
        out = self.basic_conv(x)
        out = self.inception_a(out)
        out = self.inception_b(out)
        out = self.inception_c(out)

        return out
    
    def size_confirmation(self, x):
        print(x.size())
        out = self.basic_conv(x)
        print(out.size())
        out = self.inception_a(out)
        print(out.size())
        out = self.inception_b(out)
        print(out.size())
        out = self.inception_c(out)

        return out

def main(): # channnel,height,width,netを変える
    batch_size = 10
    channel = 3
    height = 180
    width = 320
    net = ECO_2D() # network
    print(net.train())
    x = torch.rand(batch_size, channel, height, width)
    y = net.size_confirmation(x)
    print(y.size())

if __name__ == '__main__':
    main()