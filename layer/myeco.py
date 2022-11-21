# network定義

import torch.nn as nn

class myECO(nn.Module):
    def __init__(self):
        super(ECO, self).__init__()
        self.fc1 = nn.Linear(24*3*180*320, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        print(x.size())
        out = self.fc1(x)
        out = self.sigmoid(out)
        return out