import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CNN3_base(nn.Module):
    def __init__(self, input_channel=1, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN3_base, self).__init__()
        self.c1=nn.Conv2d(input_channel, 32,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(1152, 128)
        self.linear2=nn.Linear(128, 64)
        self.linear3=nn.Linear(64, n_outputs)

    def forward(self, x,):
        h=x
        h=F.relu(self.c1(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h=F.relu(self.c2(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h=F.relu(self.c3(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h = h.view(h.size(0), -1)
        h=F.relu(self.linear1(h))
        h=F.relu(self.linear2(h))
        logit=self.linear3(h)
        return logit
        
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

class CNN3(nn.Module):
    def __init__(self, input_channel=1, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN3, self).__init__()
        self.c1=nn.Conv2d(input_channel, 32,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(1152, 128)
        self.linear2=nn.Linear(128, 64)
        self.linear3=nn.Linear(64, n_outputs)
        self.dropout = nn.Dropout2d(p=self.dropout_rate)

    def forward(self, x,):
        h=x
        h=F.relu(self.dropout(self.c1(h)))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h=F.relu(self.dropout(self.c2(h)))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h=F.relu(self.dropout(self.c3(h)))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h = h.view(h.size(0), -1)
        h=F.relu(self.dropout(self.linear1(h)))
        h=F.relu(self.dropout(self.linear2(h)))
        logit=self.linear3(h)
        return logit
        
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)