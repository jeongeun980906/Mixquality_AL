import torch
import torch.nn.functional as F
import torch.nn as nn

from core.DDU.sn_cnv import spectral_norm_conv
from core.DDU.sn_fc import spectral_norm_fc

class DDU_CNN3(nn.Module):
    def __init__(self, input_channel=1, n_outputs=10, dropout_rate=0.25,mod=True,
                        coeff=3,n_power_iterations=1):
        self.dropout_rate = dropout_rate
        super(DDU_CNN3, self).__init__()

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(conv, coeff, shapes, n_power_iterations)

            return wrapped_conv

        self.c1=wrapped_conv(28,input_channel, 32,kernel_size=3,stride=1)
        self.c2=wrapped_conv(14,32,64,kernel_size=3,stride=1)
        self.c3=wrapped_conv(7,64,128,kernel_size=3,stride=1)
        self.linear1=nn.Linear(1152, 128)
        self.linear2=nn.Linear(128, 64)
        self.linear3=nn.Linear(64, n_outputs)
    def feature(self,h):
        h=F.relu(self.c1(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h=F.relu(self.c2(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h=F.relu(self.c3(h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        
        h = h.view(h.size(0), -1)
        return h

    def to_logit(self,h):
        h=F.relu(self.linear1(h))
        h=F.relu(self.linear2(h))
        logit=self.linear3(h)
        return logit

    def forward(self, x):
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
            if isinstance(m, nn.Linear):  # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)