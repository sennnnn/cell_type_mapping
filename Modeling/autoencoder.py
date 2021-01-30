import torch
import torch.nn as nn

from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super(AutoEncoder, self).__init__()
        self.input_layer = nn.Linear(in_channels, hidden_channels, bias=False)
        self.input_batch_norm = nn.BatchNorm1d(hidden_channels, momentum=0.9, eps=1e-5)
        
        self.encode_layer = nn.Linear(hidden_channels, hidden_channels // 4, bias=False)
        self.encode_batch_norm = nn.BatchNorm1d(hidden_channels // 4, momentum=0.9, eps=1e-5)
        
        self.decode_layer = nn.Linear(hidden_channels // 4, hidden_channels, bias=False)
        self.decode_batch_norm = nn.BatchNorm1d(hidden_channels, momentum=0.9, eps=1e-5)
        
        self.output_layer = nn.Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x):
        o = self.input_layer(x)
        o = self.input_batch_norm(o)
        o = self.relu(o)

        o = self.encode_layer(o)
        o = self.encode_batch_norm(o)
        o = self.relu(o)

        o = self.decode_layer(o)
        o = self.decode_batch_norm(o)
        o = self.relu(o)

        o = self.output_layer(o)

        return o
