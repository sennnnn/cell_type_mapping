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


class GNN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=1024):
        super(GNN_unit, self).__init__()
        self.linear_Q = nn.Linear(in_channels, middle_channels, bias=False)
        self.batch_norm_Q = nn.BatchNorm1d(middle_channels, momentum=0.9, eps=1e-5)
        self.linear_K = nn.Linear(in_channels, middle_channels, bias=False)
        self.batch_norm_K = nn.BatchNorm1d(middle_channels, momentum=0.9, eps=1e-5)
        self.linear_V = nn.Linear(in_channels, middle_channels, bias=False)
        self.batch_norm_V = nn.BatchNorm1d(middle_channels, momentum=0.9, eps=1e-5)
        
        self.output_layer = nn.Linear(middle_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x):
        Q = self.linear_Q(x)
        Q = self.batch_norm_Q(Q)
        Q = self.relu(Q)
        K = self.linear_K(x)
        K = self.batch_norm_K(K)
        K = self.relu(K)
        V = self.linear_V(x)
        V = self.batch_norm_V(V)
        V = self.relu(V)

        o = Q * K
        # o = torch.mm(Q, K.transpose(1, 0))
        o = self.softmax(o)
        # o = torch.mm(o, V)
        o = o * V

        o = self.output_layer(o)
        o = self.relu(o)

        return o


class GNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, middle_channels=1024, n=4):
        super(GNN, self).__init__()
        self.GNN_unit_list = nn.ModuleList()
        self.n = n
        for i in range(n):
            if i == (n-1):
                unit = GNN_unit(middle_channels, out_channels, hidden_channels)
            elif i == 0:
                unit = GNN_unit(in_channels, middle_channels, hidden_channels)
            else:
                unit = GNN_unit(middle_channels, middle_channels, hidden_channels)
            self.GNN_unit_list.append(unit)
    
    def forward(self, x):
        o = x
        for i in range(self.n):
            o = self.GNN_unit_list[i](o)
        return o
