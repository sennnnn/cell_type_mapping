import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super(AutoEncoder, self).__init__()
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.encode_layer = nn.Linear(hidden_channels, hidden_channels // 4)
        self.decode_layer = nn.Linear(hidden_channels // 4, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        o = self.input_layer(x)
        o = self.encode_layer(o)
        o = self.decode_layer(o)
        o = self.output_layer(o)

        return o
