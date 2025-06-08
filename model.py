import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, layer, functional,surrogate


import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch, C, L = x.size()
        avg = self.avg_pool(x).view(batch, C)  # [batch, C]
        channel_att = self.fc(avg).view(batch, C, 1)  # [batch, C, 1]
        return x * channel_att.expand_as(x)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = (kernel_size-1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, L]
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg, max_val], dim=1)  # [batch, 2, L]
        spatial_att = self.conv(combined)  # [batch, 1, L]
        spatial_att = self.sigmoid(spatial_att)
        return x * spatial_att


class ConvAttnModel(nn.Module):
    def __init__(self, input_features=114, num_classes=7):
        super().__init__()

  
        self.conv_layers = nn.Sequential(
            # [batch, 1, 10] -> [batch, 32, 10] 
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # [batch, 32, 10] -> [batch, 64, 5]
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            ChannelAttention(in_channels=64),

            # [batch, 64, 5] -> [batch, 128, 3]
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

      
        self.snn_layer = neuron.LIFNode(tau=2.0, v_threshold=1.0, step_mode='m')  # change surrogate function here

      
        self.spatial_attn = SpatialAttention(kernel_size=3)

       
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 128, 1]

      
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes))

    def forward(self, x):

        x = x.unsqueeze(1)  # [batch, 1, 10]


        x = self.conv_layers(x)  # [batch, 128, 3]


        x = x.permute(2,0,1) # [3, batch, 128]

        functional.reset_net(self.snn_layer)
        x = self.snn_layer(x) # [3, batch, 128]

        x =x.permute(1,2,0) # [batch, 128, 3]

        x = self.spatial_attn(x)  # [batch, 128, 3]


        x = self.global_pool(x)  # [batch, 128, 1]
        x = x.flatten(1)         # [batch, 128]

        x = self.classifier(x)   # [batch, num_classes]
        return x
    

