import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional,surrogate

class HybridSNN(nn.Module):
    def __init__(self, input_length=4097, num_classes=5, T=16):
        super().__init__()
        self.T = T  # 时间步长
        
        # 编码器：将原始信号转换为脉冲序列
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        # SNN处理部分（全部使用支持多步的层）
        self.snn = nn.Sequential(
            layer.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, step_mode='m'),
            neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m',surrogate_function=surrogate.Sigmoid_Simulation()),
            layer.MaxPool1d(3, stride=2, padding=1, step_mode='m'),
            
            layer.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, step_mode='m'),
            neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m', surrogate_function=surrogate.Sigmoid_Simulation()),
            layer.Dropout(0.3, step_mode='m'),  # 使用脉冲层Dropout
            
            layer.AdaptiveAvgPool1d(1, step_mode='m'),
            layer.Flatten(step_mode='m')
        )
        
        # 全连接分类器
        self.classifier = nn.Sequential(
            layer.Linear(64, 32, step_mode='m'),
            neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='m', surrogate_function=surrogate.Sigmoid_Simulation()),
            layer.Dropout(0.5, step_mode='m'),
            layer.Linear(32, num_classes, step_mode='m')
        )

    def forward(self, x):
        # 输入形状: [batch, 4097]
        x = x.unsqueeze(1)  # [batch, 1, 4097]
        
        # 特征提取
        x = self.encoder(x)  # [batch, 16, 512]
        
        # 添加时间维度并扩展
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, batch, 16, 512]
        
        # 处理多步时序
        x = self.snn(x)  # 输出形状 [T, batch, 64]
        
        # 分类处理
        x = self.classifier(x)  # 输出形状 [T, batch, num_classes]
        
        # 时域累积
        x = torch.mean(x, dim=0)  # [batch, num_classes]
        return x


    