# model_def.py — satu-satunya tempat definisi arsitektur
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.pad, dilation=dilation)
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.pad] if self.pad != 0 else x

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()
    def forward(self, x):
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

class HESS_TCN_v2(nn.Module):
    def __init__(self, num_inputs=5, channels=(64, 128, 128),
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []; ic = num_inputs
        for i, oc in enumerate(channels):
            layers.append(TCNBlock(ic, oc, kernel_size, 2**i, dropout))
            ic = oc
        self.tcn    = nn.Sequential(*layers)
        self.fc1    = nn.Linear(channels[-1], 64)
        self.relu1  = nn.ReLU()
        self.drop_fc = nn.Dropout(dropout)
        self.fc2    = nn.Linear(64, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.drop_fc(self.relu1(self.fc1(x)))
        return self.fc2(x)