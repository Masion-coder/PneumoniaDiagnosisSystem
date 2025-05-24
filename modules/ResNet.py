import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        out = out1 + out2
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels: int=1):
        super(ResNet, self).__init__()

        # (32, 159, 159)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )


        self.layers = nn.ModuleList()

        # (64, 79, 79)
        self.layers.append(ResBlock(32, 64, stride=2))

        # (128, 39, 39)
        self.layers.append(ResBlock(64, 128, stride=2))

        # (256, 19, 19)
        self.layers.append(ResBlock(128, 256, stride=2))

        # # (512, 9, 9)
        self.layers.append(ResBlock(256, 512, stride=2))

        # (1024, 4, 4)
        # self.layers.append(ResBlock(512, 1024, stride=2))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(512, 2)


    def forward(self, x):
        out = self.conv(x)


        for layer in self.layers:
            out = layer(out)
        
        out = self.dropout(out)

        out = self.pool(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out
