import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(channels),             # instanceNorm standardizes each bright and dark img seperately
            nn.ReLU(inplace=True),

            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self,x):
        return x + self.block(x)