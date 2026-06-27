import torch.nn as nn
from models.residual_block import ResidualBlock
# from residual_block import ResidualBlock

class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # downsampling
            nn.Conv2d(3,32,kernel_size=9,stride=1,padding=4),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(128,affine=True),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),       # for 5 residual blks we need 50-80 epochs 
            ResidualBlock(128),

            # upsampling
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,3,kernel_size=9,stride=1,padding=4),
            # nn.Sigmoid()                               # output will be in [0,1]
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)                           # output will be in [0,1]