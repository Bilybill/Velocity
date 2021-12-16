import torch
import torch.nn as nn
from .dense_block import DenseBlock
from .resize_features import Up,Down


class MRUDSR(nn.Module):

    def __init__(self, in_channels, out_channels,
                 mid_channels=16, num_blocks=30):
        super().__init__()

        self.num_blocks = num_blocks

        self.first_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        self.down1 = Down(mid_channels) # (conv+leaky Relu)*2
        self.down2 = Down(mid_channels*2)
        self.down3 = Down(mid_channels*4)
        self.down4 = Down(mid_channels*8)

        self.zip = nn.Conv2d(mid_channels*16, mid_channels*4, 1)

        self.denses = nn.ModuleList()

        channels = mid_channels*4
        
        for i in range(num_blocks):
            dense = DenseBlock(channels, mid_channels*4)
            channels += mid_channels*4
            self.denses.append(dense)

        self.dezip = nn.Conv2d(mid_channels*4, mid_channels*16, 3, 1, 1)
        self.up4 = Up(mid_channels*8)
        self.up3 = Up(mid_channels*4)
        self.up2 = Up(mid_channels*2)
        self.up1 = Up(mid_channels)
        self.last_conv = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

    def forward(self, x):

        x0 = self.first_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x4 = self.zip(x4)
        
        for i in range(self.num_blocks):
            x_dense = self.denses[i](x4)
            if i < self.num_blocks-1:
                x4 = torch.cat((x_dense, x4), 1)
            else:
                x4 = x_dense

        x4 = self.dezip(x4)
        x3 = self.up4(low=x4, high=x3)
        x2 = self.up3(low=x3, high=x2)
        x1 = self.up2(low=x2, high=x1)
        x0 = self.up1(low=x1, high=x0)
        
        return self.last_conv(x0)

if __name__ == '__main__':
    x = torch.rand((2, 3, 266, 788))
    mrudsr = MRUDSR(3, 1, 16, 8)
    y = mrudsr(x)
    print(y.shape)
