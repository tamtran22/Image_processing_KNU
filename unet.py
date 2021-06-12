import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Bilinear

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, device=None):
        super().__init__()
        self.device = device
        self.input_layer = DoubleConv(in_channels, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024 // 2)
        self.up1 = UpLayer(1024, 512 // 2)
        self.up2 = UpLayer(512, 256 // 2)
        self.up3 = UpLayer(256, 128 // 2)
        self.up4 = UpLayer(128, 64)
        self.output_layer = OutLayer(64, out_channels)
        # self.to(self.device)

    def forward(self, input):
        x1 = self.input_layer(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.output_layer(x)
        return output



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.double_conv(input)



class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_maxpool = nn.Sequential(
            nn.MaxPool2d((2,2)),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, input):
        return self.conv_maxpool(input)



class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, input1, input2):
        input1 = self.up(input1)
        print(input1.shape)
        dx = input2.size()[3] - input2.size()[3]
        dy = input2.size()[2] - input2.size()[2]
        input1 = F.pad(
            input1,
            [dx//2, dx - dx//2, dy, dy-dy//2] 
        )
        input = torch.cat([input2, input1], dim=1)
        return self.conv(input)



class OutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, input):
        return self.conv(input)

if __name__ == '__main__':
    print('Testing Unet...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(
        in_channels=3,
        out_channels=1,
        device=device
    )

    image = torch.rand((3,3,512,512))
    output = unet(image)
    print(output.shape)