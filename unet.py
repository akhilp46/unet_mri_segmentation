import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder1 = DownSample(in_channels, 64)
        self.encoder2 = DownSample(64, 128)
        self.encoder3 = DownSample(128, 256)
        self.encoder4 = DownSample(256, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)

        self.decoder1 = UpSample(1024, 512)
        self.decoder2 = UpSample(512, 256)
        self.decoder3 = UpSample(256, 128)
        self.decoder4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
    
    def forward(self, x):
        down1, p1 = self.encoder1(x)
        down2, p2 = self.encoder2(p1)
        down3, p3 = self.encoder3(p2)
        down4, p4 = self.encoder4(p3)

        b = self.bottle_neck(p4)
        
        up1 = self.decoder1(b, down4)
        up2 = self.decoder2(up1, down3)
        up3 = self.decoder3(up2, down2)
        up4 = self.decoder4(up3, down1)

        return self.out(up4)

# Debuging step
if __name__ == "__main__":
    # double_conv = DoubleConv(256, 256)
    # print(double_conv)

    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    print(model)
    output = model(input_image)
    print(output.size()) # Expected size: (1, 10, 512, 512)