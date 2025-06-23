import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)

        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip):
        x = self.upscale(x)

        if x.size() != skip.size():
            skip = F.interpolate(skip, size = x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class UNetResNet50(nn.Module):
    def __init__(self, num_classes = 2):
        super(UNetResNet50, self).__init__()

        self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        # encoder
        self.encoder_layers = [
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool),
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ]

        # decoder
        self.decoder4 = DecoderBlock(2048 + 1024, 512) # Block for layer4 + layer3
        self.decoder3 = DecoderBlock(512 + 512, 256) # Block for layer3 + layer2
        self.decoder2 = DecoderBlock(256 + 256, 128) # Block for layer2 + layer1
        self.decoder1 = DecoderBlock(128 + 64, 64) # Block for layer1 + conv1

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):

        # Save original input size for final upsampling
        original_size = x.shape[2:]  # (H, W)

        # Encoder forward pass
        x0 = self.encoder_layers[0](x)
        x1 = self.encoder_layers[1](x0) 
        x2 = self.encoder_layers[2](x1)
        x3 = self.encoder_layers[3](x2) 
        x4 = self.encoder_layers[4](x3)

        # Decoder forward pass
        x = self.decoder4(x4, x3)
        x = self.decoder3(x, x2)
        x = self.decoder2(x, x1) 
        x = self.decoder1(x, x0)

        # upsample the final output to match the input size
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        x = self.segmentation_head(x) #final seg output
        return x
    
if __name__ == '__main__':
    
    model = UNetResNet50(num_classes=2)
    
    input_tensor = torch.rand(1, 3, 480, 640)  # Batch size 1, 3 channels (RGB), 480x640 image
    output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")  # Should output (1, 2, 480, 640) for binary segmentation