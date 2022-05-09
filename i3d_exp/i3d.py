import torch
from torch import nn


class Inception3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, eps=0.001, momentum=0.9):
        super(Inception3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            padding_mode='zeros',
            bias=False
        )
        self.batchnorm = nn.BatchNorm3d(
            num_features=out_channels,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True,
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ConcatenateBlocks(nn.Module):
    def __init__(self, blocks):
        super(ConcatenateBlocks, self).__init__()
        self.blocks = nn.ModuleList(modules=blocks)
    
    def forward(self, x):
        outputs = [block(x) for block in self.blocks] # NCDHW
        return torch.cat(outputs, dim=1) # Concatenate on channels


class Inception3D(nn.Module):
    def __init__(self, num_classes=400):
        super(Inception3D, self).__init__()
        self.features_block0 = Inception3DBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
        )
        self.features_block1 = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            ceil_mode=False,
        )
        self.features_block2 = Inception3DBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.features_block3 = Inception3DBlock(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.features_block4 = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            ceil_mode=False,
        )
        self.features_block5 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=192,
                    out_channels=64,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=192,
                        out_channels=96,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=96,
                        out_channels=128,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=192,
                        out_channels=16,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=192,
                        out_channels=32,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block6 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=256,
                        out_channels=128,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=128,
                        out_channels=192,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=256,
                        out_channels=32,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=32,
                        out_channels=96,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=256,
                        out_channels=64,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block7 = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            ceil_mode=False,
        )
        self.features_block8 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=480,
                    out_channels=192,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=480,
                        out_channels=96,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=96,
                        out_channels=208,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=480,
                        out_channels=16,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=16,
                        out_channels=48,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=480,
                        out_channels=64,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block9 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=512,
                    out_channels=160,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=112,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=112,
                        out_channels=224,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=24,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=24,
                        out_channels=64,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=64,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block10 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=512,
                    out_channels=128,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=128,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=24,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=24,
                        out_channels=64,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=64,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block11 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=512,
                    out_channels=112,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=144,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=144,
                        out_channels=288,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=32,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=512,
                        out_channels=64,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block12 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=528,
                    out_channels=256,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=528,
                        out_channels=160,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=160,
                        out_channels=320,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=528,
                        out_channels=32,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=32,
                        out_channels=128,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=528,
                        out_channels=128,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block13 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=(0, 0, 0),
            ceil_mode=False,
        )
        self.features_block14 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=832,
                    out_channels=256,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=832,
                        out_channels=160,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=160,
                        out_channels=320,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=832,
                        out_channels=32,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=32,
                        out_channels=128,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=832,
                        out_channels=128,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block15 = ConcatenateBlocks(
            blocks=[
                # Inner Block 0
                Inception3DBlock(
                    in_channels=832,
                    out_channels=384,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                ),
                # Inner Block 1
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=832,
                        out_channels=192,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=192,
                        out_channels=384,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 2
                nn.Sequential(
                    Inception3DBlock(
                        in_channels=832,
                        out_channels=48,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                    Inception3DBlock(
                        in_channels=48,
                        out_channels=128,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                ),
                # Inner Block 3
                nn.Sequential(
                    nn.MaxPool3d(
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        ceil_mode=False,
                    ),
                    Inception3DBlock(
                        in_channels=832,
                        out_channels=128,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    ),
                ),
            ]
        )
        self.features_block16 = nn.AdaptiveAvgPool3d(
            output_size=(1, 1, 1)
        )
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(in_features=1024, out_features=num_classes)
    
    def forward(self, x):
        x = self.features_block0(x)
        x = self.features_block1(x)
        x = self.features_block2(x)
        x = self.features_block3(x)
        x = self.features_block4(x)
        x = self.features_block5(x)
        x = self.features_block6(x)
        x = self.features_block7(x)
        x = self.features_block8(x)
        x = self.features_block9(x)
        x = self.features_block10(x)
        x = self.features_block11(x)
        x = self.features_block12(x)
        x = self.features_block13(x)
        x = self.features_block14(x)
        x = self.features_block15(x)
        x = self.features_block16(x)
        
        # x = torch.squeeze(x, axis=(2, 3, 4))
        x = torch.squeeze(x, axis=4)
        x = torch.squeeze(x, axis=3)
        x = torch.squeeze(x, axis=2)
        
        x = torch.reshape(x, shape=(-1, 1, 1024))
        x = torch.mean(x, dim=1)
        
        x = self.dropout(x)
        x = self.output(x)
        return x       

