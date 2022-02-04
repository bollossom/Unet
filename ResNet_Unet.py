#无maxpool使用stride=2进行减半
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import os
# path = os.getcwd()+'\\runs'
# writer = SummaryWriter(path+'\\experiment')
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)
class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[16, 32, 64,128]):#修改[64,128,256,512]
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        # self.output_layer = nn.Sequential(
        #     nn.Conv2d(filters[0], 1, 1, 1),
        #     nn.Sigmoid(),
        # )
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        print('x',x.shape)
        x1 = self.input_layer(x) + self.input_skip(x)
        print('x1',x1.shape)
        x2 = self.residual_conv_1(x1)
        print('x2',x2.shape)
        x3 = self.residual_conv_2(x2)
        print('x3',x3.shape)
        # Bridge
        x4 = self.bridge(x3)
        print('x4', x4.shape)
        # Decode
        x4 = self.upsample_1(x4)
        print('x4_T', x4.shape)
        x5 = torch.cat([x4, x3], dim=1)
        print('x5_channel', x5.shape)

        x6 = self.up_residual_conv1(x5)
        print('x6', x6.shape)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)
        print('x8',x8.shape)

        x8 = self.upsample_3(x8)
        print('x8_T', x8.shape)
        x9 = torch.cat([x8, x1], dim=1)
        print('x9', x9.shape)

        x10 = self.up_residual_conv3(x9)
        print('x10', x10.shape)

        output = self.output_layer(x10)

        return output
model=ResUnet(1)
x=torch.rand(3, 1, 224, 224)
print(model(x).shape)
# dummy_input = torch.rand(10, 1, 224, 224)
# with SummaryWriter(comment='Unet') as w:
#     w.add_graph(model, (dummy_input, ))
# batch=3
# ch = 1
# h = 224
# w = 224
# summary(model, input_size=(ch, h, w), device='cpu')
#tensorboard --logdir=D:\opencv-python\条纹投影\Unet_test\runs\Jan21_10-36-03_LAPTOP-85DL9FARUnet