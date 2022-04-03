import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from data_process_140 import get_train_batch_combine
import torch.nn.functional as F
import numpy as np
# import os
# path = os.getcwd()+'\\runs'
# writer = SummaryWriter(path+'\\experiment')
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
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
            nn.LeakyReLU(),
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
        self.output_layer=nn.Conv2d(filters[0], 8, kernel_size=3, padding=1,stride=1)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        # output = self.output_layer(x10)
        output=self.output_layer(x10)

        return output
class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        # self.layer10_conv = nn.Conv2d(8, 1, kernel_size=3,
        #                               stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp=conv9
        # outp = self.layer10_conv(conv9)
        # outp = self.sigmoid(outp)
        return outp
class Unet_end(nn.Module):
    def __init__(self):
        super(Unet_end, self).__init__()
        self.layer1=ResUnet(1)
        self.layer2=Unet()
        self.layer3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,padding=1, stride=1)
        self.layer4=nn.Conv2d(in_channels=16,out_channels=2,kernel_size=1,stride=1)
    def forward(self,x1,x2):
        x1_during=self.layer1(x1)#原始图像
        x2_during=self.layer2(x2)#背景强度
        x3=torch.cat([x1_during,x2_during],dim=1)
        x4=self.layer3(x3)
        out=self.layer4(x4)
        return out

# import torch
# import torch.nn.functional as F
# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Unet_end().to(DEVICE)
# x1, x2, labels = get_train_batch_combine(1, 1)
# x1, x2, labels = x1.to(DEVICE), x2.to(DEVICE), labels.to(DEVICE)
# print(x1.shape,x2.shape,labels.shape)
# end=model(x1,x2)
# log_pre = nn.LogSoftmax(dim=1)(end)
# print('1',labels)
# print('2',log_pre)
# loss = F.kl_div(log_pre, labels, reduction='batchmean')
# print(loss)



    # def initialize(self):
    #     a = np.sqrt(3 / self.layer1)
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_uniform_(m.weight.data)
# model = Unet_end()
# x1=torch.randn(1,1,224,224)
# x2=torch.randn(1,1,224,224)
# end=model(x1,x2)
# print(end.shape)
# ch = 1
# h = 224
# w = 224
# summary(model, input_size=(ch, h, w), device='cpu')
# model = Unet()
# dummy_input1 = torch.rand(1, 1, 224, 224)
# dummy_input2 = torch.rand(1, 1, 224, 224)
# with SummaryWriter(comment='Unet') as w:
#     w.add_graph(model, (dummy_input1, ),(dummy_input2, ))
#tensorboard --logdir=D:\opencv-python\条纹投影\背景强度的训练\runs\Dec26_19-15-53_LAPTOP-85DL9FARUnet