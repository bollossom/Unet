import torch
import torch.nn as nn
import torch.nn.functional as F
class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=strides,padding=padding)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=strides,padding=padding)
        self.bn2=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.bn1(x1)
        x3=F.relu(x2)
        x4=self.conv2(x3)
        x5=self.bn2(x4)
        out=F.relu(x5)
        return  out
class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn, self).__init__()
        #out=(in-1)*stride-2*padding+kernel_size=2*in
        self.conv1=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,
                                      stride=strides,padding=0,bias=True)
        self.bn1=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x1=self.conv1(x)
        x2=self.bn1(x1)
        out=F.relu(x2)
        return out
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv=double_conv2d_bn(1,8)
        self.layer2_conv=double_conv2d_bn(8,16)
        self.layer3_conv=double_conv2d_bn(16,32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv=nn.Conv2d(out_channels=8,in_channels=1,kernel_size=3,padding=1,stride=1,bias=True)

        self.deconv1=deconv2d_bn(128,64)
        self.deconv2=deconv2d_bn(64,32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)
    def forward(self, x):
        #1*224*224
        conv1=self.layer1_conv(x)#输出为8*224*224
        pool1=F.max_pool2d(conv1,2)#输出为8*112*112

        conv2=self.layer2_conv(pool1)#输出为16*112*112
        pool2=F.max_pool2d(conv2,2)#输出为16*56*56

        conv3=self.layer3_conv(pool2)#32*56*56
        pool3=F.max_pool2d(conv2,2)#32*28*28

        conv4=self.layer4_conv(pool3)#64*28*28
        pool4=F.max_pool2d(conv4,2)#64*14*14

        conv5=self.layer5_conv(pool4)#128*14*14

        convt1=self.deconv1(conv5)#64*28*28
        concat1=torch.cat([convt1,conv4],dim=1)#128*56*56
        conv6=self.layer6_conv(concat1)#64*56*56

        convt2=self.deconv2(conv6)#32*112*112
        concat2=torch.cat([conv3,convt2],dim=1)#64*112*112
        conv7=self.layer7_conv(concat2)#32*112*112

        convt3=self.deconv3(conv7)#16*224*224
        concat3=torch.cat([convt3,conv2],dim=1)#
