import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate
from torchsummary import summary
from data_process import get_train_batch
from torch.utils.tensorboard import SummaryWriter
# path = os.getcwd()+'\\runs'
# writer = SummaryWriter(path+'\\experiment')
T = 784

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.T = T
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.LIFNode1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.LIFNode2= neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.LIFNode1(out)
        out = self.bn2(self.conv2(out))
        out = self.LIFNode2(out)
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.LIFNode1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.LIFNode1(out)
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
        self.layer10_conv = nn.Conv2d(8, 1, kernel_size=3,
                                      stride=1, padding=1, bias=True)

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
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return outp
model = Unet()
b=torch.randn(1,1,224,224)
out1=model(b)
print(out1.shape)
# ch = 1
# h = 224
# w = 224
# summary(model, input_size=(ch, h, w), device='cpu')
# model = Unet()
# train_epochs =100  #训练轮数
# batch_size =3  #批大小
# total_batch = int(2880/batch_size)  #每一轮批次数
# for ep in range(0, train_epochs):
#     for i in range(total_batch):
#         batch_x, batch_label = get_train_batch(i, batch_size)  # 读取批次数据
#         a=model(batch_x)
#         print(a.shape,a)

# print(out-out1)
# dummy_input = torch.rand(10, 1, 224, 224)
# with SummaryWriter(comment='Unet') as w:
#     w.add_graph(model, (dummy_input, ))
#tensorboard --logdir=D:\opencv-python\条纹投影\背景强度的训练\runs\Dec26_19-15-53_LAPTOP-85DL9FARUnet
