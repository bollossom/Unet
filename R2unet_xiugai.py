import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
from torch.utils.tensorboard import SummaryWriter
path = os.getcwd()+'\\runs'
writer = SummaryWriter(path+'\\experiment')
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t):
        super(RRCNN_block,self).__init__()
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        # self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1
class R2U_Net(nn.Module):
#img_ch 表示输入图片的channel数 而output_ch表示输出图片的channel数
    def __init__(self, img_ch=1, output_ch=1):
        super(R2U_Net, self).__init__()
        times = 2 # 循环模块的次数

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.channel_1 = 64  ## R2U-net activation maps in first layer
        self.channel_2 = 2 * self.channel_1
        self.channel_3 = 2 * self.channel_2
        self.channel_4 = 2 * self.channel_3
        self.channel_5 = 2 * self.channel_4


        self.channels = [self.channel_1, self.channel_2, self.channel_3, self.channel_4, self.channel_5]

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=self.channel_1, t=times)

        self.RRCNN2 = RRCNN_block(ch_in=self.channel_1, ch_out=self.channel_2, t=times)

        self.RRCNN3 = RRCNN_block(ch_in=self.channel_2, ch_out=self.channel_3, t=times)

        self.RRCNN4 = RRCNN_block(ch_in=self.channel_3, ch_out=self.channel_4, t=times)

        self.RRCNN5 = RRCNN_block(ch_in=self.channel_4, ch_out=self.channel_5, t=times)

        self.Up5 = up_conv(ch_in=self.channel_5, ch_out=self.channel_4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=self.channel_5, ch_out=self.channel_4, t=times)

        self.Up4 = up_conv(ch_in=self.channel_4, ch_out=self.channel_3)
        self.Up_RRCNN4 = RRCNN_block(ch_in=self.channel_4, ch_out=self.channel_3, t=times)

        self.Up3 = up_conv(ch_in=self.channel_3, ch_out=self.channel_2)
        self.Up_RRCNN3 = RRCNN_block(ch_in=self.channel_3, ch_out=self.channel_2, t=times)

        self.Up2 = up_conv(ch_in=self.channel_2, ch_out=self.channel_1)
        self.Up_RRCNN2 = RRCNN_block(ch_in=self.channel_2, ch_out=self.channel_1, t=times)

        self.Conv_1x1 = nn.Conv2d(self.channel_1, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        print('d5',d5.shape)
        print('x4',x4.shape)
        d5 = torch.cat((x4, d5), dim=1)
        print('d51',d5.shape)
        d5 = self.Up_RRCNN5(d5)#channels减半
        print('d52',d5.shape)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)#? 为什么有sigmmoid函数
        return d1
model=R2U_Net()
x=torch.rand(1, 1, 224, 224)
print(model(x).shape)
# ch=1
# h=224
# w=224
# summary(model, input_size=(ch, h, w),device='cpu')
# dummy_input = torch.rand(10, 1, 224, 224)
# with SummaryWriter(comment='Unet') as w:
#     w.add_graph(model, (dummy_input, ))
#tensorboard --logdir=D:\opencv-python\条纹投影\Unet_test\runs\Feb02_10-34-09_LAPTOP-85DL9FARUnet