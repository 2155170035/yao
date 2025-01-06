import torch
import torch.nn as nn
import torch.nn.functional as F
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()
        self.Softmax = nn.Softmax()


        self.d_conv1 = nn.Conv2d(2,  24, 3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv3 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv4 = nn.Conv2d(24, 1,  3, dilation=1, padding=1)

        self.d_bn1 = nn.BatchNorm2d(24)
        self.d_bn2 = nn.BatchNorm2d(24)
        self.d_bn3 = nn.BatchNorm2d(24)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.d_bn5 = nn.BatchNorm2d(128)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_bn7 = nn.BatchNorm2d(3)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)


    def forward(self, input_images, mini_batch_size=10): # 输入[3B, 2, 128, 128],输出[B, 1, 128, 128]
        net = F.max_pool2d(input_images, kernel_size=[2, 2])  # [3B, 2, 64, 64]
        net = F.max_pool2d(net, kernel_size=[2, 2])  # [3B, 2, 32, 32]
        
        net = self.d_conv1(net)
        net = self.d_bn1(net)
        net = self.leakyrelu1(net)

        net = self.d_conv2(net)
        net = self.d_bn2(net)
        net = self.leakyrelu2(net)

        net = self.d_conv3(net)
        net = self.d_bn3(net)
        net = self.leakyrelu3(net)

        net = self.d_conv4(net)
        net = self.d_bn4(net)
        net1 = self.leakyrelu4(net) # [3B, 1, 32, 32]

        net = net1.view(-1, 1024) # [3B, 1024]
        net = self.fc1(net)      # [3B, 128]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn5(net)
        net = self.Tanh1(net)    # [3B, 128, 1, 1]

        net = net.view(-1, 128) # [3B, 128]
        net = self.fc2(net)      # [3B, 64]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn6(net)
        net = self.Tanh2(net)   # [3B, 64, 1, 1]

        net = net.view(-1, 64) # [3B, 64]
        net = self.fc3(net)      # [3B, 3]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn7(net)
        net = self.Softmax(net) # [3B, 3, 1, 1]
        net = net.squeeze(3).squeeze(2)
        
        realscore0, realscore1, realscore2 = torch.split(net, mini_batch_size, dim=0)
        feat0, feat1, feat2 = torch.split(net1, mini_batch_size, dim=0)
        featDist = torch.mean(torch.pow(feat1 - feat2, 2))

        return realscore0, realscore1, realscore2, featDist
    
class G1_8(nn.Module):
    def __init__(self):
        super(G1_8, self).__init__()
        chn = 64
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        # 空洞卷积计算公式: [x+2p-k-(k-1)*(d-1)]/s + 1,中括号表示向下取整
        self.g1_conv1 = nn.Conv2d(1,     chn,   3, dilation=1, padding=1)
        self.g1_conv2 = nn.Conv2d(chn,   chn,   3, dilation=1, padding=1)
        self.g1_conv3 = nn.Conv2d(chn,   chn*2, 3, dilation=2, padding=2)
        self.g1_conv4 = nn.Conv2d(chn*2, chn*4, 3, dilation=4, padding=4)
        self.g1_conv5 = nn.Conv2d(chn*4, chn*8, 3, dilation=8, padding=8)
        self.g1_conv6 = nn.Conv2d(chn*8, chn*4, 3, dilation=4, padding=4)
        self.g1_conv7 = nn.Conv2d(chn*4, chn*2, 3, dilation=2, padding=2)
        self.g1_conv8 = nn.Conv2d(chn*2, chn,   3, dilation=1, padding=1)
        self.g1_conv9 = nn.Conv2d(chn,   1,     1, dilation=1)

        self.g1_bn1 = nn.BatchNorm2d(chn)
        self.g1_bn2 = nn.BatchNorm2d(chn)
        self.g1_bn3 = nn.BatchNorm2d(chn*2)
        self.g1_bn4 = nn.BatchNorm2d(chn*4)
        self.g1_bn5 = nn.BatchNorm2d(chn*8)
        self.g1_bn6 = nn.BatchNorm2d(chn*4)
        self.g1_bn7 = nn.BatchNorm2d(chn*2)
        self.g1_bn8 = nn.BatchNorm2d(chn)

    def forward(self, input_images): # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        net = self.g1_conv1(input_images)
        net = self.g1_bn1(net)
        net = self.leakyrelu1(net)
        
        net = self.g1_conv2(net)
        net = self.g1_bn2(net)
        net = self.leakyrelu2(net)
        
        net = self.g1_conv3(net)
        net = self.g1_bn3(net)
        net = self.leakyrelu3(net)
        
        net = self.g1_conv4(net)
        net = self.g1_bn4(net)
        net = self.leakyrelu4(net)
        
        net = self.g1_conv5(net)
        net = self.g1_bn5(net)
        net = self.leakyrelu5(net)
        
        net = self.g1_conv6(net)
        net = self.g1_bn6(net)
        net = self.leakyrelu6(net)
        
        net = self.g1_conv7(net)
        net = self.g1_bn7(net)
        net = self.leakyrelu7(net)
        
        net = self.g1_conv8(net)
        net = self.g1_bn8(net)
        net = self.leakyrelu8(net)
        
        output = self.g1_conv9(net)
        

        return output

class G2_64(nn.Module):
    def __init__(self):
        super(G2_64, self).__init__()
        chn = 64
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        self.leakyrelu9 = nn.LeakyReLU(0.2)
        self.leakyrelu10 = nn.LeakyReLU(0.2)
        self.leakyrelu11 = nn.LeakyReLU(0.2)
        self.leakyrelu12 = nn.LeakyReLU(0.2)
        self.leakyrelu13 = nn.LeakyReLU(0.2)

        self.g2_conv1 = nn.Conv2d(1,   chn, 3, dilation=1, padding=1)
        self.g2_conv2 = nn.Conv2d(chn, chn, 3, dilation=2, padding=2)
        self.g2_conv3 = nn.Conv2d(chn, chn, 3, dilation=4, padding=4)
        self.g2_conv4 = nn.Conv2d(chn, chn, 3, dilation=8, padding=8)
        self.g2_conv5 = nn.Conv2d(chn, chn, 3, dilation=16, padding=16)
        self.g2_conv6 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv7 = nn.Conv2d(chn, chn, 3, dilation=64, padding=64)
        self.g2_conv8 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv9 = nn.Conv2d(chn*2, chn, 3, dilation=16, padding=16)
        self.g2_conv10 = nn.Conv2d(chn*2, chn, 3, dilation=8, padding=8)
        self.g2_conv11 = nn.Conv2d(chn*2, chn, 3, dilation=4, padding=4)
        self.g2_conv12 = nn.Conv2d(chn*2, chn, 3, dilation=2, padding=2)
        self.g2_conv13 = nn.Conv2d(chn*2, chn, 3, dilation=1, padding=1)
        self.g2_conv14 = nn.Conv2d(chn, 1,   1, dilation=1)

        self.g2_bn1 = nn.BatchNorm2d(chn)
        self.g2_bn2 = nn.BatchNorm2d(chn)
        self.g2_bn3 = nn.BatchNorm2d(chn)
        self.g2_bn4 = nn.BatchNorm2d(chn)
        self.g2_bn5 = nn.BatchNorm2d(chn)
        self.g2_bn6 = nn.BatchNorm2d(chn)
        self.g2_bn7 = nn.BatchNorm2d(chn)
        self.g2_bn8 = nn.BatchNorm2d(chn)
        self.g2_bn9 = nn.BatchNorm2d(chn)
        self.g2_bn10 = nn.BatchNorm2d(chn)
        self.g2_bn11 = nn.BatchNorm2d(chn)
        self.g2_bn12 = nn.BatchNorm2d(chn)
        self.g2_bn13 = nn.BatchNorm2d(chn)

    def forward(self, input_images): # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        net1 = self.g2_conv1(input_images)
        net1 = self.g2_bn1(net1)
        net1 = self.leakyrelu1(net1)
        
        net2 = self.g2_conv2(net1)
        net2 = self.g2_bn2(net2)
        net2 = self.leakyrelu2(net2)
        
        net3 = self.g2_conv3(net2)
        net3 = self.g2_bn3(net3)
        net3 = self.leakyrelu3(net3)
        
        net4 = self.g2_conv4(net3)
        net4 = self.g2_bn4(net4)
        net4 = self.leakyrelu4(net4)
        
        net5 = self.g2_conv5(net4)
        net5 = self.g2_bn5(net5)
        net5 = self.leakyrelu5(net5)
        
        net6 = self.g2_conv6(net5)
        net6 = self.g2_bn6(net6)
        net6 = self.leakyrelu6(net6)
        
        net7 = self.g2_conv7(net6)
        net7 = self.g2_bn7(net7)
        net7 = self.leakyrelu7(net7)
        
        net8 = self.g2_conv8(net7)
        net8 = self.g2_bn8(net8)
        net8 = self.leakyrelu8(net8)
        
        net9 = torch.cat([net6, net8], dim=1)

        net9 = self.g2_conv9(net9)
        net9 = self.g2_bn9(net9)
        net9 = self.leakyrelu9(net9)
        
        net10 = torch.cat([net5, net9], dim=1)

        net10 = self.g2_conv10(net10)
        net10 = self.g2_bn10(net10)
        net10 = self.leakyrelu10(net10)
        
        net11 = torch.cat([net4, net10], dim=1)

        net11 = self.g2_conv11(net11)
        net11 = self.g2_bn11(net11)
        net11 = self.leakyrelu11(net11)
        
        net12 = torch.cat([net3, net11], dim=1)

        net12 = self.g2_conv12(net12)
        net12 = self.g2_bn12(net12)
        net12 = self.leakyrelu12(net12)
       
        net13 = torch.cat([net2, net12], dim=1)

        net13 = self.g2_conv13(net13)
        net13 = self.g2_bn13(net13)
        net13 = self.leakyrelu13(net13)
        
        net14 = self.g2_conv14(net13)
   
        return net14