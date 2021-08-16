import torch
import torch.utils.data
from torch.nn import Sequential,Conv2d,BatchNorm2d,ReLU,Module,Linear,BatchNorm1d,CrossEntropyLoss,Dropout,ConvTranspose2d,Sigmoid,MaxPool2d,Softmax2d,LeakyReLU
from torch.utils.data import Dataset,DataLoader
import cv2
import os
from os.path import join
import random
import numpy as np
import torch.nn.functional as F
from mobilenet import mobilenet_v2
from resnet import resnet50

#图像的像素误差loss函数
class PixelLoss(Module):
    def __init__(self):
        super(PixelLoss, self).__init__()

    def forward(self,y,gt):
        loss = torch.mean((y.contiguous() - gt.contiguous()) ** 2)
        return loss

#contentLoss，图像风格化中使用，利用与训练网络的浅层特征。
class contentLoss(Module):
    def __init__(self):
        super(contentLoss, self).__init__()

        self.mbnet = mobilenet_v2()
        sdpre = torch.load(r'./models/mobilenet_v2-b0353104.pth')
        sd = self.mbnet.state_dict()
        for key in sd:
            if key not in sdpre:
                continue
            sd[key] = sdpre[key]
        self.mbnet.load_state_dict(sd)

        self.resnet = resnet50()
        sdpre = torch.load(r'./models/resnet50-19c8e357.pth')
        sd = self.resnet.state_dict()
        for key in sd:
            if key not in sdpre:
                continue
            sd[key] = sdpre[key]
        self.resnet.load_state_dict(sd)


        for param in self.mbnet.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self.mean = torch.from_numpy(mean).cuda()
        self.std = torch.from_numpy(std).cuda()

    def forward(self,pred,gt):
        pred = pred.contiguous()
        gt = gt.contiguous()
        loss1 = torch.mean((self.mbnet.features((pred - self.mean) / self.std) - self.mbnet.features((gt - self.mean) / self.std)) ** 2)
        loss2 = torch.mean((self.resnet((pred - self.mean) / self.std) - self.resnet((gt - self.mean) / self.std)) ** 2)
        loss = loss1+loss2
        return loss

#加载反卷积核
def loadKernel(path,size=32):
    flist = os.listdir(path)
    kernel = []
    for fname in flist:
        fpath = join(path,fname)
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # _,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = cv2.resize(img,(size,size),cv2.INTER_AREA)
        for i in range(1):
            kernel.append(img)
    kernel = np.array(kernel)
    kernel = kernel.astype(np.float32).reshape(8,1,size,size)
    kernel = torch.from_numpy(kernel)
    return kernel

#绘画操作符
class opsDraw(Module):
    def __init__(self,inNum,inSize,outSize,drawSize,stride,pad,kernalPath):
        super(opsDraw, self).__init__()
        #输出为N,3,H,W,代表每个位置的BGR的数值，范围为0-1
        self.color = Sequential(
            Conv2d(inNum, 128, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2d(128, 32, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(32),
            LeakyReLU(),
            Conv2d(32, 3, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(3),
            Sigmoid()
        )

        #输出为N,8,H,W，8代表着每种画笔的八个方向
        self.draw = Sequential(
            Conv2d(inNum, 128, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2d(128, 32, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(32),
            LeakyReLU(),
            Conv2d(32, 8, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(8),
            # Softmax2d(),
        )

        #输出为N,2,H,W,表示每一个位置的权重，0代表该位置不绘制，取值范围0-1
        self.mask = Sequential(
            Conv2d(inNum, 128, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2d(128, 32, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(32),
            LeakyReLU(),
            Conv2d(32, 2, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(2),
            Softmax2d(),
        )

        #三个反卷积，对应BGR三个通道
        self.deconvB = ConvTranspose2d(8, 1, drawSize, stride=stride, padding=pad, bias=False)
        self.deconvG = ConvTranspose2d(8, 1, drawSize, stride=stride, padding=pad, bias=False)
        self.deconvR = ConvTranspose2d(8, 1, drawSize, stride=stride, padding=pad, bias=False)

        #加载反卷积核，参数的本质就是画笔
        kernel = loadKernel(kernalPath,drawSize)
        self.deconvB.weight.data = kernel
        self.deconvG.weight.data = kernel
        self.deconvR.weight.data = kernel

        #该卷积核固定，不做训练。
        for param in self.deconvB.parameters():
            param.requires_grad = False
        for param in self.deconvG.parameters():
            param.requires_grad = False
        for param in self.deconvR.parameters():
            param.requires_grad = False

    def  forward(self,x):

        c = self.color(x)
        d = self.draw(x)
        #因为其中一个loss的缘故，此处做了数值阶段，否则会因为数值过大经过exp出现NaN
        d = torch.clamp(d,-10,10)

        #经过softmax，代表使用哪一种方向的画笔
        d = torch.softmax(d,1)

        b = self.deconvB(c[:,0:1,...]*d)
        g = self.deconvG(c[:,1:2,...]*d)
        r = self.deconvR(c[:,2:3,...]*d)

        m_ = self.mask(x)

        m = self.deconvB(d*m_[:,:1,:,:])/255

        y = torch.cat([b,g,r],1)/255

        return y,m,d,m_,c

class conv_3x3(Module):
    def __init__(self,inNum,outNum,kernalSize,stride,pad):
        super(conv_3x3, self).__init__()
        self.conv = Conv2d(inNum,outNum,kernalSize,stride,pad,bias=False)
        self.bn = BatchNorm2d(outNum)
        self.relu = LeakyReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class seqNet(Module):
    #某一个尺度的绘图网络，outSize为输出特征图的尺寸，isOffset代表画笔是否偏移。
    def __init__(self,outSize,isOffset = False):
        super(seqNet,self).__init__()

        self.conv1 = conv_3x3(6,32,3,1,1)
        self.conv2 = conv_3x3(32,32,3,1,1)
        self.conv3 = conv_3x3(32,64,3,1,1)
        self.conv4 = conv_3x3(64,64,3,1,1)
        self.conv5 = conv_3x3(64,128,3,1,1)
        self.conv6 = conv_3x3(128,128,3,1,1)

        self.outSize = outSize
        self.isOffset = isOffset

        if outSize <=12:
            self.d = opsDraw(128, outSize, 192, 192//outSize, 192//outSize, 0, r'./kernal/050')
        else:
            self.d = opsDraw(128, outSize, 192, 192 // outSize, 192 // outSize, 0, r'./kernal/032')

    #输入img为要绘制的图像，可以认为是ground truth，canvas代表着画布
    def forward(self,img,canvas):
        #这里的网络是特意设计的，让大画笔和小画笔的卷积和下采样尽可能均衡
        x = torch.cat([img,canvas],1)
        x = self.conv1(x)
        x = torch.max_pool2d(x,2,2)
        x = self.conv2(x)
        if self.outSize == 6:
            x = torch.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        if self.outSize == 16 or self.outSize == 8:
            x = torch.max_pool2d(x, 3, 3)
        else:
            x = torch.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        if self.outSize == 6 or self.outSize == 8 or self.outSize == 12:
            x = torch.max_pool2d(x, 2, 2)
        x = self.conv5(x)
        x = torch.max_pool2d(x, 2, 2)
        x = self.conv6(x)

        #如果画笔需要偏移，则将卷积层输出用插值来使尺寸+1
        if self.isOffset:
            x = F.interpolate(x , (self.outSize+1,self.outSize+1))

        y,m,d,m_,c = self.d(x)

        #对输出做crop，使输出192x192
        if self.isOffset:
            ox = (192 // self.outSize)//2
            y = y[:,:,ox:-ox,ox:-ox]
            m = m[:,:,ox:-ox,ox:-ox]
        return y,m,d,m_,c

closs = contentLoss().cuda()

class drawNet(Module):
    def __init__(self):
        super(drawNet,self).__init__()
        #整个绘制网络包含了十个网络，对应了五个尺度，每个尺度有两种偏移相互交错，使其覆盖整个区域。
        self.net6 = seqNet(6)
        self.net6o = seqNet(6, True)

        self.net8 = seqNet(8)
        self.net8o = seqNet(8, True)

        self.net12 = seqNet(12)
        self.net12o = seqNet(12, True)

        self.net16 = seqNet(16)
        self.net16o = seqNet(16, True)

        self.net24 = seqNet(24)
        self.net24o = seqNet(24, True)

        self.pixelloss = PixelLoss()


    def forward(self,img):

        canvas = img.new_zeros(img.size()).detach()
        y6,m6,d6,m_6,c6 = self.net6(img,canvas)
        pred6 = canvas * (1 - m6) + y6 * m6

        canvas = pred6*1
        y6o,m6o,d6o,m_6o,c6o = self.net6o(img,canvas)
        pred6o = canvas * (1 - m6o) + y6o * m6o

        canvas = pred6o*1
        y8,m8,d8,m_8,c8 = self.net8(img,canvas)
        pred8 = canvas * (1 - m8) + y8 * m8

        canvas = pred8*1
        y8o,m8o,d8o,m_8o,c8o = self.net8o(img,canvas)
        pred8o = canvas * (1 - m8o) + y8o * m8o

        canvas = pred8o*1
        y12,m12,d12,m_12,c12 = self.net12(img,canvas)
        pred12 = canvas * (1 - m12) + y12 * m12

        canvas = pred12*1
        y12o,m12o,d12o,m_12o,c12o = self.net12o(img,canvas)
        pred12o = canvas * (1 - m12o) + y12o * m12o

        canvas = pred12o*1
        y16,m16,d16,m_16,c16 = self.net16(img,canvas)
        pred16 = canvas * (1 - m16) + y16 * m16

        canvas = pred16*1
        y16o, m16o, d16o,m_16o,c16o = self.net16o(img, canvas)
        pred16o = canvas * (1 - m16o) + y16o * m16o

        canvas = pred16o*1
        y24, m24, d24,m_24,c24 = self.net24(img, canvas)
        pred24 = canvas * (1 - m24) + y24 * m24

        canvas = pred24*1
        y24o, m24o, d24o,m_24o,c24o = self.net24o(img, canvas)
        pred24o = canvas * (1 - m24o) + y24o * m24o

        #loss1用于监督每一层的输出
        loss1 = self.pixelloss(img,pred6)\
               +self.pixelloss(img,pred6o)\
               +self.pixelloss(img,pred8)\
               +self.pixelloss(img,pred8o)\
               +self.pixelloss(img,pred12)\
               +self.pixelloss(img,pred12o)\
               +self.pixelloss(img,pred16)\
               +self.pixelloss(img,pred16o)\
               +self.pixelloss(img,pred24)\
               +self.pixelloss(img,pred24o)

        #loss7也用于监督使每一种尺度的输出和目标图像相似，但contentLoss经过了预训练卷积层，有助于训练，使其的边缘等特征更相似
        loss7 = closs(pred24o,img)

        #loss2 让画笔和图像尽可能相似，但是越小的画笔最后画，所以权重也最大。
        loss2 = self.pixelloss(img*m6, y6*m6)*0.2 \
                + self.pixelloss(img*m6o, y6o*m6o)*0.2 \
                + self.pixelloss(img*m8, y8*m8)*0.4 \
                + self.pixelloss(img*m8o, y8o*m8o)*0.4\
                + self.pixelloss(img*m12, y12*m12)*0.6 \
                + self.pixelloss(img*m12o, y12o*m12o)*0.6\
                + self.pixelloss(img*m16, y16*m16)*0.8 \
                + self.pixelloss(img*m16o, y16o*m16o)*0.8 \
                + self.pixelloss(img*m24, y24*m24) \
                + self.pixelloss(img*m24o, y24o*m24o)

        #让mask尽可能地小，这样就可以让不必要的画笔不显示。
        loss3 = torch.mean(m_6[:, :1, :, :]) \
                + torch.mean(m_6o[:, :1, :, :]) \
                + torch.mean(m_8[:, :1, :, :]) \
                + torch.mean(m_8o[:, :1, :, :]) \
                + torch.mean(m_12[:, :1, :, :]) \
                + torch.mean(m_12o[:, :1, :, :])\
                + torch.mean(m_16[:, :1, :, :]) \
                + torch.mean(m_16o[:, :1, :, :]) \
                + torch.mean(m_24[:, :1, :, :]) \
                + torch.mean(m_24o[:, :1, :, :])

        #让每种画笔的softmax概率尽可能的远离0.5，所以loss4是越大越好
        loss4 = torch.mean(torch.abs(d6-0.5))\
                +torch.mean(torch.abs(d6o-0.5))\
                +torch.mean(torch.abs(d8-0.5))\
                +torch.mean(torch.abs(d8o-0.5))\
                +torch.mean(torch.abs(d12-0.5))\
                +torch.mean(torch.abs(d12o-0.5))\
                +torch.mean(torch.abs(d16-0.5))\
                +torch.mean(torch.abs(d16o-0.5))\
                +torch.mean(torch.abs(d24-0.5))\
                +torch.mean(torch.abs(d24o-0.5))\

        #让每种画笔mask的softmax概率尽可能的远离0.5，所以loss5是越大越好
        loss5 = torch.mean(torch.abs(m_6-0.5))\
                +torch.mean(torch.abs(m_6o-0.5))\
                +torch.mean(torch.abs(m_8-0.5))\
                +torch.mean(torch.abs(m_8o-0.5))\
                +torch.mean(torch.abs(m_12-0.5))\
                +torch.mean(torch.abs(m_12o-0.5))\
                +torch.mean(torch.abs(m_16-0.5))\
                +torch.mean(torch.abs(m_16o-0.5))\
                +torch.mean(torch.abs(m_24-0.5))\
                +torch.mean(torch.abs(m_24o-0.5))\
        #在单张画的不同位置和不同画的相同位置尽可能让画笔的方向不一样，使方差最大化，loss6也是越大越好
        loss6 = torch.mean(torch.std(d6 * m_6[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d6o * m_6o[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d8 * m_8[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d8o * m_8o[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d12 * m_12[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d12o * m_12o[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d16 * m_16[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d16o * m_16o[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d24 * m_24[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d24o * m_24o[:, :1, :, :], [2, 3])) \
                + torch.mean(torch.std(d6 * m_6[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d6o * m_6o[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d8 * m_8[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d8o * m_8o[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d12 * m_12[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d12o * m_12o[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d16 * m_16[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d16o * m_16o[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d24 * m_24[:, :1, :, :], 0)) \
                + torch.mean(torch.std(d24o * m_24o[:, :1, :, :], 0))


        result = {
            '6':[y6,m6,d6,c6,m_6,pred6],
            '6o':[y6o,m6o,d6o,c6o,m_6o,pred6o],
            '8':[y8,m8,d8,c8,m_8,pred8],
            '8o':[y8o,m8o,d8o,c8o,m_8o,pred8o],
            '12':[y12,m12,d12,c12,m_12,pred12],
            '12o':[y12o,m12o,d12o,c12o,m_12o,pred12o],
            '16':[y16,m16,d16,c16,m_16,pred16],
            '16o':[y16o,m16o,d16o,c16o,m_16o,pred16o],
            '24':[y24,m24,d24,c24,m_24,pred24],
            '24o':[y24o,m24o,d24o,c24o,m_24o,pred24o],
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'loss4': loss4,
            'loss5': loss5,
            'loss6': loss6,
            'loss7': loss7,
        }

        return result

class lineDataSet(Dataset):
    def __init__(self,path):
        flist = os.listdir(path)
        self.paths=[]
        for fname in flist:
            self.paths.append(join(path,fname))
        self.lastImage = None


    def __getitem__(self, index):
        imgSrc = cv2.imread(self.paths[index])
        h,w,c = imgSrc.shape
        #在图片中随机裁剪
        sizeLst=[192,256,384,512]
        size = random.choice(sizeLst)
        if min(w,h)<513:
            imgSrc = cv2.resize(imgSrc,(513,513))
            w,h=513,513
        for i in range(5):
            x = random.randint(0,w-size)
            y = random.randint(0,h-size)
            img = imgSrc[y:y+size,x:x+size,:]
            img = cv2.resize(img,(192,192))
            img = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
            std = np.mean(np.std(img.reshape(3,-1),1))
            img = img/255
            #尽量选择方差大的图片训练。
            if std>25:
                self.lastImage = img
                return {'img':img}
        # print('std too small.')

        return {'img': self.lastImage}

    def __len__(self):
        return len(self.paths)//32*32


def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

if __name__=='__main__':
    torch.backends.cudnn.enabled = False
    trainset = lineDataSet(r'E:\public_dataset\detect\COCO\test2017')

    dataloader_train = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

    net = drawNet()

    # sd = net.state_dict()
    # sdpre = torch.load(r'./models/pretrain.pkl')
    #
    # for key in sd :
    #     if key not in sdpre :
    #         print(key)
    #         continue
    #     sd[key] = sdpre[key]
    # net.load_state_dict(sd)

    net.cuda()
    net.train()

    optimizer_G = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(101):
        if epoch == 10:
            adjust_learning_rate(optimizer_G)
        if epoch == 20:
            adjust_learning_rate(optimizer_G)
        if epoch == 40:
            adjust_learning_rate(optimizer_G)

        for i, sample in enumerate(dataloader_train):
            img = sample['img'].cuda()

            optimizer_G.zero_grad()
            result  = net(img)

            loss1 = result['loss1']
            loss2 = result['loss2']
            loss3 = result['loss3']
            loss4 = result['loss4']
            loss5 = result['loss5']
            loss6 = result['loss6']
            loss7 = result['loss7']

            pred = result['24o'][-1]
            m24 = result['24o'][1]
            m12 = result['12'][1]
            m8 = result['8'][1]
            m8o = result['8o'][1]

            w=1
            loss = loss1*5+loss2*5+loss3*0.1+w/(loss4+1)+5/(loss5+1)+w/(loss6+1)+loss7
            loss.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(epoch, loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item(),loss7.item())
            if i % 10 == 0:
                pred = pred[0].detach().data.cpu().numpy() * 255
                pred = pred.swapaxes(0, 1).swapaxes(1, 2)
                pred = pred.astype(np.uint8)

                m24 = m24[0].detach().data.cpu().numpy() * 255
                m24 = m24.swapaxes(0, 1).swapaxes(1, 2)
                m24 = m24.astype(np.uint8)

                m12 = m12[0].detach().data.cpu().numpy() * 255
                m12 = m12.swapaxes(0, 1).swapaxes(1, 2)
                m12 = m12.astype(np.uint8)

                m8 = m8[0].detach().data.cpu().numpy() * 255
                m8 = m8.swapaxes(0, 1).swapaxes(1, 2)
                m8 = m8.astype(np.uint8)

                m8o = m8o[0].detach().data.cpu().numpy() * 255
                m8o = m8o.swapaxes(0, 1).swapaxes(1, 2)
                m8o = m8o.astype(np.uint8)

                img = img[0].detach().data.cpu().numpy() * 255
                img = img.swapaxes(0, 1).swapaxes(1, 2)
                img = img.astype(np.uint8)

                imgShow = np.zeros((192, 192 * 5, 3), dtype=np.uint8)
                imgShow[:, 192*0 : 192*1 , :] = m8
                imgShow[:, 192*1 : 192*2 , :] = m8o
                imgShow[:, 192*2 : 192*3 , :] = m24
                imgShow[:, 192*3 : 192*4 , :] = pred
                imgShow[:, 192*4 : 192*5 , :] = img

                cv2.imshow('gt', imgShow)
                cv2.waitKey(1)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), r'./models/drawNet_epoch{}.pkl'.format(epoch))


