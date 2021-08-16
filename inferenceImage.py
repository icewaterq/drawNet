from train import drawNet
import torch
import cv2
import numpy as np
import os
from os.path import join
from torch.nn import Sequential,Conv2d,BatchNorm2d,ReLU,Module,Linear,BatchNorm1d,CrossEntropyLoss,Dropout,ConvTranspose2d,Sigmoid,MaxPool2d,Softmax2d,LeakyReLU

def loadKernal(path,size=32):
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

class opsDraw(Module):
    def __init__(self,drawSize,stride,pad,kernalPath):
        super(opsDraw, self).__init__()
        self.drawSize = drawSize
        self.deconvB = ConvTranspose2d(8, 1, drawSize, stride=stride, padding=pad, bias=False)
        self.deconvG = ConvTranspose2d(8, 1, drawSize, stride=stride, padding=pad, bias=False)
        self.deconvR = ConvTranspose2d(8, 1, drawSize, stride=stride, padding=pad, bias=False)

        kernel = loadKernal(kernalPath,drawSize)
        self.deconvB.weight.data = kernel
        self.deconvG.weight.data = kernel
        self.deconvR.weight.data = kernel

        for param in self.deconvB.parameters():
            param.requires_grad = False
        for param in self.deconvG.parameters():
            param.requires_grad = False
        for param in self.deconvR.parameters():
            param.requires_grad = False

    def  forward(self,c,d,m_):
        b = self.deconvB(c[:,0:1,...]*d)
        g = self.deconvG(c[:,1:2,...]*d)
        r = self.deconvR(c[:,2:3,...]*d)
        #mask小于阈值则不绘制
        m_[m_<0.1] = 0

        m = self.deconvB(d*m_[:,:1,:,:])/255

        y = torch.cat([b,g,r],1)/255

        return y,m


class Draw:
    def __init__(self,scale):
        self.net = drawNet()
        sd = self.net.state_dict()
        sdpre = torch.load(r'./models/pretrain.pkl')

        for key in sd:
            if key not in sdpre:
                print(key)
                continue
            sd[key] = sdpre[key]
        self.net.load_state_dict(sd)

        self.net.cuda()
        self.net.eval()

        self.scale = scale
        net6 = opsDraw(32*scale,32*scale,0,r'./kernal/050').cuda()
        net8 = opsDraw(24*scale,24*scale,0,r'./kernal/050').cuda()
        net12 = opsDraw(16*scale,16*scale,0,r'./kernal/050').cuda()
        net16 = opsDraw(12*scale,12*scale,0,r'./kernal/032').cuda()
        net24 = opsDraw(8*scale,8*scale,0,r'./kernal/032').cuda()

        # 画笔类型,起始坐标,步长,尺寸,画笔网络
        self.idxLst = {
            '6': ['050', 0, 32, 6, net6],
            '6o': ['050', -16, 32, 7,net6],
            '8': ['050', 0, 24, 8, net8],
            '8o': ['050', -12, 24, 9, net8],
            '12': ['050', 0, 16, 12, net12],
            '12o': ['050', -8, 16, 13, net12],
            '16': ['032', 0, 12, 16, net16],
            '16o': ['032', -6, 12, 17, net16],
            '24': ['032', 0, 8, 24, net24],
            '24o': ['032', -4, 8, 25, net24],
        }


    def draw(self,imgSrc):
        height, width, _ = imgSrc.shape
        #把大图做切片处理
        cropW = (width - 1) // 192 + 1
        cropH = (height - 1) // 192 + 1

        img = np.zeros((192 * cropH, 192 * cropW, 3), dtype=np.uint8)
        img[:height, :width, :] = imgSrc

        img = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        img = img / 255

        img = torch.from_numpy(img).cuda()
        img = img.unsqueeze(0)

        drawLst = {
            '6': [],
            '6o': [],
            '8': [],
            '8o': [],
            '12': [],
            '12o': [],
            '16': [],
            '16o': [],
            '24': [],
            '24o': [],
        }

        with torch.no_grad():
            canvas = torch.zeros((3, (cropH * 192 + 32)*self.scale, (cropW * 192 + 32)*self.scale)).float().cuda()
            for wid in range(cropW):
                for hid in range(cropH):
                    imgCrop = img[:, :, hid * 192:(hid + 1) * 192, wid * 192:(wid + 1) * 192]
                    result = self.net(imgCrop)

                    for idx in self.idxLst:
                        dOps = self.idxLst[idx][-1]
                        drawSize = self.idxLst[idx][2]
                        y, m, d, c, m_, pred  = result[idx]
                        ox = 0
                        if idx.endswith('o'):
                            ox = drawSize//2
                        y,m = dOps(c,d,m_)
                        drawLst[idx].append([(wid * 192+16-ox)*self.scale,((wid + 1) * 192+16+ox)*self.scale,(hid * 192+16-ox)*self.scale,((hid + 1) * 192+16+ox)*self.scale,y[0],m[0]])
            for idx in ['6','6o','8','8o','12','12o','16','16o','24','24o']:
                for x1,x2,y1,y2,y,m in drawLst[idx]:
                    canvas[:,y1:y2,x1:x2] = canvas[:,y1:y2,x1:x2]*(1-m) + y*m

            canvas = canvas.detach().data.cpu().numpy() * 255
            canvas = canvas.swapaxes(0, 1).swapaxes(1, 2)
            canvas = canvas.astype(np.uint8)
        return canvas

if __name__ == '__main__':
    import time
    draw = Draw(2)
    for i in range(10):
        img = cv2.imread(r'./test_images/apple.jpg')
        img = cv2.resize(img, (192, 192))
        start = time.time()
        img = draw.draw(img)
        end = time.time()
        print('time:',end-start)
    cv2.imshow('',img)
    cv2.waitKey()





