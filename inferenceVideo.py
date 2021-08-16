from train import drawNet
import torch
import cv2
import numpy as np
import os
from os.path import join
import pyflann
import time
from tqdm import tqdm

def loadKernal(path):
    flist = os.listdir(path)
    kernel = []
    for fname in flist:
        fpath = join(path,fname)
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel.append(img)
    return kernel


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
        self.kernal032 = loadKernal(r'./kernal/032')
        self.kernal050 = loadKernal(r'./kernal/050')

        self.scale = scale

        # 画笔类型,起始坐标,步长,尺寸
        self.idxLst = {
            '6': ['050', 0, 32, 6],
            '6o': ['050', -16, 32, 7],
            '8': ['050', 0, 24, 8],
            '8o': ['050', -12, 24, 9],
            '12': ['050', 0, 16, 12],
            '12o': ['050', -8, 16, 13],
            '16': ['032', 0, 12, 16],
            '16o': ['032', -6, 12, 17],
            '24': ['032', 0, 8, 24],
            '24o': ['032', -4, 8, 25],
        }

    def draw(self,imgSrc,isSave=False):

        height, width, _ = imgSrc.shape
        cropW = (width - 1) // 192 + 1
        cropH = (height - 1) // 192 + 1

        img = np.zeros((192 * cropH, 192 * cropW, 3), dtype=np.uint8)
        img[:height, :width, :] = imgSrc

        img = img.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        img = img / 255

        img = torch.from_numpy(img).cuda()
        img = img.unsqueeze(0)

        drawLst = {
            32: [],
            24: [],
            16: [],
            12: [],
            8: [],
        }
        sTime = time.time()
        for wid in range(cropW):
            for hid in range(cropH):
                imgCrop = img[:, :, hid * 192:(hid + 1) * 192, wid * 192:(wid + 1) * 192]
                with torch.no_grad():
                    result = self.net(imgCrop)
                canvas = torch.zeros((3, 192 + 32, 192 + 32)).float().cuda()
                gt =  torch.zeros((3, 192 + 32, 192 + 32)).float().cuda()
                gt[:,16:-16,16:-16] = imgCrop*1
                lastDiff = gt*1
                for idx in self.idxLst:
                    y, m, d, c, m_, pred = result[idx]
                    drawType, start, stride, size = self.idxLst[idx]
                    canvas[:,16:-16,16:-16]=pred
                    curDiff = torch.abs(gt-canvas)
                    diff = lastDiff - curDiff
                    lastDiff = curDiff*1
                    for i in range(size):
                        for j in range(size):
                            mask = m_[0, 0, j, i].item()
                            drawIdx = torch.argmax(d[0, :, j, i]).item()
                            b = int(c[0, 0, j, i].item() * 255)
                            g = int(c[0, 1, j, i].item() * 255)
                            r = int(c[0, 2, j, i].item() * 255)
                            x1 = start + stride * i + 16
                            y1 = start + stride * j + 16
                            x2 = x1 + stride
                            y2 = y1 + stride
                            err = torch.sum(diff[:,y1:y2,x1:x2]).item()
                            if mask > 0.1:
                                drawLst[stride].append([drawIdx, mask, b, g, r, x1+ wid*192, y1+ hid*192, x2+ wid*192, y2+ hid*192,err])

        eTime = time.time()
        print('inference time:',eTime-sTime)

        sTime = time.time()
        fid = 0
        canvas = np.zeros(((cropH*192+32)*self.scale,(cropW*192+32)*self.scale,3),dtype=np.float32)
        sizeLst = [32,24,16,12,8]
        for size in sizeLst:
            drawLst[size].sort(key=lambda x: x[-1], reverse=True)
            count = len(drawLst[size])
            flagLst=[0 for i in range(count)]
            xyLst = [[drawLst[size][i][5]+drawLst[size][i][7],drawLst[size][i][6]+drawLst[size][i][8]] for i in range(count)]
            xyLst = np.array(xyLst)
            pyflann.set_distance_type('euclidean')
            flann = pyflann.FLANN()
            params = flann.build_index(xyLst, algorithm='kdtree', trees=4)
            neighbours, distances = flann.nn_index(xyLst, 10, checks=params['checks'])

            sortLst = []
            for i in range(count):
                if flagLst[i] == 1:
                    continue
                sortLst.append(i)
                flagLst[i] = 1
                j = i
                while True:
                    maxIdx = None
                    maxErr = -100000
                    for nearIdx in neighbours[j][1:9]:
                        if flagLst[nearIdx] == 1:
                            continue
                        if drawLst[size][nearIdx][-1]>maxErr:
                            maxIdx = nearIdx
                            maxErr = drawLst[size][nearIdx][-1]
                    if maxIdx is None:
                        break
                    j = maxIdx
                    flagLst[j] = 1
                    sortLst.append(j)

            for idx in sortLst:
                drawIdx, mask, b, g, r, x1, y1, x2, y2, err = drawLst[size][idx]
                if size in [32, 24, 16]:
                    kernal = self.kernal050
                else:
                    kernal = self.kernal032
                draw = kernal[drawIdx]
                draw = cv2.resize(draw, (size*self.scale, size*self.scale))
                draw = draw.astype(np.float32) / 255
                mask = draw * mask

                x1*=self.scale
                y1*=self.scale
                x2*=self.scale
                y2*=self.scale

                cx = (x1+x2)/2
                cy = (y1+y2)/2

                canvas[y1:y2, x1:x2, 0] = canvas[y1:y2, x1:x2, 0] * (1 - mask) + mask * draw * b
                canvas[y1:y2, x1:x2, 1] = canvas[y1:y2, x1:x2, 1] * (1 - mask) + mask * draw * g
                canvas[y1:y2, x1:x2, 2] = canvas[y1:y2, x1:x2, 2] * (1 - mask) + mask * draw * r

                imgShow = (canvas).astype(np.uint8)
                if isSave:
                    cv2.imwrite(r'./output/{:0>7d}.jpg'.format(fid), imgShow)
                cv2.imshow('', imgShow)
                cv2.waitKey(25)
                fid += 1
        eTime = time.time()
        print('calculate path time:',eTime - sTime)
        if isSave:
            print('save video.')
            fps = 60
            size = (imgShow.shape[1], imgShow.shape[0])
            videoWrite = cv2.VideoWriter(r'./output/result.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
            flist = os.listdir(r'./output')

            fid = 0
            for fname in tqdm(flist):
                if not fname.endswith('.jpg'):
                    continue
                img = cv2.imread(join(r'./output', fname))
                img = cv2.resize(img, size)
                videoWrite.write(img)
                fid += 1
            videoWrite.release()

        return imgShow


if __name__ == '__main__':
    draw = Draw(2)

    for i in range(1):
        img = cv2.imread(r'./test_images/apple.jpg')
        img = cv2.resize(img, (192, 192))
        img = draw.draw(img,isSave=True)
    cv2.imshow('',img)
    cv2.waitKey()



