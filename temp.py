import torch
import torch.onnx
from train import seqNet
from inferenceImage import opsDraw
import numpy as np
import cv2


# net = seqNet(6)
# x = torch.rand((1,3,192,192))
# canvas = torch.rand((1,3,192,192))
# torch.onnx.export(net,(x,canvas),r'./seqNet.onnx')

d = opsDraw(32 , 32 , 0, r'./kernal/050')

c1 = torch.rand((1,3,5,5))
d1 = torch.zeros((1,8,5,5))
m1 = torch.ones((1,1,5,5))
d1[:,5,:,:] = 1

c2 = torch.rand((1,3,6,6))
d2 = torch.zeros((1,8,6,6))
m2 = torch.ones((1,1,6,6))
d2[:,2,:,:] = 1

y1,m1 = d(c1,d1,m1)
y2,m2 = d(c2,d2,m2)

h1,w1 = y1.size(3), y1.size(2)
h2,w2 = y2.size(3), y2.size(2)
print(y1.size())
print(y2.size())

ox = int((w2-w1)/2)
oy = int((h2-h1)/2)
# y = y1 + y2[:,:,oy:-oy,ox:-ox]
# y = y1 + y2[:,:,oy:-oy,ox:-ox]
y = y1
print(y.size())

y = y*255
print(torch.max(m1),torch.min(m1))

y = y.detach().numpy().reshape(3,h1,w1)
y = y.swapaxes(0,1).swapaxes(1,2)
# y = y.detach().numpy().reshape(h1,w1)
y = y.astype(np.uint8)
cv2.imshow('',y)
cv2.waitKey()
