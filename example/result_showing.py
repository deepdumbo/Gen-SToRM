#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:45:14 2020
This code is used to read the generator param and the latent variables to
produce the results.
@author: zouqing
"""

import numpy as np
from numpy import save
import os
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


dtype = torch.float

# results save folder
if not os.path.isdir('slices4_result'):
    os.mkdir('slices4_result')
if not os.path.isdir('slices4_result/images'):
    os.mkdir('slices4_result/images')
    
#%% The generator
class generator(nn.Module):
    # initializers
    def __init__(self,siz_latent=2,d=24, out_channel=2):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(siz_latent, 100, 1, 1, 0) #1x1
        #self.deconv1_bn = nn.BatchNorm2d(100)
        self.deconv2 = nn.ConvTranspose2d(100, d*8, 3, 1, 0)  #3x3
        #self.deconv2_bn = nn.BatchNorm2d(d*8)
        self.deconv3 = nn.ConvTranspose2d(d*8, d*8, 3, 1, 0) # 5x5
        #self.deconv3_bn = nn.BatchNorm2d(d*8)
        self.deconv4 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)  #10x10
        #self.deconv4_bn = nn.BatchNorm2d(d*4)
        self.deconv5 = nn.ConvTranspose2d(d*4, d*4, 4, 2, 1)  #20x20
        #self.deconv5_bn = nn.BatchNorm2d(d*4)
        self.deconv6 = nn.ConvTranspose2d(d*4, d*4, 3, 2, 0)  #41x41
        #self.deconv6_bn = nn.BatchNorm2d(d*4) #41x41
        self.deconv7 = nn.ConvTranspose2d(d*4, d*2, 5, 2, 0) #85x85
        #self.deconv7_bn = nn.BatchNorm2d(d*2)        
        self.deconv8 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) #170x170
        #self.deconv8_bn = nn.BatchNorm2d(d)
        self.deconv9 = nn.ConvTranspose2d(d, d, 4, 2, 1) #340x340
        #self.deconv9_bn = nn.BatchNorm2d(d)
        self.deconv10 = nn.ConvTranspose2d(d, out_channel, 3, 1, 1) #340x340


    # forward method
    def forward(self, input):
        x = torch.tanh(self.deconv1(input))
        x = F.leaky_relu((self.deconv2(x)),0.2)
        x = F.leaky_relu((self.deconv3(x)),0.2)
        x = F.leaky_relu((self.deconv4(x)),0.2)
        x = F.leaky_relu((self.deconv5(x)),0.2)
        x = F.leaky_relu((self.deconv6(x)),0.2)
        x = F.leaky_relu((self.deconv7(x)),0.2)
        x = F.leaky_relu((self.deconv8(x)),0.2)
        x = F.leaky_relu((self.deconv9(x)),0.2)
        return torch.tanh(self.deconv10(x))


#%% Load the generator papram.
G = generator(2)
G.load_state_dict(torch.load('generator_param.pkl'), strict=True)
G.cuda()

#%% Load the latent variables
zi = np.load('zs.npy')

#%% Re-generating results
z_ = torch.zeros((150, 2, 1, 1))
z_ = Variable(z_.cuda(), requires_grad=True)
z_im = zi
z_im = torch.FloatTensor(z_im).view(-1,2, 1, 1)
z_im = z_im.cuda()
z_.data = z_im

pad = nn.ReplicationPad2d((1,0,1,0))
blur = nn.MaxPool2d(2, stride=1)

G_result = blur(pad(G(z_))

    
#%% Saving data to file
ztemp = z_.data.detach()
    
images = []
my_dpi = 100 # Good default - doesn't really matter
h = 340
w = 340

        
for k in range(150):
    test_image1 = G_result.squeeze().data.cpu().numpy()
    test_image1 = test_image1[:,0,:,:] + test_image1[:,1,:,:]*1j
    fig = plt.figure(figsize=(2*w/my_dpi, h/my_dpi), dpi=my_dpi)

    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow((255*abs(test_image1[k,:,:])), cmap='gray')
    ax1.axis('off')
        
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ztemp.squeeze().cpu().numpy())
    ax2.axvline(x=k,color='r')

    fig.savefig('slices4_result/images/frame_' + str(k) + '.png',bbox_inches=Bbox([[0,0],[2*w/my_dpi,h/my_dpi]]),dpi=my_dpi)
    plt.close()
    img_name = 'slices4_result/images/frame_' + str(k) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('slices4_result/slice4.gif', images, fps=5)
