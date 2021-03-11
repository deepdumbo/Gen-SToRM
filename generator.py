#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

class generator(nn.Module):
    # initializers
    def __init__(self,params, out_channel=2):
        super().__init__()
        factor = params['factor']
        siz_latent = params['siz_l']
        d = params['gen_base_size']
        self.gen_reg = params['gen_reg']
        self.nlevels = int(9-np.log2(factor)) 
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
        
        if(self.nlevels ==6):
            self.deconv7 = nn.ConvTranspose2d(d*4,out_channel,3,1,0) #43x43
        elif(self.nlevels==7):
            self.deconv7 = nn.ConvTranspose2d(d*4, d*2, 5, 2, 0)  #85x85
            #self.deconv7_bn = nn.BatchNorm2d(d*2)
            self.deconv8 = nn.ConvTranspose2d(d*2,out_channel,3,1,1) #85x85
        elif(self.nlevels==8):
            self.deconv7 = nn.ConvTranspose2d(d*4, d*2, 5, 2, 0) #85x85
            #self.deconv7_bn = nn.BatchNorm2d(d*2)        
            self.deconv8 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) #170x170
            #self.deconv8_bn = nn.BatchNorm2d(d)
            self.deconv9 = nn.ConvTranspose2d(d, out_channel, 3, 1, 1) #170x170
        else:
            self.deconv7 = nn.ConvTranspose2d(d*4, d*2, 5, 2, 0) #85x85
            #self.deconv7_bn = nn.BatchNorm2d(d*2)        
            self.deconv8 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) #170x170
            #self.deconv8_bn = nn.BatchNorm2d(d)
            self.deconv9 = nn.ConvTranspose2d(d, d, 4, 2, 1) #340x340
            #self.deconv9_bn = nn.BatchNorm2d(d)
            self.deconv10 = nn.ConvTranspose2d(d, out_channel, 3, 1, 1) #340x340
    
    def weight_init(self):
        for layer in self._modules:
            if layer.find('Conv') != -1:
                layer.weight.data.normal_(0.0, 0.02)
            elif layer.find('BatchNorm') != -1:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)


    # forward method
    def forward(self, input):
        x = torch.tanh(self.deconv1(input))
        x = F.leaky_relu((self.deconv2(x)),0.2)
        x = F.leaky_relu((self.deconv3(x)),0.2)
        x = F.leaky_relu((self.deconv4(x)),0.2)
        x = F.leaky_relu((self.deconv5(x)),0.2)
        x = F.leaky_relu((self.deconv6(x)),0.2)
        #x = torch.tanh(self.deconv1_bn(self.deconv1(input)))
        #x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)),0.2)
        #x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)),0.2)
        #x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)),0.2)
        #x = F.leaky_relu(self.deconv5_bn(self.deconv5(x)),0.2)
        #x = F.leaky_relu(self.deconv6_bn(self.deconv6(x)),0.2)

        if(self.nlevels==6):
            return torch.tanh(self.deconv7(x))
        elif(self.nlevels==7):
            x = F.leaky_relu((self.deconv7(x)),0.2)
            #x = F.leaky_relu(self.deconv7_bn(self.deconv7(x)),0.2)
            return torch.tanh(self.deconv8(x))
        elif(self.nlevels==8):
            x = F.leaky_relu((self.deconv7(x)),0.2)
            x = F.leaky_relu((self.deconv8(x)),0.2)
            #x = F.leaky_relu(self.deconv7_bn(self.deconv7(x)),0.2)
            #x = F.leaky_relu(self.deconv8_bn(self.deconv8(x)),0.2)
            return torch.tanh(self.deconv9(x))
        else:
            x = F.leaky_relu((self.deconv7(x)),0.2)
            x = F.leaky_relu((self.deconv8(x)),0.2)
            x = F.leaky_relu((self.deconv9(x)),0.2)
            #x = F.leaky_relu(self.deconv7_bn(self.deconv7(x)),0.2)
            #x = F.leaky_relu(self.deconv8_bn(self.deconv8(x)),0.2)
            #x = F.leaky_relu(self.deconv9_bn(self.deconv9(x)),0.2)
            return torch.tanh(self.deconv10(x))
#%%        
    def weightl1norm(self):
        L1norm = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                L1norm = L1norm + torch.norm(param, 1)
        return(self.gen_reg*L1norm)
    
    def gradient_penalty(self, output, input):
        musk = torch.ones_like(output)
        gradients = grad(output, input, grad_outputs=musk,
                         retain_graph=True, create_graph=True,
                         allow_unused=True)[0]
        gradients = gradients.view(gradients.size(0),-1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.norm(gradient_norm, 2)
        return(self.gen_reg*gradient_penalty)
  