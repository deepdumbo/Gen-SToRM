#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:06:18 2020

@author: jcb
"""
import torch
import numpy as np
from torch.autograd import Variable

class latentVariable():
    def __init__(self,params,z_in=1,init='interpolate',alpha=[0,0]):
        
        
        self.alpha = alpha
        device = params['device']
        nFrames = params['nFramesDesired']
        siz_l = params['siz_l']
        if(init=='ones'):
            zpnew = 0.5*np.ones((nFrames,siz_l))
            zpnew = np.matmul(zpnew,np.diag(np.linspace(-0.5,0.5,siz_l)))
        elif(init=='random'):
            zpnew = 0.15*np.random.randn(nFrames,siz_l)
        elif(init=='zeros'):
            zpnew = np.zeros((nFrames,siz_l))
        else:
            zinput = np.squeeze(z_in.z_.data.cpu().numpy())
            nin = np.size(zinput,0)
            if(nFrames == nin):
                zpnew = zinput
            else:
                x = np.arange(0,nFrames)
                nf = np.size(zinput,0)          
                xp = np.arange(0,nf)*nFrames/nf
                zpnew = np.zeros((nFrames,np.size(zinput,1)))
                for i in range(np.size(zinput,1)):
                    zpnew[:,i] = np.interp(x, xp, zinput[:,i])                   
        
        z_in = torch.FloatTensor(zpnew).unsqueeze(2).unsqueeze(2)
        z_in = z_in.cuda(device)
        
        
        #self.scalingMatrix = torch.ones(z_in.shape)
        #self.scalingMatrix[:,1,:,:] = 0
        #self.scalingMatrix = self.scalingMatrix.to(device)

        #z_in = z_in*self.scalingMatrix

        self.z_ = torch.zeros((nFrames, siz_l,1,1))
        self.z_ = Variable(self.z_.cuda(device), requires_grad=True)
        self.z_.data = z_in
        
        
              # Creeating the filters    
        #acqDuration = params['TR']*params['nintlPerFrame']*params['nFramesDesired']
        #nCardiacCycles = (acqDuration/params['cardiacPeriod'])
        #nRespCycles = (acqDuration/params['respPeriod'])
        # cutOff = int((2*nCardiacCycles+2*nRespCycles)/4)
        # maxFreq = int(min(nCardiacCycles*2,params['nFramesDesired']))
        # noiseFreq = int(max(maxFreq+1,params['nFramesDesired']/2-10))

        # filterArray = torch.ones(siz_l,int(nFrames/2+1),2) 
        # filterArray[0,0:cutOff,:] = 0
        # if(siz_l > 1):
        #     filterArray[1,cutOff+1:maxFreq,:] = 0
        # if(siz_l > 2):
        #     filterArray[2:siz_l,maxFreq:noiseFreq,:] = 0
            
        # self.filterArray = filterArray.to(device)   
                
        
        
    def Reg(self):
        
        zsmoothness = self.z_[1:,:,:,:]-self.z_[:-1,:,:,:]
        zsmoothness = torch.sum(zsmoothness*zsmoothness,axis=0).squeeze()
        zsmoothness = torch.sum(self.alpha*zsmoothness,axis=0)
        #tmp = self.z_.squeeze()
        #tmp = torch.matmul(tmp.t(),tmp)
        #tmp = tmp - torch.diag(torch.diag(tmp))
        #zcorr = torch.norm(tmp,'fro')
        
        # #znew = torch.transpose(self.z_.squeeze(),0,1)
        # znew = torch.rfft(znew,signal_ndim = 1)*self.filterArray
        # zfilter = torch.norm(znew,'fro')
    
        return(zsmoothness)
        #return(self.alpha[0]*zsmoothness + self.alpha[2]*zfilter)
        #    return(self.alpha[0]*zsmoothness + self.alpha[1]*zcorr + self.alpha[2]*zfilter)

    
 