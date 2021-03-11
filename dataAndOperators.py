#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:20:39 2020

@author: jcb
"""

from os import path
import numpy as np
import torch
#from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft
from torchkbnufft.torchkbnufft import ToepSenseNufft
from torchkbnufft.torchkbnufft.nufft.toep_functions import calc_toep_kernel
#rom torchkbnufft.torchkbnufft.mri import dcomp_calc

from espirit.espirit import espirit, fft
import pickle
import mat73

#%% Reads the data and operators

class dataAndOperators:
    def __init__(self,params):
        self.params = params
        dtype = params['dtype']
        gpu=torch.device(params['device'])
        self.gpu = gpu
        # Reading h data from mat file
        #----------------------------------------------
        if(params['filename'][-3:-1] =='ma'):  # mat file
            fnamepickle = params['filename'].replace('.mat','.pickle')    
            if(not(path.exists(fnamepickle))):
                data_dict = mat73.loadmat(params['filename'])
                kdata = data_dict['kdata']
                ktraj=np.asarray(data_dict['k'])    
                dcf=np.asarray(data_dict['dcf'])

                # save with pickle for fast reading
                with open(fnamepickle, 'wb') as f:
                    pickle.dump([kdata,ktraj,dcf],f,protocol=4)
            else:
                with open(fnamepickle, 'rb') as f:
                    [kdata,ktraj,dcf] = pickle.load(f)
    
        
        else: # read pickle file
            fname = params['filename']  
            with open(fname, 'rb') as f:
                [kdata,ktraj,dcf] = pickle.load(f)
    
        #Reshaping the variables
        #----------------------------------------------

        kdata = np.squeeze(kdata[:,:,:,params['slice']])
        kdata = kdata.astype(np.complex64)
        ktraj=ktraj.astype(np.complex64)
    
        kdata=np.transpose(kdata,(1,2,0))
        dcf = np.transpose(dcf,(1,0))
        ktraj = np.transpose(ktraj,(1,0))


        # Reducing the image size if factor < 1
        #----------------------------------------------

        im_size = np.int_(np.divide(params["im_size"],params["factor"]))
        self.im_size = im_size
        # ktrajsq = np.max(np.abs(ktraj),axis=0)
        # indices = np.squeeze(np.argwhere(ktrajsq< 0.5/params['factor']))
        # ktraj = ktraj[:,indices]*params['factor']
        # kdata = kdata[:,:,indices]
        # dcf = dcf[:,indices]
        ktraj=np.squeeze(ktraj)*2*np.pi
        nintlvs = np.size(kdata,1)
        nintlvsToDelete = 240
        nintlvsLeft = nintlvs - nintlvsToDelete
        self.nintlvs = nintlvsLeft

        #CoilCombination
        #----------------------------------------------
        nch = np.size(kdata,0)
        nkpts = np.size(kdata,2)
        self.nkpts = nkpts
        kdata = kdata[:,nintlvsToDelete:nintlvs,:]
        ktraj = ktraj[nintlvsToDelete:nintlvs,:]
        dcf = dcf[nintlvsToDelete:nintlvs,:]
        kdataSingleFrame = np.reshape(kdata,(nch,nintlvsLeft*nkpts))
        ktrajSingleFrame = np.reshape(ktraj,(1,nintlvsLeft*nkpts))

        # Coil combine
        #----------------------------------------------
        thres=0.95
        Rs=np.real(kdataSingleFrame@np.transpose(np.conj(kdataSingleFrame)))
        [w,v]=np.linalg.eig(Rs)

        ind=np.flipud(np.argsort(w))
        # ind=np.argsort(w)
        w=w[ind]
        v=v[:,ind]
        w=w/sum(w)
        w=np.cumsum(w)
        nvch=np.min(np.where(w>thres))
        kdataSingleFrame=np.transpose(v[:,0:nvch])@kdataSingleFrame
        nch=kdataSingleFrame[:,0].size
        self.nch = nch

        # Estimating coil images and coil senstivity maps
        #----------------------------------------------


        ktrajSingleFrame = np.stack((np.real(ktrajSingleFrame), np.imag(ktrajSingleFrame)),axis=1)
        dcfSingleFrame = np.reshape(dcf,(1,nintlvsLeft*nkpts)) 
        kdataSingleFrameUW = kdataSingleFrame
        kdataSingleFrame = kdataSingleFrame * dcfSingleFrame[None,:]
        kdataSingleFrame = np.stack((np.real(kdataSingleFrame), np.imag(kdataSingleFrame)),axis=2)
        kdataSingleFrameUW = np.stack((np.real(kdataSingleFrameUW), np.imag(kdataSingleFrameUW)),axis=1)
        kdataSingleFrameUW = np.expand_dims(kdataSingleFrameUW,axis=0)


        nintlPerFrame = params['nintlPerFrame']
        Nframes = np.int(nintlvsLeft/nintlPerFrame)
        if(Nframes > params['nFramesDesired']):
            Nframes = params['nFramesDesired']
    
        # startval = Nframes*nintlPerFrame*nkpts
        #endval = Nframes*nintlPerFrame*nkpts
        
        # startval = 20*nintlPerFrame*nkpts
        endval = Nframes*nintlPerFrame*nkpts

        self.kdataSingleFrameUW = kdataSingleFrameUW
        self.ktrajSingleFrame = ktrajSingleFrame
        self.dcfSingleFrame = dcfSingleFrame
        
        # Omitting two initial frames
        kdata = np.reshape(kdataSingleFrameUW[:,:,:,0:endval],(1,nch,2,Nframes,nintlPerFrame*nkpts))
        ktraj = np.reshape(ktrajSingleFrame[:,:,0:endval],(1,2,Nframes,nintlPerFrame*nkpts))
        dcf = np.reshape(dcfSingleFrame[:,0:endval],(Nframes,1,1,nintlPerFrame*nkpts))
        # kdata = np.reshape(kdataSingleFrameUW[:,:,:,-startval-1:-1],(1,nch,2,Nframes,nintlPerFrame*nkpts))
        # ktraj = np.reshape(ktrajSingleFrame[:,:,-startval-1:-1],(1,2,Nframes,nintlPerFrame*nkpts))
        # dcf = np.reshape(dcfSingleFrame[:,-startval-1:-1],(Nframes,1,1,nintlPerFrame*nkpts))
        kdata = np.transpose(kdata,(3,1,2,4,0))
        ktraj = np.transpose(ktraj,(2,1,3,0))
        dcf = dcf/nintlPerFrame/nintlPerFrame
        
        kdataSingleFrame = torch.tensor(kdataSingleFrame).to(dtype)
        ktrajSingleFrame = torch.tensor(ktrajSingleFrame).to(dtype)
    
        # convert them to gpu
        kdataSingleFrame=kdataSingleFrame.to(gpu)
        ktrajSingleFrame=ktrajSingleFrame.to(gpu)

        adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
        adjnuf_ob=adjnuf_ob.to(gpu)
        
        #nuf_ob = KbNufft(im_size=im_size).to(dtype)
        #nuf_ob=nuf_ob.to(gpu)
        
        coilimages=torch.zeros((1,nch,2,im_size[0],im_size[1]))
        for i in range(nch):
            coilimages[:,i,...] = adjnuf_ob(kdataSingleFrame[:,i,:,:].unsqueeze(1),ktrajSingleFrame)
            
        X=coilimages.cpu().numpy()
        X= X[:,:,0,...]+X[:,:,1,...]*1j
        X=np.transpose(X,(2,3,0,1))

        # ESPIRI
        x_f = fft(X, (0, 1))
        csmTrn = espirit(x_f, 6, 24, 0.04, 0.8925)
        csm=csmTrn[:,:,0,:,0]
        csm=np.transpose(csm,(2,0,1))
        smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
        smapT = torch.tensor(smap).to(dtype)
        smapT = smapT.unsqueeze(0)
        smapT=smapT.to(gpu)

        ktrajSingleFrame = 1
        kdataSingleFrame = 1
        coilimages = 1
        dcfSingleFrame = 1
        adjnuf_ob = 1

        # convert them to gpu
        kdata = torch.tensor(kdata).to(dtype).squeeze(4)
        ktraj = torch.tensor(ktraj).to(dtype).squeeze(3)
        dcf = torch.tensor(dcf).to(dtype)

        kdata=kdata.to(gpu)
        ktraj=ktraj.to(gpu)
        dcf = dcf.to(gpu)

         #dcomp = dcomp_calc.calculate_radial_dcomp_pytorch(nufft_ob, adjnufft_ob, ktraj)
        self.smap = smap
        smap = np.tile(np.expand_dims(smap,axis=0),[Nframes,1,1,1,1])
        smapT = torch.tensor(smap).to(dtype)
        smapT=smapT.to(gpu)

        nufft_ob = MriSenseNufft(im_size=tuple(im_size), smap=smapT).to(dtype)
        nufft_ob = nufft_ob.to(gpu)
        adjnufft_ob = AdjMriSenseNufft(im_size=tuple(im_size), smap=smapT).to(dtype)
        adjnufft_ob = adjnufft_ob.to(gpu)

        self.toep_ob = ToepSenseNufft(smap=smapT)
        Atb = adjnufft_ob(kdata*dcf,ktraj)
        maxvalue = Atb.max()
        Atb = Atb/maxvalue/2

        self.toep_ob = self.toep_ob.to(gpu)
        self.dcomp_kern = calc_toep_kernel(adjnufft_ob, ktraj, weights=dcf)  # with density compensation
        self.Atb = Atb
    
        temp = self.toep_ob(self.Atb,self.dcomp_kern)
        maxvalue = temp.max()
        self.dcomp_kern = self.dcomp_kern/maxvalue
        
        self.dcomp_kern.cpu()
        self.toep_ob = self.toep_ob.cpu()
        self.Atb.cpu()
        self.mask = self.Atb.squeeze(1).abs() == 0.00

        #self.ktraj = ktraj
        #self.kdata = kdata
        #self.dcf = dcf
       
        #self.dcomp_kern = dcomp_ker

# Define operators        
        
    # Projection operator
    def P(self,x):
        return self.toep_ob.to(self.gpu)(x, self.dcomp_kern.to(self.gpu))
    
     # image energy; norm outside the mask
    def image_energy(self,x):
        return torch.norm(x*self.mask.to(self.gpu),'fro')

    def changeNumFrames(self,params):
        nintlPerFrame = params['nintlPerFrame']
        nFramesDesired = params['nFramesDesired']
        Nframes = np.int(self.nintlvs/nintlPerFrame)
        if(Nframes > nFramesDesired):
            Nframes = nFramesDesired
            
    
        # startval = 20*nintlPerFrame*self.nkpts
        endval = Nframes*nintlPerFrame*self.nkpts
        
        kdata = np.reshape(self.kdataSingleFrameUW[:,:,:,0:endval],(1,self.nch,2,Nframes,nintlPerFrame*self.nkpts))
        ktraj = np.reshape(self.ktrajSingleFrame[:,:,0:endval],(1,2,Nframes,nintlPerFrame*self.nkpts))
        dcf = np.reshape(self.dcfSingleFrame[:,0:endval],(Nframes,1,1,nintlPerFrame*self.nkpts))

        # kdata = np.reshape(self.kdataSingleFrameUW[:,:,:,-startval-1:-1],(1,self.nch,2,Nframes,nintlPerFrame*self.nkpts))
        # ktraj = np.reshape(self.ktrajSingleFrame[:,:,-startval-1:-1],(1,2,Nframes,nintlPerFrame*self.nkpts))
        # dcf = np.reshape(self.dcfSingleFrame[:,-startval-1:-1],(Nframes,1,1,nintlPerFrame*self.nkpts))
        kdata = np.transpose(kdata,(3,1,2,4,0))
        ktraj = np.transpose(ktraj,(2,1,3,0))
        dcf = dcf/nintlPerFrame/nintlPerFrame
        
        # convert them to gpu
        kdata = torch.tensor(kdata).to(self.params['dtype']).squeeze(4)
        ktraj = torch.tensor(ktraj).to(self.params['dtype']).squeeze(3)
        dcf = torch.tensor(dcf).to(self.params['dtype'])

        kdata=kdata.to(self.params['device'])
        ktraj=ktraj.to(self.params['device'])
        dcf = dcf.to(self.params['device'])

         #dcomp = dcomp_calc.calculate_radial_dcomp_pytorch(nufft_ob, adjnufft_ob, ktraj)
        smap = np.tile(np.expand_dims(self.smap,axis=0),[Nframes,1,1,1,1])
        smapT = torch.tensor(smap).to(self.params['dtype'])
        smapT=smapT.to(self.params['device'])

        #nufft_ob = MriSenseNufft(im_size=tuple(im_size), smap=smapT).to(dtype)
        #nufft_ob = nufft_ob.to(gpu)
        adjnufft_ob = AdjMriSenseNufft(im_size=tuple(self.im_size), smap=smapT).to(self.params['dtype'])
        adjnufft_ob = adjnufft_ob.to(self.params['device'])

        self.toep_ob = ToepSenseNufft(smap=smapT)
        Atb = adjnufft_ob(kdata*dcf,ktraj)
        maxvalue = Atb.max()
        Atb = Atb/maxvalue/2

        self.toep_ob = self.toep_ob.to(self.params['device'])
        self.dcomp_kern = calc_toep_kernel(adjnufft_ob, ktraj, weights=dcf)  # with density compensation
        self.Atb = Atb
    
        temp = self.toep_ob(self.Atb,self.dcomp_kern)
        maxvalue = temp.max()
        self.dcomp_kern = self.dcomp_kern/maxvalue
        self.params = params
        
        self.dcomp_kern.cpu()
        self.toep_ob = self.toep_ob.cpu()
        self.Atb = self.Atb.cpu()
        self.mask = self.Atb.squeeze(1).abs() == 0.00
     # orward operator
    #def A(self,x):
    #    return self.nufft_obj(x, self.ktraj)
    
    
    #def At(self,x):
    #    return self.nufft_obj(x, self.ktraj)
   
    #def At_weighted(self,x):
    #    return self.nufft_obj(x*self.dcf, self.ktraj)