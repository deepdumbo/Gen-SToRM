#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import h5py
import numpy.lib.recfunctions as rf
from numpy import linalg as LA
import math


def optimize_generator(dop,G,z,params,train_epoch=1,proj_flag=True):

    lr_g = params['lr_g']
    lr_z = params['lr_z']
    gpu = params['device']
    
    #optimizer = optim.SGD([
    #{'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_g},
    #{'params': z.z_, 'lr': lr_z}
    #], momentum=(0.9))
    optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_g},
    {'params': z.z_, 'lr': lr_z}
    ], betas=(0.4, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=30, verbose=True, min_lr=1e-6)
   
    train_hist = {}
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    loss = nn.MSELoss(reduction='sum')
    
    G_old = G.state_dict()
    z_old = z.z_.data
    divergence_counter = 0          
    print('training start!')
    start_time = time.time()
    G_losses = []
    SER = np.zeros(train_epoch)
    for epoch in range(train_epoch):
        epoch_start_time = time.time()
        G_result = G(z.z_)
        if(proj_flag):
            G_result_projected = dop.P(G_result.unsqueeze(1))
        else:
            G_result_projected = G_result.unsqueeze(1)
        optimizer.zero_grad()
        
        
        G_loss = loss(G_result_projected,dop.Atb.to(gpu))  # data conisistency
        G_loss += dop.image_energy(G_result)  # image regularization to zero out regions outside maks
        G_loss +=  G.weightl1norm()    # Netowrk regularization
        # G_loss += G.gradient_penalty(G_result,z.z_)
        G_loss += z.Reg()      # latent variable regularization
        G_loss.backward()
        
    
        optimizer.step()
        G_losses.append(G_loss.item())
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
   
        # If cost increase, load an old state and decrease step size
        #print(G_loss.item())
        if(epoch >10):
            if((G_loss.item() > 1.15*train_hist['G_losses'][-1])): # higher cost
                G.load_state_dict(G_old)
                z.z_.data = z_old
                print('loading old state; reducing stp siz')
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.98
                divergence_counter = divergence_counter+1
            else:       # lower cost; converging
                divergence_counter = divergence_counter-1
                if((divergence_counter<0)):
                    divergence_counter=0
                train_hist['G_losses'].append(G_loss.item())
        else:
            train_hist['G_losses'].append(G_loss.item())
                

        if(divergence_counter>100):
            print('Optimization diverging; exiting')
            return G,z,train_hist
        
        G_old = G.state_dict()
        z_old = z.z_.data
    

        #Display results

        if(np.mod(epoch,50)==0):
            test_image1 = G_result[-1,:,:,:].squeeze().cpu().data.numpy()
            test_image1 = test_image1[0,:,:] + test_image1[1,:,:]*1j
        
            # saving states
             #torch.save(G.state_dict(), "generator_param.pkl")
            #zi = z.z_.data.cpu().numpy()
            #np.save('zs.npy', zi)

             
            plt.subplot(1, 2, 1)
            plt.imshow(abs(test_image1),cmap='gray')
        
            plt.subplot(1, 2, 2)
            temp = z.z_.data.squeeze().cpu().numpy()
            plt.plot(temp)
            plt.pause(0.00001)
            print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))
   
    print('Optimization done in %d seconds', time.time()-start_time)
    return G,z,train_hist,SER
