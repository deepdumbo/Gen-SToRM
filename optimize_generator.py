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

# def tv_loss(var, tv_weight):
#     h_cur = var[:,:-1,:]
#     h_lat = var[:,1:,:]
#     w_cur = var[:,:,:-1]
#     w_lat = var[:,:,1:]
#     h_result = h_lat - h_cur
#     w_result = w_lat - w_cur
#     h_result = h_result*h_result
#     w_result = w_result*w_result
#     result = tv_weight*(torch.sum(h_result) + torch.sum(w_result))
#     return result


def optimize_generator(dop,G,z,params,train_epoch=1,proj_flag=True):
    
    f = h5py.File('slice2_500.mat','r')
    variables = f.items()
    for var in variables:
        name = var[0]
        data = var[1]
        if type(data) is h5py.Dataset:
            value = data[()]
    
    Imgs = rf.structured_to_unstructured(value) 
    Imgs = Imgs.transpose((0, 2, 1, 3)) 
    Orig = Imgs[10:160,:,:,:]
    Orig = Orig[:,:,:,0] + Orig[:,:,:,1]*1j #150x340x340

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
        
        # G_s = G_result.squeeze().cpu().data.numpy()
        # G_s = G_s[:,0,:,:] + G_s[:,1,:,:]*1j
        # G_s = abs(G_s)
        # G_s = torch.tensor(G_s).cuda()
        
        G_loss = loss(G_result_projected,dop.Atb.to(gpu))  # data conisistency
        G_loss += dop.image_energy(G_result)  # image regularization to zero out regions outside maks
        G_loss +=  G.weightl1norm()    # Netowrk regularization
        # G_loss += G.gradient_penalty(G_result,z.z_)
        G_loss += z.Reg()      # latent variable regularization
        # G_loss += tv_loss(G_s,0.01)
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
        
        # #Compute the SER at the last level
        # z_value = z_old.cpu().detach().numpy().squeeze()
        # if (np.size(z_value,0)==150):
        #     recon = G_result[:,:,:,:].squeeze().cpu().data.numpy()
        #     recon = recon[:,0,:,:] + recon[:,1,:,:]*1j #150x340x340
        #     SER_sub = 0
        #     for i in range(150):
        #         orig_sub = Orig[i,:,:]
        #         # orig_sub = orig_sub.flatten()
        #         orig_sub = abs(orig_sub)
        #         recon_sub = recon[i,:,:]
        #         recon_sub = abs(recon_sub)
        #         orig_sub = ((orig_sub - np.amin(orig_sub[:]))/(np.amax(orig_sub[:])-np.amin(orig_sub[:])))*(np.amax(recon_sub[:])+0.08)
        #         orig_sub = orig_sub.flatten()
        #         recon_sub = recon_sub.flatten()
        #         # recon_sub = abs(recon_sub)
        #         # scaling = np.dot(orig_sub,recon_sub)/np.dot(orig_sub,orig_sub)
        #         # recon_sub = recon_sub/scaling
        #         diff = orig_sub - recon_sub
        #         ser = 20*math.log10(LA.norm(orig_sub)/LA.norm(diff))
        #         SER_sub = SER_sub + ser
            
        #     SER_sub = SER_sub/150
        #     SER[epoch] = SER_sub
        # else:
        #     x = np.arange(0,150)
        #     nf = np.size(z_value,0)  
        #     xp = np.arange(0,nf)*150/nf
        #     zpnew = np.zeros((150,np.size(z_value,1)))
        #     for i in range(np.size(z_value,1)):
        #         zpnew[:,i] = np.interp(x, xp, z_value[:,i])
        #     z_in = torch.FloatTensor(zpnew).view(-1, 2, 1, 1)
        #     z_in = z_in.cuda()
        #     G_result = G(z_in)
        #     recon = G_result[:,:,:,:].squeeze().cpu().data.numpy()
        #     recon = recon[:,0,:,:] + recon[:,1,:,:]*1j #150x340x340
        #     SER_sub = 0
        #     for i in range(150):
        #         orig_sub = Orig[i,:,:]
        #         orig_sub = orig_sub.flatten()
        #         orig_sub = abs(orig_sub)
        #         scale1 = np.amax(orig_sub[:])
        #         recon_sub = recon[i,:,:]
        #         recon_sub = recon_sub.flatten()
        #         recon_sub = abs(recon_sub)
        #         scale2 = np.amax(recon_sub[:])
        #         scaling = scale2/scale1
        #         recon_sub = recon_sub/scaling
        #         diff = orig_sub - recon_sub
        #         ser = 20*math.log10(LA.norm(orig_sub)/LA.norm(diff))
        #         SER_sub = SER_sub + ser
            
        #     SER_sub = SER_sub/150
        #     SER[epoch] = SER_sub
    

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