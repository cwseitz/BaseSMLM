import numpy as np
import matplotlib.pyplot as plt
import os
import secrets
import string
import json
import scipy.sparse as sp
import torch

from skimage.io import imsave
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

from ..utils import *
from ..psf.psf2d.psf2d import *
from perlin_noise import PerlinNoise

class Mix2D:
    def __init__(self,config):
    
        self.config = config
        self.cmos_params = [np.load(self.config['gain'])['arr_0'],
                            np.load(self.config['offset'])['arr_0'],
                            np.load(self.config['var'])['arr_0']]

    def sample_uniform_circle(self, x0, y0, r, n_samples):
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        radius = np.sqrt(np.random.uniform(0, 1, n_samples)) * r
        x = x0 + radius * np.cos(theta)
        y = y0 + radius * np.sin(theta)
        return x, y

    def generate(self,r=4,plot=False):
        gain, offset, var = self.cmos_params
        self.nx,self.ny = offset.shape
        _xyz_np = []
        theta = np.zeros((4,self.config['particles']))
        nx,ny = offset.shape
        xsamp,ysamp = self.sample_uniform_circle(nx/2,ny/2,r,self.config['particles'])
        theta[0,:] = xsamp
        theta[1,:] = ysamp
        theta[2,:] = self.config['sigma']
        theta[3,:] = self.config['N0']
        srate, xyz_np = self.get_srate(theta)
        _xyz_np.append(xyz_np)
        brate = self.get_brate()
        signal_electrons = self.shot_noise(srate)
        backrd_electrons = self.shot_noise(brate)
        electrons = signal_electrons + backrd_electrons     
        signal_adu = gain[np.newaxis,:,:]*signal_electrons
        signal_adu = signal_adu.astype(np.int16) #round
        backrd_adu = gain[np.newaxis,:,:]*backrd_electrons
        backrd_adu = backrd_adu.astype(np.int16) #round
        rnoise_adu = self.read_noise()
        rnoise_adu = rnoise_adu.astype(np.int16) #round
        adu = signal_adu + backrd_adu + rnoise_adu
        adu = np.clip(adu,0,None)
        _xyz_np = np.array(_xyz_np)
        spikes = self.get_spikes(_xyz_np)
        if plot:
            self.show(srate,brate,electrons,adu)
        adu = np.squeeze(adu)
        return adu, spikes, theta

    def get_brate(self):
        nx,ny = self.nx,self.ny
        noise = PerlinNoise(octaves=1,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        brate = self.config['B0']*(bg/bg.max())
        return brate

    def get_srate(self,theta,patch_hw=5):
        nx,ny = self.nx,self.ny
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        srate = np.zeros((nx,ny),dtype=np.float32)
        xy_np = np.zeros((self.config['particles'],3))
        for n in range(self.config['particles']):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu = self.config['texp']*self.config['eta']*N0*lam
            srate[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
            xy_np[n] = [x0,y0,0]
        return srate, xy_np
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self):
        gain, offset, var = self.cmos_params
        noise = np.random.normal(offset,np.sqrt(var),size=(self.nx,self.ny))
        return noise
        
    def get_spikes(self,xyz_np,upsample=4):
        grid_shape = (self.nx,self.ny,1)
        boolean_grid = batch_xyz_to_boolean_grid(xyz_np,
                                                 upsample,
                                                 self.config['pixel_size_lateral'],
                                                 1,
                                                 0,
                                                 grid_shape)
        return boolean_grid

    def show(self,srate,brate,electrons,adu):
    
        fig, ax = plt.subplots(1,4,figsize=(8,1.5))
        
        im1 = ax[0].imshow(srate,cmap='gray')
        ax[0].set_xticks([]);ax[0].set_yticks([])
        plt.colorbar(im1, ax=ax[0], label=r'$\mu_{s}$')
        
        im2 = ax[1].imshow(brate,cmap='gray')
        ax[1].set_xticks([]);ax[1].set_yticks([])
        plt.colorbar(im2, ax=ax[1], label=r'$\mu_{b}$')

        im3 = ax[2].imshow(electrons,cmap='gray')
        ax[2].set_xticks([]);ax[2].set_yticks([])
        plt.colorbar(im1, ax=ax[2], label=r'$e^{-}$')
        
        im4 = ax[3].imshow(np.squeeze(adu),cmap='gray')
        ax[3].set_xticks([]);ax[3].set_yticks([])
        plt.colorbar(im4, ax=ax[3], label=r'ADU')
        
        plt.tight_layout()
        plt.show()


class Mix2D_Ring:
    def __init__(self,config):
    
        self.config = config
        self.cmos_params = [np.load(self.config['gain'])['arr_0'],
                            np.load(self.config['offset'])['arr_0'],
                            np.load(self.config['var'])['arr_0']]

    def ring2d(self,n,radius=3):
        phase = np.random.uniform(0,2*np.pi)
        thetas = np.arange(0,n,1)*2*np.pi/n
        xs = radius*np.cos(thetas+phase)
        ys = radius*np.sin(thetas+phase)
        return xs,ys

    def generate(self,r=4,ring_radius=2,plot=False):
        gain, offset, var = self.cmos_params
        nx,ny = offset.shape
        _xyz_np = []
        theta = np.zeros((4,self.config['particles']))
        patch_hw = self.config['patchw']
        nx,ny = offset.shape
        xsamp,ysamp = self.ring2d(self.config['particles'],radius=self.config['ring_radius'])
        x0 = nx//2; y0 = ny//2
        theta[0,:] = xsamp + x0
        theta[1,:] = ysamp + y0
        theta[2,:] = self.config['sigma']
        theta[3,:] = self.config['N0']

        srate, xyz_np = self.get_srate(theta)
        _xyz_np.append(xyz_np)
        brate = self.get_brate()
        signal_electrons = self.shot_noise(srate)
        backrd_electrons = self.shot_noise(brate)
        electrons = signal_electrons + backrd_electrons     
        signal_adu = gain[np.newaxis,:,:]*signal_electrons
        signal_adu = signal_adu.astype(np.int16) #round
        backrd_adu = gain[np.newaxis,:,:]*backrd_electrons
        backrd_adu = backrd_adu.astype(np.int16) #round
        rnoise_adu = self.read_noise()
        rnoise_adu = rnoise_adu.astype(np.int16) #round
        adu = signal_adu + backrd_adu + rnoise_adu
        adu = np.clip(adu,0,None)
        _xyz_np = np.array(_xyz_np)
        spikes = self.get_spikes(_xyz_np)
        if plot:
            self.show(srate,brate,electrons,adu)
        adu = np.squeeze(adu)
        return adu, spikes, theta

    def get_brate(self):
        gain, offset, var = self.cmos_params
        nx,ny = offset.shape
        noise = PerlinNoise(octaves=1,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        brate = self.config['B0']*(bg/bg.max())
        return brate

    def get_srate(self,theta,patch_hw=5):
        gain, offset, var = self.cmos_params
        nx,ny = offset.shape
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        srate = np.zeros((nx,ny),dtype=np.float32)
        xy_np = np.zeros((self.config['particles'],3))
        for n in range(self.config['particles']):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu = self.config['texp']*self.config['eta']*N0*lam
            srate[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
            xy_np[n] = [x0,y0,0]
        return srate, xy_np
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self):
        gain, offset, var = self.cmos_params
        nx,ny = offset.shape
        noise = np.random.normal(offset,np.sqrt(var),size=(nx,ny))
        return noise
        
    def get_spikes(self,xyz_np,upsample=4):
        gain, offset, var = self.cmos_params
        nx,ny = offset.shape
        grid_shape = (nx,ny,1)
        boolean_grid = batch_xyz_to_boolean_grid(xyz_np,
                                                 upsample,
                                                 self.config['pixel_size'],
                                                 1,
                                                 0,
                                                 grid_shape)
        return boolean_grid

    def show(self,srate,brate,electrons,adu):
    
        fig, ax = plt.subplots(1,4,figsize=(8,1.5))
        
        im1 = ax[0].imshow(srate,cmap='gray')
        ax[0].set_xticks([]);ax[0].set_yticks([])
        plt.colorbar(im1, ax=ax[0], label=r'$\mu_{s}$')
        
        im2 = ax[1].imshow(brate,cmap='gray')
        ax[1].set_xticks([]);ax[1].set_yticks([])
        plt.colorbar(im2, ax=ax[1], label=r'$\mu_{b}$')

        im3 = ax[2].imshow(electrons,cmap='gray')
        ax[2].set_xticks([]);ax[2].set_yticks([])
        plt.colorbar(im1, ax=ax[2], label=r'$e^{-}$')
        
        im4 = ax[3].imshow(np.squeeze(adu),cmap='gray')
        ax[3].set_xticks([]);ax[3].set_yticks([])
        plt.colorbar(im4, ax=ax[3], label=r'ADU')
        
        plt.tight_layout()
        plt.show()
        
class Mix2D_SR3:
    """Generates a 2D mixture, with both low 
       resolution and high resolution noiseless/normalized images"""
    def __init__(self,config):
    
        self.config = config
        self.cmos_params = [np.load(self.config['gain'])['arr_0'],
                            np.load(self.config['offset'])['arr_0'],
                            np.load(self.config['var'])['arr_0']]

    def generate_l(self,theta,npixels,patch_hw=5):
        gain, offset, var = self.cmos_params
        srate = self.get_srate(theta,npixels,patch_hw=patch_hw)
        brate = self.get_brate(npixels)
        signal_electrons = self.shot_noise(srate)
        backrd_electrons = self.shot_noise(brate)
        electrons = signal_electrons + backrd_electrons     
        signal_adu = gain[np.newaxis,:,:]*signal_electrons
        signal_adu = signal_adu.astype(np.int16) #round
        backrd_adu = gain[np.newaxis,:,:]*backrd_electrons
        backrd_adu = backrd_adu.astype(np.int16) #round
        rnoise_adu = self.read_noise(npixels)
        rnoise_adu = rnoise_adu.astype(np.int16) #round
        adu = signal_adu + backrd_adu + rnoise_adu
        adu = np.clip(adu,0,None)
        return adu    

    def generate_h(self,theta,npixels):
        srate = self.get_srate(theta,npixels,patch_hw=14)
        return srate   

    def generate(self,r=4,plot=False,patch_hw=5):
        theta = np.zeros((4,self.config['particles']))
        nx,ny = self.config['lpixels'],self.config['lpixels']
        xsamp = np.random.uniform(patch_hw,nx-patch_hw,self.config['particles'])
        ysamp = np.random.uniform(patch_hw,ny-patch_hw,self.config['particles'])
        theta[0,:] = xsamp
        theta[1,:] = ysamp
        theta[2,:] = self.config['sigma']
        theta[3,:] = self.config['N0']

        adu_l = self.generate_l(theta,self.config['lpixels'],patch_hw=patch_hw)
        
        theta[0,:] *= self.config['hpixels']/self.config['lpixels']
        theta[1,:] *= self.config['hpixels']/self.config['lpixels']
        theta[2,:] = 50/self.config['hpixel_size'] #in pixels
        
        adu_h = self.generate_h(theta,self.config['hpixels'])

        
        if plot:
            fig,ax=plt.subplots(1,2)
            ax[0].imshow(np.squeeze(adu_h),cmap='gray')
            ax[1].imshow(np.squeeze(adu_l),cmap='gray')
            plt.show()
            
        return adu_h,adu_l


    def get_brate(self,npixels):
        nx,ny = npixels,npixels
        noise = PerlinNoise(octaves=1,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        brate = self.config['B0']*(bg/bg.max())
        return brate

    def get_srate(self,theta,npixels,patch_hw=5):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        srate = np.zeros((npixels,npixels),dtype=np.float32)
        for n in range(self.config['particles']):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu = self.config['texp']*self.config['eta']*N0*lam
            srate[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
        return srate
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,npixels):
        gain, offset, var = self.cmos_params
        noise = np.random.normal(offset,np.sqrt(var),size=(npixels,npixels))
        return noise
        
    def get_spikes(self,xyz_np,upsample=4):
        grid_shape = (self.config['nx'],self.config['ny'],1)
        boolean_grid = batch_xyz_to_boolean_grid(xyz_np,
                                                 upsample,
                                                 self.config['pixel_size_lateral'],
                                                 1,
                                                 0,
                                                 grid_shape)
        return boolean_grid

class Mix2D_SR3_Ring:
    """Generates 2D particles on a ring, with both low 
       resolution and high resolution noiseless/normalized images"""
    def __init__(self,config):
    
        self.config = config
        self.cmos_params = [np.load(self.config['gain'])['arr_0'],
                            np.load(self.config['offset'])['arr_0'],
                            np.load(self.config['var'])['arr_0']]

    def generate_l(self,theta,npixels,patch_hw=5):
        gain, offset, var = self.cmos_params
        srate = self.get_srate(theta,npixels,patch_hw=patch_hw)
        brate = self.get_brate(npixels)
        signal_electrons = self.shot_noise(srate)
        backrd_electrons = self.shot_noise(brate)
        electrons = signal_electrons + backrd_electrons     
        signal_adu = gain[np.newaxis,:,:]*signal_electrons
        signal_adu = signal_adu.astype(np.int16) #round
        backrd_adu = gain[np.newaxis,:,:]*backrd_electrons
        backrd_adu = backrd_adu.astype(np.int16) #round
        rnoise_adu = self.read_noise(npixels)
        rnoise_adu = rnoise_adu.astype(np.int16) #round
        adu = signal_adu + backrd_adu + rnoise_adu
        adu = np.clip(adu,0,None)
        return adu    

    def generate_h(self,theta,npixels):
        srate = self.get_srate(theta,npixels,patch_hw=10)
        return srate
        
    def ring2d(self,n,radius=3):
        phase = np.random.uniform(0,2*np.pi)
        thetas = np.arange(0,n,1)*2*np.pi/n
        xs = radius*np.cos(thetas+phase)
        ys = radius*np.sin(thetas+phase)
        return xs,ys

    def generate(self,r=4,plot=False,patch_hw=5,ring_radius=3):
        self.theta = np.zeros((4,self.config['particles']))
        nx,ny = self.config['lpixels'],self.config['lpixels']
        xsamp,ysamp = self.ring2d(self.config['particles'],radius=ring_radius)
        x0 = np.random.uniform(patch_hw+ring_radius,nx-patch_hw-ring_radius)
        y0 = np.random.uniform(patch_hw+ring_radius,ny-patch_hw-ring_radius)
        self.theta[0,:] = xsamp + x0
        self.theta[1,:] = ysamp + y0
        self.theta[2,:] = self.config['sigma']
        self.theta[3,:] = self.config['N0']

        adu_l = self.generate_l(self.theta,self.config['lpixels'],patch_hw=patch_hw)
        
        self.theta[0,:] *= self.config['hpixels']/self.config['lpixels']
        self.theta[1,:] *= self.config['hpixels']/self.config['lpixels']
        self.theta[2,:] = 50/self.config['hpixel_size'] #in pixels
        
        adu_h = self.generate_h(self.theta,self.config['hpixels'])

        if plot:
            fig,ax=plt.subplots(1,2)
            ax[0].imshow(np.squeeze(adu_h),cmap='gray')
            ax[1].imshow(np.squeeze(adu_l),cmap='gray')
            plt.show()
            
        return adu_h,adu_l


    def get_brate(self,npixels):
        nx,ny = npixels,npixels
        noise = PerlinNoise(octaves=1,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        brate = self.config['B0']*(bg/bg.max())
        return brate

    def get_srate(self,theta,npixels,patch_hw=5):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        srate = np.zeros((npixels,npixels),dtype=np.float32)
        for n in range(self.config['particles']):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu = self.config['texp']*self.config['eta']*N0*lam
            srate[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
        return srate
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,npixels):
        gain, offset, var = self.cmos_params
        noise = np.random.normal(offset,np.sqrt(var),size=(npixels,npixels))
        return noise
        
    def get_spikes(self,xyz_np,upsample=4):
        grid_shape = (self.config['nx'],self.config['ny'],1)
        boolean_grid = batch_xyz_to_boolean_grid(xyz_np,
                                                 upsample,
                                                 self.config['pixel_size_lateral'],
                                                 1,
                                                 0,
                                                 grid_shape)
        return boolean_grid
