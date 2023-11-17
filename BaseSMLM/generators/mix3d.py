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
from ..psf.psf3d.psf3d import *
from perlin_noise import PerlinNoise

class Mix3D:
    def __init__(self,config):
    
        self.config = config
        self.cmos_params = [np.load(self.config['gain'])['arr_0'],
                            np.load(self.config['offset'])['arr_0'],
                            np.load(self.config['var'])['arr_0']]
        self.config['pixel_size_axial'] = 2*self.config['zhrange']/self.config['nz']

    def generate(self,margin=12):
        gain, offset, var = self.cmos_params
        xyz_npt = []
        theta = np.zeros((4,self.config['particles']))
        theta[0,:] = np.random.uniform(margin,self.config['nx']-margin,self.config['particles'])
        theta[1,:] = np.random.uniform(margin,self.config['ny']-margin,self.config['particles'])
        zz = self.config['zhrange']
        theta[2,:] = np.random.uniform(-zz,zz,self.config['particles'])
        theta[3,:] = self.config['N0']
        srate, xyz_np = self.get_srate(theta)
        xyz_npt.append(xyz_np)
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
        spikes = self.get_spikes(np.array(xyz_npt))
        return adu, spikes, theta

    def get_brate(self):
        nx,ny = self.config['nx'],self.config['ny']
        noise = PerlinNoise(octaves=10,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        brate = self.config['B0']*(bg/bg.max())
        return brate

    def get_srate(self,theta,patch_hw=12):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        srate = np.zeros((self.config['nx'],self.config['ny']),dtype=np.float32)
        xyz_np = np.zeros((self.config['particles'],3))
        for n in range(self.config['particles']):
            x0,y0,z0,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            sigma_x = sx(z0); sigma_y = sy(z0)
            lam = lamx(X,x0p,sigma_x)*lamy(Y,y0p,sigma_y)
            mu = self.config['texp']*self.config['eta']*N0*lam
            srate[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
            xyz_np[n] = [x0,y0,z0]
        return srate, xyz_np
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self):
        gain, offset, var = self.cmos_params
        noise = np.random.normal(offset,np.sqrt(var),size=(self.config['nx'],self.config['ny']))
        return noise
        
    def get_spikes(self,xyz_np,upsample=4):
        grid_shape = (self.config['nx'],self.config['ny'],self.config['nz'])
        boolean_grid = batch_xyz_to_boolean_grid(xyz_np,
                                                 upsample,
                                                 self.config['pixel_size_lateral'],
                                                 self.config['pixel_size_axial'],
                                                 self.config['zhrange'],
                                                 grid_shape)
        return boolean_grid




