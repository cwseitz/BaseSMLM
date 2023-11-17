import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from perlin_noise import PerlinNoise
from ..psf import psf3d

class Iso3D:
    def __init__(self,theta,config):
        self.theta = theta
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']] 
        
    def generate(self,plot=False):
        srate = self.get_srate()
        brate = self.get_brate()
        electrons = self.shot_noise(srate+brate)              
        adu = self.cmos_params[4]*(electrons)
        adu = self.read_noise(adu)
        adu = adu.astype(np.int16) #digitize
        if plot:
            self.show(srate,brate,electrons,adu)
        return adu
        
    def get_srate(self):
        ntheta = self.theta.shape
        x0,y0,z0,N0 = self.theta
        sigma_x = psf3d.sx(z0)
        sigma_y = psf3d.sy(z0)
        x = np.arange(0,self.config['nx']); y = np.arange(0,self.config['ny'])
        X,Y = np.meshgrid(x,y)
        lam = psf3d.lamx(X,x0,sigma_x)*psf3d.lamy(Y,y0,sigma_y)
        rate = N0*self.config['texp']*self.config['eta']*lam  #use theta N0 (not config['N0']) - need to fix this
        return rate

    def get_brate(self):
        noise = PerlinNoise(octaves=10,seed=None)
        nx,ny = self.config['nx'],self.config['ny']
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        bg_rate = self.config['B0']*(bg/bg.max())
        return bg_rate
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate) 
        return electrons
                
    def read_noise(self,adu):
        offset = self.cmos_params[3]
        var = self.cmos_params[4]
        nx,ny = offset.shape
        noise = np.random.normal(offset,np.sqrt(var),size=(nx,ny))
        adu = adu + noise
        adu = np.clip(adu,0.0,None)
        return adu
                 
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
        
        im4 = ax[3].imshow(adu,cmap='gray')
        ax[3].set_xticks([]);ax[3].set_yticks([])
        plt.colorbar(im4, ax=ax[3], label=r'ADU')
        
        plt.tight_layout()
        plt.show()


