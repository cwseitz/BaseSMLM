import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from perlin_noise import PerlinNoise

class Iso2D:
    def __init__(self,theta,setup_params):
        self.theta = theta
        self.setup_params = setup_params
        self.cmos_params = [setup_params['nx'],setup_params['ny'],
                           setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']] 
        
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
        x0,y0,sigma,N0 = self.theta
        alpha = np.sqrt(2)*sigma
        x = np.arange(0,self.setup_params['nx']); y = np.arange(0,self.setup_params['ny'])
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
        lam = lambdx*lambdy
        i0 = N0*self.setup_params['texp']*self.setup_params['eta'] #use theta N0 (not setup_params['N0']) - need to fix this
        rate = i0*lam
        return rate

    def get_brate(self):
        noise = PerlinNoise(octaves=10,seed=None)
        nx,ny = self.setup_params['nx'],self.setup_params['ny']
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        bg_rate = self.setup_params['B0']*(bg/bg.max())
        return bg_rate
        
    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate) 
        return electrons
                
    def read_noise(self,adu):
        nx,ny = adu.shape
        noise = np.random.normal(self.cmos_params[5],np.sqrt(self.cmos_params[6]),size=(nx,ny))
        adu = adu + noise
        adu = np.clip(adu,0,None)
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

