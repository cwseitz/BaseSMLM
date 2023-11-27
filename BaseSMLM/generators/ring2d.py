import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.psf.psf2d.psf2d import *

class Generator:
    def __init__(self):
        pass

class Ring2D(Generator):
    def __init__(self,config):
        super().__init__()
        self.config = config

    def _mu(self,theta,npixels,patch_hw=5):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        srate = np.zeros((npixels,npixels),dtype=np.float32)
        for n in range(self.config['particles']):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu = self.config['eta']*N0*lam
            srate[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += mu
        return srate

    def ring(self,n,radius=3,phase=0):
        thetas = np.arange(0,n,1)*2*np.pi/n
        xs = radius*np.cos(thetas+phase)
        ys = radius*np.sin(thetas+phase)
        return xs,ys

    def forward(self,r=4,show=False,patch_hw=5,ring_radius=10,gain=2.2):
        theta = np.zeros((4,self.config['particles']))
        nx,ny = self.config['npixels'],self.config['npixels']
        xsamp,ysamp = self.ring(self.config['particles'],radius=ring_radius)
        x0 = nx/2; y0 = ny/2
        theta[0,:] = xsamp + x0
        theta[1,:] = ysamp + y0
        theta[2,:] = self.config['sigma']
        theta[3,:] = self.config['N0']
        mu = self._mu(theta,self.config['npixels'],patch_hw=patch_hw)
        adu = gain*self.shot_noise(mu)
        adu = self.read_noise(adu)
        if show: self.show(theta,mu,adu)
        return adu,theta

    def shot_noise(self,rate):
        electrons = np.random.poisson(lam=rate)
        return electrons

    def read_noise(self,adu,offset=0.0,variance=0.0):
        nx,ny = adu.shape
        noise = np.random.normal(offset,variance,size=(nx,ny))
        adu = adu + noise
        adu = np.clip(adu,0,None)
        return adu

    def show(self,theta,mu,adu):
        fig,ax=plt.subplots(1,3)
        ax[0].scatter(theta[1,:],theta[0,:],color='black',s=5)
        ax[0].set_aspect(1.0)
        ax[1].imshow(mu,cmap=plt.cm.BuGn_r)
        ax[2].imshow(adu,cmap='gray')
        ax[0].set_xlim([0,self.config['npixels']])
        ax[0].set_ylim([0,self.config['npixels']])
        ax[0].invert_yaxis()
