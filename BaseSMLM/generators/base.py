import numpy as np
import matplotlib.pyplot as plt
from BaseSMLM.psf.psf2d.psf2d import *

class Density:
    def __init__(self):
        pass
        
class Disc(Density):
    """Uniform distribution on a disc"""
    def __init__(self,radius):
        super().__init__()
        self.radius=radius
    def sample(self,n):
        theta = np.random.uniform(0,2*np.pi,n)
        radius = self.radius*np.sqrt(np.random.uniform(0, 1, n))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y
        
class Ring(Density):
    """Equally spaced delta functions on a ring with a random phase"""
    def __init__(self,radius):
        super().__init__()
        self.radius=radius
    def sample(self,n):
        thetas = np.arange(0,n,1)*2*np.pi/n
        phase = np.random.uniform(0,2*np.pi)
        x = self.radius * np.cos(thetas+phase)
        y = self.radius * np.sin(thetas+phase)
        return x,y

class Generator:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
    def _mu_s(self,theta,texp=1.0,eta=1.0,N0=1.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        mu = np.zeros((self.nx,self.ny),dtype=np.float32)
        nspots,ntheta = theta.shape
        i0 = eta*N0*texp
        for n in range(nspots):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            this_mu = i0*lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += this_mu
        return mu

    def _mu_b(self,B0):
        nx,ny = self.nx,self.ny
        noise = PerlinNoise(octaves=1,seed=None)
        bg = [[noise([i/nx,j/ny]) for j in range(nx)] for i in range(ny)]
        bg = 1 + np.array(bg)
        rate = B0*(bg/bg.max())
        return rate
       
    def shot_noise(self,rate):
        """Universal for all types of detectors"""
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,offset=100.0,var=5.0):
        """Used primarily for sCMOS cameras"""
        noise = np.random.normal(offset,np.sqrt(var),size=(self.nx,self.ny))
        return noise
        
    def spikes(self,theta,upsample=4):
        new_nx = self.nx * upsample
        new_ny = self.ny * upsample
        theta = theta[:2, :, np.newaxis, np.newaxis]
        x_vals = np.linspace(0, new_nx, new_nx, endpoint=False)
        y_vals = np.linspace(0, new_ny, new_ny, endpoint=False)

        x_indices = np.floor(theta[0] * upsample).astype(int)
        y_indices = np.floor(theta[1] * upsample).astype(int)

        x_indices = np.clip(x_indices, 0, new_nx - 1)
        y_indices = np.clip(y_indices, 0, new_ny - 1)

        spikes = np.zeros((new_nx, new_ny), dtype=int)
        np.add.at(spikes, (x_indices, y_indices), 1)

        return spikes
        

