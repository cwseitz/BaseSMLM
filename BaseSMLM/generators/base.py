import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from BaseSMLM.psf.psf2d.psf2d import *
from hmmlearn import hmm

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

class Brownian(Density):
    """Brownian motion or a Gaussian chain"""
    def __init__(self,sigma):
        super().__init__()
        self.sigma=sigma
    def sample(self,n):
        chain = np.zeros((n,2))
        chain[1:,:] = np.random.normal(0,self.sigma,size=(n-1,2))
        chain = np.cumsum(chain, axis=0)
        return chain[:,1], chain[:,0]

class Generator:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
    def _mu_s(self,theta,texp=1.0,eta=1.0,N0=1.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        mu = np.zeros((self.nx,self.ny),dtype=np.float32)
        ntheta,nspots = theta.shape
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
        
class TwoStateGenerator:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
        
    def simulate(self, nspots, nframes, texp=1.0, k_on=0.5, k_off=0.5):

        model = hmm.MultinomialHMM(n_components=2,n_trials=1)
        model.startprob_ = np.array([1.0, 0.0])  # Start in state 'off'
        model.transmat_ = np.array([[1 - k_on*texp, k_on*texp], [k_off*texp, 1 - k_off*texp]])
        model.emissionprob_ = np.array([[1, 0], [0, 1]])  # Emissions are the hidden states
        state_matrix = []
        for n in range(nspots):
            _, states = model.sample(n_samples=nframes)
            state_matrix.append(states)
        state_matrix = np.array(state_matrix)
        return state_matrix


    def _mu_s(self,theta,state,texp=1.0,eta=1.0,N0=1.0,B0=0.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        mu = np.zeros((self.nx,self.ny),dtype=np.float32)
        ntheta,nspots = theta.shape
        i0 = eta*N0*texp
        for n in range(nspots):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            this_mu = i0*state[n]*lamx(X,y0p,sigma)*lamy(Y,x0p,sigma)
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
        
    def spikes(self,theta,size):
        new_nx = size
        new_ny = size
        theta = theta[:2, :, np.newaxis, np.newaxis]
        x_vals = np.linspace(0, new_nx, new_nx, endpoint=False)
        y_vals = np.linspace(0, new_ny, new_ny, endpoint=False)

        x_indices = np.floor(theta[0] * size/self.nx).astype(int)
        y_indices = np.floor(theta[1] * size/self.ny).astype(int)

        x_indices = np.clip(x_indices, 0, new_nx - 1)
        y_indices = np.clip(y_indices, 0, new_ny - 1)

        spikes = np.zeros((new_nx, new_ny), dtype=int)
        np.add.at(spikes, (x_indices, y_indices), 1)

        return spikes
        
    
        

