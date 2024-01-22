import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from BaseSMLM.psf.psf2d.psf2d import *
from scipy.stats import beta, bernoulli, poisson
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

class GaussianRing(Density):
    """A ring of Gaussians with a random phase"""
    def __init__(self,radius,sigma_ring=3.0):
        super().__init__()
        self.radius=radius
        self.sigma_ring=sigma_ring
    def sample(self,n):
        thetas = np.arange(0,n,1)*2*np.pi/n
        phase = np.random.uniform(0,2*np.pi)
        x = self.radius * np.cos(thetas+phase)
        y = self.radius * np.sin(thetas+phase)
        xnoise = np.random.normal(size=x.shape,scale=self.sigma_ring)
        ynoise = np.random.normal(size=y.shape,scale=self.sigma_ring)
        x = x + xnoise
        y = y + ynoise
        return x,y

class Generator:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
    def sample_frames(self,theta,nframes,texp,eta,N0,B0,gain,offset,var):
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,texp=texp,eta=eta,N0=N0)
            S = self.shot_noise(muS)
            if B0 is not None:
                muB = self._mu_b(B0)
                B = self.shot_noise(muB)
            else:
                B = 0
            adu = gain*(S+B) + self.read_noise(offset=offset,var=var)
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta)
            _adu.append(adu); _spikes.append(spikes)
            if show:
                self.show(adu,muS,muB,theta)
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes
    def _mu_s(self,theta,texp=1.0,eta=1.0,N0=1.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y,indexing='ij')
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
        rate = B0*np.ones((self.nx,self.ny))
        return rate
       
    def shot_noise(self,rate):
        """Universal for all types of detectors"""
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,offset=100.0,var=5.0):
        """Gaussian readout noise"""
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

    def sample_states(self, nspots, nframes, p=None, N00=20.0, N01=300.0, a=2, b=2):
        if p is None:
            p = beta.rvs(a, b, size=(nspots,))
        else:
            p = p*np.ones((nspots,))
        z = np.array([bernoulli.rvs(pi, size=(nframes,)) for pi in p])
        x = N01*z + N00*(1-z)
        return x
        
    def sample_frames(self,theta,nframes,states,texp,eta,B0,gain,offset,var):
        _adu = []; _spikes = []
        for n in range(nframes):
            print(f'Simulating frame {n}')
            muS = self._mu_s(theta,states[:,n],texp=texp,eta=eta)
            S = self.shot_noise(muS)
            if B0 is not None:
                muB = self._mu_b(B0)
                B = self.shot_noise(muB)
            else:
                B = np.zeros_like(muS)
            adu = gain*(S+B) + self.read_noise(offset=offset,var=var)
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta,78)
            _adu.append(adu); _spikes.append(spikes)
                            
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes

    def _mu_s(self,theta,states,texp=1.0,eta=1.0,B0=0.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        mu = np.zeros((self.nx,self.ny),dtype=np.float32)
        ntheta,nspots = theta.shape
        for n in range(nspots):
            x0,y0,sigma = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            this_mu = eta*texp*states[n]*lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += this_mu
        return mu

    def _mu_b(self,B0):
        rate = B0*np.ones((self.nx,self.ny))
        return rate
       
    def shot_noise(self,rate):
        """Universal for all types of detectors"""
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,offset=100.0,var=5.0):
        """Gaussian readout noise"""
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
        

    
        

