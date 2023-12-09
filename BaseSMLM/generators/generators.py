import numpy as np
import matplotlib.pyplot as plt
from .base import *

class Brownian2D(Generator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,B0=None,nframes=1,offset=100.0,var=5.0):
        density = Brownian(radius); f=True
        while f:
            theta = np.zeros((4,nspots))
            x,y = density.sample(nspots)
            r = np.sqrt(x**2 + y**2)
            f = np.any(r > self.nx/3)

        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,texp=texp,eta=eta,N0=N0)
            S = self.shot_noise(muS)
            if B0 is not None:
                muB = self._mu_b()
                B = self.shot_noise(muB)
            else:
                B = 0
            adu = gain*(S+B) + self.read_noise(offset=offset,var=var)
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta)
            _adu.append(adu); _spikes.append(spikes)
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes,theta

class Ring2D(Generator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,B0=None,nframes=1,offset=100.0,var=5.0):
        density = Ring(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,texp=texp,eta=eta,N0=N0)
            S = self.shot_noise(muS)
            if B0 is not None:
                muB = self._mu_b()
                B = self.shot_noise(muB)
            else:
                B = 0
            adu = gain*(S+B) + self.read_noise(offset=offset,var=var)
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta)
            _adu.append(adu); _spikes.append(spikes)
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes,theta
        
class Disc2D(Generator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,B0=None,nframes=1,offset=100.0,var=5.0):
        density = Disc(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,texp=texp,eta=eta,N0=N0)
            S = self.shot_noise(muS)
            if B0 is not None:
                muB = self._mu_b()
                B = self.shot_noise(muB)
            else:
                B = 0
            adu = gain*(S+B) + self.read_noise(offset=offset,var=var)
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta)
            _adu.append(adu); _spikes.append(spikes)
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes,theta
        
        
class Ring2D_TwoState(TwoStateGenerator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,B0=None,nframes=100,offset=100.0,var=5.0,show=False):
        density = Ring(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        
        self.states = self.simulate(nspots,nframes)
        
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,self.states[:,n],texp=texp,eta=eta,N0=N0)
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
            if show:
                self.show(adu,muS,muB,theta)
                
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes,theta
        
    def show(self,adu,muS,muB,theta):
        fig,ax=plt.subplots(1,3)
        ax[0].imshow(adu,cmap='gray')
        #ax[0].scatter(theta[1,:],theta[0,:],color='red',marker='x')
        ax[1].imshow(muS,cmap='gray')
        ax[2].imshow(muB,cmap='gray')
        #ax[1].scatter(theta[1,:],theta[0,:],color='red',marker='x')
        plt.show()
        
class Disc2D_TwoState(TwoStateGenerator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,B0=None,nframes=100,offset=100.0,var=5.0,show=False):
        density = Disc(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        
        self.states = self.simulate(nspots,nframes)
        
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,self.states[:,n],texp=texp,eta=eta,N0=N0)
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
            if show:
                self.show(adu,muS,muB,theta)
                
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes,theta
        
    def show(self,adu,muS,muB,theta):
        fig,ax=plt.subplots(1,3)
        ax[0].imshow(adu,cmap='gray')
        #ax[0].scatter(theta[1,:],theta[0,:],color='red',marker='x')
        ax[1].imshow(muS,cmap='gray')
        ax[2].imshow(muB,cmap='gray')
        #ax[1].scatter(theta[1,:],theta[0,:],color='red',marker='x')
        plt.show()
        
        
