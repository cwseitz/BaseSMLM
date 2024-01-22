import numpy as np
import matplotlib.pyplot as plt
from .base import *

class Ring2D(Generator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,
                B0=None,nframes=1,offset=100.0,var=5.0,show=False):
        density = Ring(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        adu,spikes = self.sample_frames(theta,nframes,texp,eta,N0,B0,gain,offset,var)
        return adu,spikes,theta
        
class Disc2D(Generator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,
                B0=None,nframes=1,offset=100.0,var=5.0,show=False):
        density = Disc(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        adu,spikes = self.sample_frames(theta,nframes,texp,eta,N0,B0,gain,offset,var)
        return adu,spikes,theta
              
class Ring2D_TwoState(TwoStateGenerator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N00=20.0,N01=300.0,eta=1.0,
                gain=1.0,B0=None,nframes=100,offset=100.0,var=5.0,p=None,show=False):
        density = Ring(radius)
        theta = np.zeros((3,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0
        theta[1,:] = y + y0
        theta[2,:] = sigma
        states = self.sample_states(nspots,nframes,p=p,N00=N00,N01=N01)
        adu,spikes = self.sample_frames(theta,nframes,states,texp,eta,B0,gain,offset,var)
        return adu,spikes,theta

class Disc2D_TwoState(TwoStateGenerator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N00=20.0,N01=300.0,eta=1.0,
                gain=1.0,B0=None,nframes=100,offset=100.0,var=5.0,p=None,show=False):
        density = Disc(radius)
        theta = np.zeros((3,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0
        theta[1,:] = y + y0
        theta[2,:] = sigma
        states = self.sample_states(nspots,nframes,p=p,N00=N00,N01=N01)
        adu,spikes = self.sample_frames(theta,nframes,states,texp,eta,B0,gain,offset,var)
        return adu,spikes,theta
        
class GaussianRing2D_TwoState(TwoStateGenerator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N00=20.0,N01=300.0,eta=1.0,
                gain=1.0,B0=None,nframes=100,offset=100.0,var=5.0,p=None,
                sigma_ring=2.0,show=False):
        density = GaussianRing(radius,sigma_ring)
        theta = np.zeros((3,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0
        theta[1,:] = y + y0
        theta[2,:] = sigma
        states = self.sample_states(nspots,nframes,p=p,N00=N00,N01=N01)
        adu,spikes = self.sample_frames(theta,nframes,states,texp,eta,B0,gain,offset,var)
        return adu,spikes,theta
        
              
        
