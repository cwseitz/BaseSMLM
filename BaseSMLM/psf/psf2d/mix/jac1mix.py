import numpy as np
from ..jac2d import jaciso2d

def jac1mix(x,y,theta,eta,texp,gain,var):
    ntheta,nspots = theta.shape
    jacblock = [jaciso2d(x,y,*theta[:,n],eta,texp,gain,var) for n in range(nspots)]
    return np.concatenate(jacblock)
