import numpy as np
from ..psf2d import jac1

def jac1mix(x,y,theta,cmos_params):
    nspots = len(theta) // 4
    ntheta,nspots = theta.shape
    jacblock = [jac1(x,y,theta[:,n],cmos_params) for n in range(nspots)]
    return np.concatenate(jacblock)
