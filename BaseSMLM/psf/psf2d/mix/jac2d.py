import numpy as np
from .jac1mix import *
from .jac2mix import *
from .mll2d import *

def jacmix(theta,adu,cmos_params):
    lx, ly = adu.shape
    ntheta,nspots = theta.shape
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jac1mix(X,Y,theta,cmos_params)
    J1 = J1.reshape((ntheta*nspots,lx**2))
    J2 = jac2mix(adu,X,Y,theta,cmos_params)
    J = J1 @ J2
    J = J.reshape((ntheta,nspots))
    return J


