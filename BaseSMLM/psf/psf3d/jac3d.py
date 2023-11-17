import numpy as np
from .psf3d import *
from .ill3d_auto import *

def jaciso3d(theta,adu,cmos_params):
    lx, ly = adu.shape
    ntheta = len(theta)
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jac1(X,Y,theta,cmos_params)
    J1 = J1.reshape((ntheta,lx**2))
    J2 = jac2(adu,X,Y,theta,cmos_params)
    J = J1 @ J2
    return J
    
def jaciso_auto3d(theta,adu,cmos_params):
    nx,ny,eta,texp,gain,offset,var = cmos_params
    ntheta = len(theta)
    theta = theta.reshape((ntheta,))
    ill = isologlike_auto3d(adu,eta,texp,gain,var)
    jacobian_ = jacobian(ill)
    jac = jacobian_(theta)
    return jac
