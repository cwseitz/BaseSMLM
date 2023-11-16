import numpy as np
from .jac1mix import *
from .jac2mix import *
from .mll2d import *
from .mll2d_auto import *

def jacmix(theta,adu,cmos_params):
    lx, ly = adu.shape
    ntheta, nspots = theta.shape
    X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
    J1 = jac1mix(X,Y,theta,*cmos_params)
    J1 = J1.reshape((ntheta*nspots,lx**2))
    J2 = jac2mix(adu,X,Y,theta,*cmos_params)
    J = J1 @ J2
    return J

def jacmix_auto(theta,adu,cmos_params):
    eta,texp,gain,var = cmos_params
    ntheta, nspots = theta.shape
    theta = theta.reshape((ntheta*nspots,))
    ll = mixloglike_auto(adu,eta,texp,gain,var,nspots)
    jacobian_ = jacobian(ll)
    jac = jacobian_(theta)
    return jac
