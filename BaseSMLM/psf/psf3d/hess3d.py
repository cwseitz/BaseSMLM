import numpy as np
from .psf3d import *
from .ill3d_auto import *

def hessiso_auto3d(theta,adu,cmos_params):
    L,eta,texp,gain,var = cmos_params
    ntheta = len(theta)
    theta = theta.reshape((ntheta,))
    ill = isologlike_auto3d(adu,eta,texp,gain,var)
    hessian_ = hessian(ill)
    hess = hessian_(theta)
    return hess

