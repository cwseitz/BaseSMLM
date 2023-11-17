import numpy as np
from .psf2d import *
from .ill2d_auto import *

def hessiso_auto2d(theta,adu,cmos_params):
    eta,texp,gain,offset,var = cmos_params
    ntheta = len(theta)
    theta = theta.reshape((ntheta,))
    ill = isologlike_auto2d(adu,eta,texp,gain,var)
    hessian_ = hessian(ill)
    hess = hessian_(theta)
    return hess

