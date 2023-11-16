import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from scipy.special import erf
from .psf3d import *


def isologlike3d(theta,adu,cmos_params):
    x0,y0,z0,N0 = theta
    eta,texp,gain,offset,var = cmos_params
    sigma_x = sx(z0); sigma_y = sy(z0)
    nx, ny = offset.shape
    x = np.arange(0,nx); y = np.arange(0,ny)
    X,Y = np.meshgrid(x,y)
    lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stirling = adu * np.nan_to_num(np.log(adu)) - adu
    p = adu*np.log(muprm)
    warnings.filterwarnings("default", category=RuntimeWarning)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
