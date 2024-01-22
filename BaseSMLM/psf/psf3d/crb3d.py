import numpy as np
from numpy.linalg import inv
from .psf3d import *

def crlb3d(theta,cmos_params):
    """This is identical to 2D CRLB"""
    ntheta = len(theta)
    x0,y0,z0,sigma,N0 = theta
    eta,texp,gain,offset,var = cmos_params
    x = np.arange(0,nx); y = np.arange(0,ny)
    X,Y = np.meshgrid(x,y,indexing='ij')
    sigma_x = sx(z0); sigma_y = sy(z0)
    lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    J = jac1(X,Y,theta,cmos_params)
    I = np.zeros((ntheta,ntheta))
    for n in range(ntheta):
       for m in range(ntheta):
           I[n,m] = np.sum(J[n]*J[m]/muprm)
    return np.sqrt(np.diagonal(inv(I)))



