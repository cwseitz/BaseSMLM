import numpy as np
from scipy.special import erf

def jac2mix(adu,X,Y,theta,eta,texp,gain,var):
    ntheta,nspots = theta.shape
    nlam = np.zeros_like(adu,dtype=np.float32)
    for n in range(nspots):
        x0,y0,sigma,N0 = theta[:,n]
        alpha = np.sqrt(2)*sigma
        lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
        nlam += N0*lambdx*lambdy
    mu = gain*eta*texp*nlam + var
    jac = 1 - adu/mu
    return jac.flatten()


