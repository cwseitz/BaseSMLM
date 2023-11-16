import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

def isologlike_auto2d(adu,eta,texp,gain,var):
    def isologlike(theta,adu=adu,gain=gain,var=var):
        lx, ly = adu.shape
        x0,y0,sigma,N0 = theta
        alpha = np.sqrt(2)*sigma
        X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
        lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
        lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
        lam = lamdx*lamdy
        i0 = gain*eta*texp*N0
        muprm = i0*lam + var
        stirling = adu*np.log(adu+1e-8) - adu
        nll = stirling + muprm - adu*np.log(muprm)
        nll = np.sum(nll)
        return nll
    return isologlike
