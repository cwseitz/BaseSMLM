import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.scipy.special import erf

def mixloglike_auto(adu,eta,texp,gain,var,nspots):
    def mixloglike(theta,adu=adu,gain=gain,var=var):
        lx, ly = adu.shape
        X,Y = np.meshgrid(np.arange(0,lx),np.arange(0,ly))
        mu = np.zeros_like(adu)
        for n in range(nspots):
            x0,y0,sigma,N0 = theta[4*n:4*(n+1)]
            alpha = np.sqrt(2)*sigma
            lamdx = 0.5*(erf((X+0.5-x0)/alpha) - erf((X-0.5-x0)/alpha))
            lamdy = 0.5*(erf((Y+0.5-y0)/alpha) - erf((Y-0.5-y0)/alpha))
            lam = lamdx*lamdy
            mu += eta*texp*N0*lam
        stirling = adu*np.log(adu+1e-8) - adu
        nll = stirling + gain*mu + var - adu*np.log(gain*mu + var)
        nll = np.sum(nll)
        return nll
    return mixloglike

