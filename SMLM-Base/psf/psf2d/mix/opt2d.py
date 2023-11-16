import numpy as np
import matplotlib.pyplot as plt
from ..psf2d import *
from .mll2d import *
from .mll2d_auto import *
from .jac2d import *


class IsoLogLikelihood:
    def __init__(self,func,cmos_params):
        self.func = func
        self.cmos_params = cmos_params
    def __call__(self,theta,adu):
        return self.func(theta,adu,self.cmos_params)

class MLE2DMix:
    def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
                                         
    def optimize(self,max_iters=1000,lr=None,plot_fit=False,tol=1e-8):
        if plot_fit:
           thetat = []
        if lr is None:
           lr = np.array([0.001,0.001,0,0])
        loglike = []
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        niters = 0
        converged = False
        while niters < max_iters:
            niters += 1
            loglike.append(mixloglike_auto(theta,self.adu,self.cmos_params))
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            theta = theta - lr*jac
            if plot_fit:
                thetat.append(theta)
            dd = lr[:-1]*jac[:-1]
            if np.all(np.abs(dd) < tol):
                converged = True
                break

        return theta, loglike, converged

        

