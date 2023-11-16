import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from ...localize import Metropolis2D
from .psf2d import *
from .ill2d import *
from .jac2d import *
from .hess2d import *

class IsoLogLikelihood:
    def __init__(self,func,cmos_params):
        self.func = func
        self.cmos_params = cmos_params
    def __call__(self,theta,adu):
        return self.func(theta,adu,self.cmos_params)

class MLE2D_MCMC:
    def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
                            
    def show(self,theta0,theta):
        fig,ax = plt.subplots(figsize=(4,4))
        ax.imshow(self.adu,cmap='gray')
        ax.scatter(theta0[0],theta0[1],color='red',label='raw')
        ax.scatter(theta[0],theta[1],color='blue',label='fit')
        ax.legend()
        plt.tight_layout()
        
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        if self.theta_gt is not None:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        if self.theta_gt is not None:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        if self.theta_gt is not None:
            ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].plot(thetat[:,3])
        if self.theta_gt is not None:
            ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
        
    def metropolis(self,theta0,iters=3000,tburn=300,prop_cov=0.05,beta=0.02,plot_mcmc=False):
        mean = np.zeros((4,)); cov = prop_cov*np.eye(4)
        loglike = IsoLogLikelihood(isologlike2d,self.cmos_params)
        sampler = Metropolis2D(mean,cov,loglike)
        samples = sampler.run(self.adu,theta0,iters=iters,beta=beta,tburn=tburn,diag=plot_mcmc)
        return samples
        
    def optimize(self,max_iters=1000,mcmc_iters=1000,tburn=None,lr=None,prop_cov=0.05,beta=0.02,plot_fit=False,plot_mcmc=False,tol=1e-8):
        if plot_fit:
           thetat = []
        if lr is None:
           lr = np.array([0.001,0.001,0,0])
        loglike = []
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        niters = 0
        while niters < max_iters:
            niters += 1
            loglike.append(isologlike2d(theta,self.adu,self.cmos_params))
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            theta = theta - lr*jac
            if plot_fit:
                thetat.append(theta)
            dd = lr[:-1]*jac[:-1]
            if np.all(np.abs(dd) < tol):
                break
        if plot_fit:
            self.plot(np.array(thetat),niters)
            plt.show()

        samples = self.metropolis(theta,iters=mcmc_iters,tburn=tburn,prop_cov=prop_cov,beta=beta,plot_mcmc=plot_mcmc) #seed MH at the MLE
        return theta, loglike, samples

class MLE2D:
    def __init__(self,theta0,adu,setup_params,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.setup_params = setup_params
       self.cmos_params = [setup_params['eta'],setup_params['texp'],
                            np.load(setup_params['gain'])['arr_0'],
                            np.load(setup_params['offset'])['arr_0'],
                            np.load(setup_params['var'])['arr_0']]
                            
    def show(self,theta0,theta):
        fig,ax = plt.subplots(figsize=(4,4))
        ax.imshow(self.adu,cmap='gray')
        ax.scatter(theta0[0],theta0[1],color='red',label='raw')
        ax.scatter(theta[0],theta[1],color='blue',label='fit')
        ax.legend()
        plt.tight_layout()
        
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        if self.theta_gt is not None:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        if self.theta_gt is not None:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        if self.theta_gt is not None:
            ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].plot(thetat[:,3])
        if self.theta_gt is not None:
            ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()

    def get_errors(self,theta):
        """This doesn't seem to work correctly"""
        hess = hessiso_auto2d(theta,self.adu,self.cmos_params)
        try:
            errors = np.sqrt(np.diag(inv(hess)))
        except:
            errors = np.empty((4,))
            errors[:] = np.nan
        
        return errors
                
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
            loglike.append(isologlike2d(theta,self.adu,self.cmos_params))
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            theta = theta - lr*jac
            if plot_fit:
                thetat.append(theta)
            dd = lr[:-1]*jac[:-1]
            if np.all(np.abs(dd) < tol):
                converged = True
                break
        if plot_fit:
            self.plot(np.array(thetat),niters)
            plt.show()


        return theta, loglike, converged

        

