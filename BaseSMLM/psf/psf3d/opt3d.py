import numpy as np
import matplotlib.pyplot as plt
from ...localize import Metropolis3D
from .psf3d import *
from .jac3d import *
from .ill3d import *
from .hess3d import *
from numpy.linalg import inv

class IsoLogLikelihood:
    def __init__(self,func,cmos_params):
        self.func = func
        self.cmos_params = cmos_params
    def __call__(self,theta,adu):
        return self.func(theta,adu,self.cmos_params)

class MLE3D_MCMC:
   def __init__(self,theta0,adu,config,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.config = config
       self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
       self.theta_gt = theta_gt
   def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,3,figsize=(8,2))
       ax[0].plot(thetat[:,0])
       ax[0].set_xlabel('Iteration')
       ax[0].set_ylabel('x (px)')
       if self.theta_gt:
           ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
       ax[1].plot(thetat[:,1])
       ax[1].set_xlabel('Iteration')
       ax[1].set_ylabel('y (px)')
       if self.theta_gt:
           ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
       ax[2].plot(thetat[:,2])
       ax[2].set_xlabel('Iteration')
       ax[2].set_ylabel(r'z (nm)')
       if self.theta_gt:
           ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()

   def metropolis(self,theta0,iters=3000,tburn=300):
       mean = np.zeros((4,)); cov = 0.01*np.eye(4)
       loglike = IsoLogLikelihood(isologlike3d,self.cmos_params)
       sampler = Metropolis3D(mean,cov,loglike)
       samples = sampler.run(self.adu,theta0,iters=iters,beta=5.0,tburn=tburn,diag=True)
       return samples

   def optimize(self,iters=1000,lr=None,plot=False):
       if plot:
           thetat = np.zeros((iters,4))
       if lr is None:
           lr = np.array([0.001,0.001,0.001,0])
       loglike = np.zeros((iters,))
       theta_mle = np.zeros_like(self.theta0)
       theta_mle += self.theta0
       for n in range(iters):
           loglike[n] = isologlike3d(theta_mle,self.adu,self.cmos_params)
           jac = jaciso3d(theta_mle,self.adu,self.cmos_params)
           theta_mle = theta_mle - lr*jac
           if plot:
               thetat[n,:] = theta_mle
       if plot:
           self.plot(thetat,iters)
           
       samples = self.metropolis(theta_mle) #seed MH at the MLE
       return theta_mle, loglike, samples

class MLE3D:
   def __init__(self,theta0,adu,config,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.config = config
       self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
       self.theta_gt = theta_gt
   def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,3,figsize=(8,2))
       ax[0].plot(thetat[:,0])
       ax[0].set_xlabel('Iteration')
       ax[0].set_ylabel('x (px)')
       if self.theta_gt is not None:
           ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
       ax[1].plot(thetat[:,1])
       ax[1].set_xlabel('Iteration')
       ax[1].set_ylabel('y (px)')
       if self.theta_gt is not None:
           ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
       ax[2].plot(thetat[:,2])
       ax[2].set_xlabel('Iteration')
       ax[2].set_ylabel(r'z (um)')
       if self.theta_gt is not None:
           ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()
   def optimize(self,max_iters=1000,lr=None,plot=False,tol=1e-6):
       if plot:
           thetat = []
       if lr is None:
           lr = np.array([0.001,0.001,0.001,0])
       loglike = []
       theta = np.zeros_like(self.theta0)
       theta += self.theta0
       niters = 0
       while niters < max_iters:
           niters += 1
           loglike.append(isologlike3d(theta,self.adu,self.cmos_params))
           jac = jaciso3d(theta,self.adu,self.cmos_params)
           theta = theta - lr*jac
           if plot:
               thetat.append(theta)
           dd = lr[:-1]*jac[:-1]
           if np.all(np.abs(dd) < tol):
               break

       if plot:
           self.plot(np.array(thetat),niters)
           
       return theta, loglike
  
class SGLDSampler3D:
    def __init__(self,theta0,adu,config,theta_gt=None):
       self.theta0 = theta0
       self.adu = adu
       self.config = config
       self.cmos_params = [config['nx'],config['ny'],
                           config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
       self.dfcs_params = [config['zmin'],config['alpha'],config['beta']]
       self.theta_gt = theta_gt
    def plot(self,thetat,iters):
       fig,ax = plt.subplots(1,3,figsize=(8,2))
       ax[0].plot(thetat[:,0])
       ax[0].set_xlabel('Iteration')
       ax[0].set_ylabel('x')
       ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
       ax[1].plot(thetat[:,1])
       ax[1].set_xlabel('Iteration')
       ax[1].set_ylabel('y')
       ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
       ax[2].plot(thetat[:,2])
       ax[2].set_xlabel('Iteration')
       ax[2].set_ylabel('z')
       ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
       plt.tight_layout()
       plt.show()
    def sample(self,iters=1000,lr=None,tburn=0,plot=False):
        ntheta = len(self.theta0)
        thetat = np.zeros((iters,ntheta))
        thetat[0,:] = self.theta0
        if lr is None:
            lr = np.array([0.0001,0.0001,2.0,0,0])
        t = np.arange(0,iters,1)
        for n in range(1,iters):
            jac = jaciso3d(thetat[n-1,:],self.adu,self.cmos_params,self.dfcs_params)
            epsx = np.random.normal(0,1)
            epsy = np.random.normal(0,1)
            epsz = np.random.normal(0,1)
            thetat[n,0] = thetat[n-1,0] - lr[0]*jac[0] + np.sqrt(lr[0])*epsx
            thetat[n,1] = thetat[n-1,1] - lr[1]*jac[1] + np.sqrt(lr[1])*epsy
            thetat[n,2] = thetat[n-1,2] - lr[2]*jac[2] + np.sqrt(lr[2])*epsz
            thetat[n,3] = thetat[n-1,3]
            thetat[n,4] = thetat[n-1,4]
        thetat = thetat[tburn:,:]
        if plot:
            self.plot(thetat,iters-tburn)
        return thetat
       
