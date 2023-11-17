import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class Sampler:
    def __init__(self):
        pass

class Metropolis2D(Sampler):
    def __init__(self,mu,cov,negloglike):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.prop = multivariate_normal(mu,cov)
        self.negloglike = negloglike
        
    def summarize(self,like_new,like_old,theta_new,theta_old,ratio,accept,n):
        print(f'MCMC iteration: {n}')
        print(f'Like new {like_new}')
        print(f'Like old {like_old}')
        print(f'Theta new {theta_new}')
        print(f'Theta old {theta_old}')
        print(f'Acceptance ratio {ratio}')
        print(f'Accepted {accept}\n')

    def thin(self,thetas,skip=5,tburn=500):
        thetas = thetas[:,tburn:]
        thetas = thetas[:,::5]
        return thetas

    def diagnostic(self,thetas,acc,tburn=500):
        acc = np.array(acc)
        acc = acc.astype(np.int)
        f = np.cumsum(acc)
        f = f/np.arange(1,len(f)+1,1)
        fig, ax = plt.subplots()
        ax.plot(f,color='black')
        plt.tight_layout()
        fig, ax = plt.subplots(1,4,figsize=(8,2))
        k1t = thetas[0,tburn:]
        k2t = thetas[1,tburn:]
        k3t = thetas[2,tburn:]
        k4t = thetas[3,tburn:]
        ax[0].plot(k1t[::5],color='black')
        ax[1].plot(k2t[::5],color='black')
        ax[2].plot(k3t[::5],color='black')
        ax[3].plot(k4t[::5],color='black')
        self.post_marginals(thetas,tburn=tburn)
        plt.tight_layout()
        plt.show()
        
    def sample(self,theta_old,data,like_old,beta,n):
        accept = True
        dtheta = self.prop.rvs()
        theta_new = theta_old + dtheta
        
        #for 2d, none of the parameters should be negative
        if np.any(theta_new < 0):
            accept = False
            #self.summarize(None,like_old,theta_new,theta_old,None,accept,n)
            return theta_old, like_old, accept

        like_new = self.negloglike(theta_new,data)
        a = np.exp(beta*(like_old-like_new))        
        u = np.random.uniform(0,1)
        if u <= a:
            theta = theta_new
            like = like_new
        else:
            accept = False
            theta = theta_old
            like = like_old
            
        #self.summarize(like_new,like_old,theta_new,theta_old,a,accept,n)
        return theta, like, accept
        
    def post_marginals(self,thetas,tburn=500,bins=10):
        ntheta,nsamples = thetas.shape
        fig, ax = plt.subplots(1,ntheta,figsize=(2*ntheta,2))
        for n in range(ntheta):
            ax[n].hist(thetas[n,tburn:],bins=bins,color='black',density=True)
            ax[n].set_title(np.round(np.std(thetas[n,tburn:]),2))
        plt.tight_layout()

    def run(self,data,theta0,iters=1000,tburn=500,skip=5,beta=1,thin=False,diag=False):
        theta = theta0
        thetas = np.zeros((len(theta),iters))
        like = self.negloglike(theta0,data)
        acc = []
        for n in range(iters):
            theta, like, accept = self.sample(theta,data,like,beta,n)
            acc.append(accept)
            thetas[:,n] = theta
        if thin:
            thetas = self.thin(thetas,skip=skip,tburn=tburn)
        if diag:
            self.diagnostic(thetas,acc,tburn=tburn)
        return thetas
        
class Metropolis3D(Sampler):
    def __init__(self,mu,cov,negloglike):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.prop = multivariate_normal(mu,cov)
        self.negloglike = negloglike
        
    def summarize(self,like_new,like_old,theta_new,theta_old,ratio,accept,n):
        print(f'MCMC iteration: {n}')
        print(f'Like new {like_new}')
        print(f'Like old {like_old}')
        print(f'Theta new {theta_new}')
        print(f'Theta old {theta_old}')
        print(f'Acceptance ratio {ratio}')
        print(f'Accepted {accept}\n')

    def thin(self,thetas,skip=5,tburn=500):
        thetas = thetas[:,tburn:]
        thetas = thetas[:,::5]
        return thetas

    def diagnostic(self,thetas,acc,theta0=None,tburn=500):
        acc = np.array(acc)
        acc = acc.astype(np.int)
        f = np.cumsum(acc)
        f = f/np.arange(1,len(f)+1,1)
        fig, ax = plt.subplots(figsize=(3,2))
        ax.plot(f,color='black')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Acceptance Rate')
        param_names = ['x (nm)','y (nm)','z (nm)',r'$N_{0}$ (cps)']
        plt.tight_layout()

        self.post_marginals(thetas,theta0=theta0,tburn=tburn)
        plt.tight_layout()

        
    def sample(self,theta_old,data,like_old,beta,n):
        accept = True
        dtheta = self.prop.rvs()
        theta_new = theta_old + dtheta   
             
        #for 3d, only z can be negative
        like_new = self.negloglike(theta_new,data)
        a = np.exp(beta*(like_old-like_new))        
        u = np.random.uniform(0,1)
        if u <= a:
            theta = theta_new
            like = like_new
        else:
            accept = False
            theta = theta_old
            like = like_old
            
        #self.summarize(like_new,like_old,theta_new,theta_old,a,accept,n)
        return theta, like, accept
        
    def post_marginals(self,thetas,theta0=None,tburn=500,bins=20):

        ntheta,nsamples = thetas.shape
        fig, ax = plt.subplots(3,2,figsize=(7,5))
        param_names = [r'$x$ (nm)',r'$y$ (nm)',r'$z$ (nm)',r'$N_0$ (cps)']
        colors = ['white','white','white','white']

        thetas[0,:] = 108.3*thetas[0,:]
        thetas[1,:] = 108.3*thetas[1,:]
        thetas[2,:] = 1000*thetas[2,:]
        
        theta0 = theta0 * np.array([108.3,108.3,1000,1])
        for n in range(3):
            if theta0 is not None:
                ax[n,0].hlines(theta0[n],0,nsamples,color='blue',linestyle='--',label='MLE')
            ax[n,0].plot(thetas[n,tburn:],color='black',alpha=0.5)
            ax[n,0].set_xlabel('Iteration')
            ax[n,0].set_ylabel(param_names[n])
            ax[n,1].hist(thetas[n,tburn:],bins=bins,color=colors[n],edgecolor='black',density=False,orientation='horizontal')
            ax[n,1].set_title(r'$\sigma=$' + str(np.round(np.std(thetas[n,tburn:]),2)) + 'nm')
            ax[n,1].set_ylabel(param_names[n])
            ax[n,1].set_xlabel('Frequency')
            ax[n,0].legend()
        plt.tight_layout()
        
        
    def run(self,data,theta0,iters=1000,tburn=500,skip=5,beta=1,thin=False,diag=False):
        theta = theta0
        thetas = np.zeros((len(theta),iters))
        like = self.negloglike(theta0,data)
        acc = []
        for n in range(iters):
            theta, like, accept = self.sample(theta,data,like,beta,n)
            acc.append(accept)
            thetas[:,n] = theta
        if thin:
            thetas = self.thin(thetas,skip=skip,tburn=tburn)
        if diag:
            self.diagnostic(thetas,acc,theta0=theta0,tburn=tburn)
        return thetas
