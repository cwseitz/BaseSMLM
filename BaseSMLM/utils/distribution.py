import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, multivariate_normal, uniform

class Distribution:
    def __init__(self):
        pass

class PoissonNormal(Distribution):
    def __init__(self,mu_norm,mu_psn,sigma_norm):
        self.p = poisson(mu_psn)
        self.g = norm(loc=mu_norm, scale=sigma_norm)
        self.mu_norm=mu_norm
        self.mu_psn=mu_psn
        self.sigma_norm=sigma_norm
    def sample(self,nsamples):
        samples = np.zeros((nsamples,))
        for n in range(nsamples):
            x = self.p.rvs(size=1) #photoelectrons
            y = self.g.rvs(size=1) #readout noise
            samples[n] = x + y
        return samples
    def get_pmf(self,x):
        fp = self.p.pmf(x)
        n = x.shape[0]
        mat = np.zeros((n,n))
        #convolution of poisson pmf with gaussian
        for i, fp_i in enumerate(fp):
            mat[i] = self.g.pdf(x-x[i])*fp_i
        fpg = np.sum(mat,axis=0)
        fpg /= np.sum(fpg)
        return fpg

class PoissonNormalApproximate(Distribution):
    def __init__(self,mu_norm,mu_psn,sigma_norm):
        self.p = poisson(mu_psn)
        self.g = norm(loc=mu_norm, scale=sigma_norm)
        self.mu_norm=mu_norm
        self.mu_psn=mu_psn
        self.sigma_norm=sigma_norm
    def get_pmf(self,x):
        papprox = poisson(self.mu_psn+self.sigma_norm**2)
        pmf = papprox.pmf(x)
        pmf = pmf/np.sum(pmf)
        return pmf

class Normal(Distribution):
    def __init__(self,mu,cov):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.f = multivariate_normal(mean=self.mu,cov=self.cov)
    def eval(self,x):
        vals = self.f.pdf(x)
        return vals
    def sample(self,nsamples):
        return self.f.rvs(size=nsamples)
    def test(self):
        x = np.linspace(-5,5,100)
        y = self.eval(x)
        s = self.sample(1000)
        plt.hist(s,color='blue',alpha=0.5,bins=10,density=True)
        plt.plot(x,y,color='black')
        plt.show()

class Poisson(Distribution):
    def __init__(self,mu):
        super().__init__()
        self.mu = mu
        self.f = poisson(mu)
    def eval(self,x):
        vals = self.f.pmf(x)
        return vals
    def sample(self,nsamples):
        return self.f.rvs(size=nsamples)
    def test(self):
        x = np.arange(0,100,1)
        y = self.eval(x)
        s = self.sample(1000)
        plt.hist(s,color='blue',alpha=0.5,bins=10,density=True)
        plt.plot(x,y,color='black')
        plt.show()

class Uniform(Distribution):
    """Must be defined on the same interval [xmin,xmax] along each dimension"""
    def __init__(self,N,xmin=0,xmax=1):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.N = N
    def eval(self,x):
        return 1/((self.xmax-self.xmin)**self.N)
    def sample(self,nsamples):
        """One obtains the uniform distribution on [loc, loc + scale]"""
        loc = self.xmin
        scale = self.xmax-self.xmin
        self.f = uniform(loc=loc,scale=scale)
        return self.f.rvs(size=nsamples)
