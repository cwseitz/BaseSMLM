from .distribution import PoissonNormal, PoissonNormalApproximate
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import json

def get_kdist(offset,rate,std,gain=2.2,w=1000,ax=None):
    pnorm = PoissonNormal(offset,rate,std)
    pnorm_approx = PoissonNormalApproximate(offset,rate,std)
    x = np.arange(0,w,1)
    pmf = pnorm.get_pmf(x)
    pmf_approx = pnorm_approx.get_pmf(x-offset)
    cmf = np.cumsum(pmf)
    cmf_approx = np.cumsum(pmf_approx)
    if ax:
        ax.plot(cmf,color='red',label='CMF')
        ax.plot(cmf_approx,color='gray',label='Poisson CMF')
        ax.plot(np.abs(cmf-cmf_approx),color='cornflowerblue',label=r'$\Delta$CMF')
        ax.set_xlim([offset+rate-50,offset+rate+50])
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    kdist = np.max(np.abs(cmf-cmf_approx))
    return kdist
