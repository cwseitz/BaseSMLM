import numpy as np
import matplotlib.pyplot as plt
from ..psf.psf2d.psf2d import *

class KDE:
    def __init__(self,spots):
        self.spots = spots
    def forward(self,margin=100,pixel_size=108.3,kde_pixel_size=10.83,sigma=1.0,nxny=None,x_range=None,y_range=None):
        patchw = int(round(3*sigma))
        a = int(pixel_size/kde_pixel_size)
        theta = a*self.spots[['x_mle','y_mle']].values
                
        if x_range is None:
            if nxny is None:
                theta[:,0] -= theta[:,0].min() #shift to zero
                theta[:,1] -= theta[:,1].min()
                x_range = theta[:,0].max() #total range
                y_range = theta[:,1].max()
                nx = int(np.ceil(x_range)); ny = int(np.ceil(y_range))
            else:
                nx = int(np.ceil(a*nxny[0]))
                ny = int(np.ceil(a*nxny[1]))
        else:
            nx = a*len(x_range); ny = a*len(y_range)
        
        kde = np.zeros((2*margin+nx,2*margin+ny),dtype=np.float32)
        
        ns,nd = theta.shape
        x = np.arange(0,2*patchw); y = np.arange(0,2*patchw)
        X,Y = np.meshgrid(x,y,indexing='ij')

        for n in range(ns):
            x0,y0 = theta[n,:]
            patchx, patchy = int(round(x0))-patchw, int(round(y0))-patchw #upper left corner of patch
            x0p = x0-patchx; y0p = y0-patchy #coordinates within patch
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            kde_xmin = patchx+margin; kde_xmax = patchx+2*patchw+margin
            kde_ymin = patchy+margin; kde_ymax = patchy+2*patchw+margin
            kde[kde_xmin:kde_xmax,kde_ymin:kde_ymax] += lam
        return kde

