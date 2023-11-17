import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import restoration
from skimage import img_as_float, img_as_uint
from scipy.special import erf

class RLDeconvolver:
    def __init__(self):
        pass
    def psf(self,sigma,nx,ny):
        alpha = np.sqrt(2)*sigma
        x0,y0 = 3,3
        x = np.arange(0,nx); y = np.arange(0,ny)
        X,Y = np.meshgrid(x,y)
        lambdx = 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
        lambdy = 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))
        lam = lambdx*lambdy
        return lam
    def deconvolve(self,frame,iters=5,plot=False):
        sigma,nx,ny = 0.92,7,7
        psf = self.psf(sigma,nx,ny)
        fframe = img_as_float(frame)
        deconv = restoration.richardson_lucy(fframe,psf,num_iter=iters)
        deconv = img_as_uint(deconv)
        if plot:
            fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
            ax[0].imshow(frame,cmap='gray')
            ax[1].imshow(deconv,cmap='gray')
            plt.show()
        return deconv
        

