import numpy as np
import matplotlib.pyplot as plt
from ..psf.psf2d.psf2d import *
import frc
from .kde import *

class FRC:
    def __init__(self,spots):
        self.spots = spots
    def compute_frc(self,x_range,y_range,nsamples=1000,sigma=2.0,scale=0.1,plot_kde=False,plot_fft=False):
        #scale has units [pixels <length unit>^-1]
        spots1 = self.spots.sample(nsamples,replace=False)
        sampled_index = spots1.index
        self.spots = self.spots.drop(index=sampled_index)
        spots2 = self.spots.sample(nsamples,replace=False)
        
        kde1 = KDE(spots1); kde2 = KDE(spots2)
        x_range = np.arange(x_range[0],x_range[1],1)
        y_range = np.arange(y_range[0],y_range[1],1)
        sr1 = kde1.get_kde(x_range=x_range,y_range=y_range,sigma=sigma)
        sr2 = kde2.get_kde(x_range=x_range,y_range=y_range,sigma=sigma)
        
        if plot_kde:
            fig,ax=plt.subplots(2,1,sharex=True,sharey=True)
            ax[0].imshow(sr1,cmap='gray',vmin=0,vmax=0.1)
            ax[1].imshow(sr2,cmap='gray',vmin=0,vmax=0.1)
            ax[0].set_xticks([]); ax[0].set_yticks([])
            ax[1].set_xticks([]); ax[1].set_yticks([])
            plt.show()
                  
        frc_curve, fourier1, fourier2 = frc.two_frc(sr1,sr2)
        if plot_fft:
            fig,ax=plt.subplots(2,1,sharex=True,sharey=True)
            ax[0].imshow(np.abs(np.asarray(fourier1)),cmap='gray',vmin=0.0,vmax=500)
            ax[1].imshow(np.abs(np.asarray(fourier2)),cmap='gray',vmin=0.0,vmax=500)
            ax[0].set_xticks([]); ax[0].set_yticks([])
            ax[1].set_xticks([]); ax[1].set_yticks([])
            plt.tight_layout(); plt.show()
        img_size = sr1.shape[0]
        xs_pix = np.arange(len(frc_curve)) / img_size
        
        xs_nm_freq = xs_pix * scale
        frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
        return xs_nm_freq, thres(xs_nm_freq), frc_curve, frc_res


