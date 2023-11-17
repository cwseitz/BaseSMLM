import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def LFilter(X,xlim,ylim,r,T,plot=False):
    """Uses Besag's L-function to filter out isolated points"""
    N = X.shape[0]
    D = distance.cdist(X,X)
    D = D[:N, :N]
    K = np.sum(D <= r, axis=1) - 1
    L = np.sqrt((np.diff(xlim) * np.diff(ylim)) * K / (np.pi * (N - 1)))
    idx = np.argwhere(L < T)
    idxx = np.argwhere(L >= T)
    if plot:
        fig,ax=plt.subplots(1,2,figsize=(6,3))
        ax[0].scatter(X[idx,0],X[idx,1],color='black',s=1,marker='x')
        ax[0].scatter(X[idxx,0],X[idxx,1],color='cornflowerblue',s=1,marker='x')
        ax[0].set_xlabel('X'); ax[0].set_ylabel('Y')
        ax[1].hist(L,bins=30,color='cornflowerblue')
        ax[1].set_xlabel('L'); ax[1].set_ylabel('Counts')
        plt.tight_layout()
        plt.show()
    if len(idxx) > 0:
        X = np.squeeze(X[idxx])
    return X

def Kclust(pts, xlim, ylim, rseq, thseq):
    N = pts.shape[0]
    D = distance.cdist(pts,pts)
    D = D[:N, :N]
    for r in rseq:
        K = np.sum(D <= r, axis=1) - 1
        L = np.sqrt((np.diff(xlim) * np.diff(ylim)) * K / (np.pi * (N - 1)))
        for th in thseq:
            idx = np.argwhere(L < th)
            idxx = np.argwhere(L >= th)
            if len(idx) > 0:
                A = D < 2*r
                A = np.delete(A,idx,axis=0)
                A = np.delete(A,idx,axis=1)
                np.fill_diagonal(A,0)
                csr = csr_matrix(A)
                components, labels = connected_components(csr,directed=False)
                plt.scatter(pts[idxx, 0], pts[idxx, 1], c=labels, cmap='rainbow', s=1,marker='x')
                plt.scatter(pts[idx, 0], pts[idx, 1], color='black', s=1,marker='x')
                plt.show()


