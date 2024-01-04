import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import trackpy as tp
import json

class Tracker2D:
    def __init__(self):   
        pass 
    def link(self,spots,search_range=3,memory=5):        
        spots = spots.dropna(subset=['x_mle','y_mle','frame'])
        pos = ['x_mle','y_mle']
        spots = tp.link_df(spots,search_range=search_range,
                           memory=memory,pos_columns=pos)
        return spots
        
    def imsd(self,spots,mpp=0.1083,fps=10.0,max_lag=10,pos_columns=['x_mle','y_mle']): 
        return tp.imsd(spots,mpp,fps,max_lagtime=max_lag,
                       statistic='msd',pos_columns=pos_columns)
                       
    def vanhove(self,spots,mpp=0.1083,lagtime=1):
        pos = spots.set_index(['frame', 'particle'])['x_mle'].unstack()
        return tp.vanhove(pos,lagtime,mpp=mpp)

