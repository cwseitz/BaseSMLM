import numpy as np
import tifffile
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from glob import glob

class SMLMDataset(Dataset):
    def __init__(self,path,name):
        self.stack = tifffile.imread(path+'.tif')
        self.name = name
        try:
            prefix = path.split('.')[0]
            npz = np.load(prefix+'.npz',allow_pickle=True)
            self.theta = npz['theta']
        except Exception as e:
            self.theta = None
    def get_theta(self,idx):
        if self.theta is not None:
            return self.theta[idx]
        else:
            return None
    def __len__(self):
        nb,nc,nx,ny = self.stack.shape
        return nb
    def __getitem__(self, idx):
        adu = self.stack[idx]
        return adu
