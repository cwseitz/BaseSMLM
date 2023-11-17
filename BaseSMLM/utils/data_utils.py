import numpy as np
import torch

def batch_xyz_to_boolean_grid(xyz_np, upsampling_factor, pixel_size_lateral, pixel_size_axial, zhrange, grid_shape):
       
    H, W, D = grid_shape
    batch_size, num_particles, nfeatures = xyz_np.shape

    zshift = xyz_np[:,:,2] + zhrange/pixel_size_axial       
    xg = (np.floor(upsampling_factor*xyz_np[:,:,0])).astype('int')
    yg = (np.floor(upsampling_factor*xyz_np[:,:,1])).astype('int')
    zg = (np.floor(zshift)).astype('int')
   
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()
    
    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indX, indY])
    else:
        ibool = torch.LongTensor([indZ, indX, indY])

    vals = torch.ones(batch_size*num_particles)
    if batch_size > 1:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([batch_size, D, H, W])).to_dense()
    else:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([D, H, W])).to_dense()
    return boolean_grid
 
def errors2d(spots,theta):

    def match_on_xy(xys,xy,tol=1.0):
        xys = np.array(xys, dtype=float)
        dist = np.linalg.norm(xys-xy, axis=1)
        a = dist <= tol
        b = np.any(a)
        if b:
            idx = np.argmin(dist)
            c = np.squeeze(xys[idx])
            xerr,yerr = xy[0]-c[0],xy[1]-c[1]
        else:
            xerr,yerr = None,None
        return b,xerr,yerr

    nt,ntheta,k = theta.shape
    all_bool = []; all_x_err = []; all_y_err = []
    for n in range(nt):
        this_theta = theta[n]
        this_spots =\
        spots.loc[spots['frame'] == n]
        xys_np = this_spots[['x','y']].to_numpy()
        for m in range(k):
            xy = this_theta[:2,m]
            bool,xerr,yerr = match_on_xy(xys_np,xy)
            all_bool.append(bool)
            if xerr is not None and yerr is not None:
                all_x_err.append(xerr)
                all_y_err.append(yerr)     
    all_bool = np.array(all_bool)
    all_x_err = np.array(all_x_err)
    all_y_err = np.array(all_y_err)
    num_found = np.sum(all_bool) #intersection (true positives)
    num_false_n = nt*k-num_found #false negatives (missed)
    num_false_p = len(spots)-num_found #false positives
    jaccard = num_found/(num_found+num_false_n+num_false_p)
    return all_x_err, all_y_err, jaccard

def errors3d(spots,theta):

    def match_on_xy(xyzs,xyz,tol=1.0):
        xys = np.array(xyzs[:,:2], dtype=float) #drop z index
        dist = np.linalg.norm(xys-xyz[:2], axis=1)
        a = dist <= tol
        b = np.any(a)
        if b:
            idx = np.argmin(dist)
            c = np.squeeze(xyzs[idx])
            xerr,yerr,zerr = xyz[0]-c[0],xyz[1]-c[1],xyz[2]-c[2]
        else:
            xerr,yerr,zerr = None,None,None
        return b,xerr,yerr,zerr
        
    nt,ntheta,k = theta.shape
    all_bool = []; all_x_err = []; all_y_err = []; all_z_err = []
    for n in range(nt):
        this_theta = theta[n]
        this_spots =\
        spots.loc[spots['frame'] == n]
        xyzs_np = this_spots[['x','y','z']].to_numpy()
        for m in range(k):
            xyz = this_theta[:3,m]
            bool,xerr,yerr,zerr = match_on_xy(xyzs_np,xyz)
            all_bool.append(bool)
            if xerr is not None and yerr is not None:
                all_x_err.append(xerr)
                all_y_err.append(yerr)  
                all_z_err.append(zerr)
                   
    all_bool = np.array(all_bool)
    all_x_err = np.array(all_x_err)
    all_y_err = np.array(all_y_err)
    all_z_err = np.array(all_z_err)
    num_found = np.sum(all_bool) #intersection (true positives)
    num_false_n = nt*k-num_found #false negatives (missed)
    num_false_p = len(spots)-num_found #false positives
    jaccard = num_found/(num_found+num_false_n+num_false_p)
    return all_x_err, all_y_err, all_z_err, jaccard
