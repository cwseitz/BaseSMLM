from numpy import exp, pi, sqrt
from scipy.special import erf
import matplotlib.pyplot as plt
import numpy as np

def lamx(X,x0,sigma_x):
    alpha_x = sqrt(2)*sigma_x
    return 0.5*(erf((X+0.5-x0)/alpha_x)-erf((X-0.5-x0)/alpha_x))
    
def lamy(Y,y0,sigma_y):
    alpha_y = sqrt(2)*sigma_y
    return 0.5*(erf((Y+0.5-y0)/alpha_y)-erf((Y-0.5-y0)/alpha_y))

def sx(z0):
    a0 = 3.053
    a1 = -0.934
    a2 = -0.325
    a3 = -0.025
    sigma_x = a0 + a1*z0 + a2*z0**2 + a3*z0**3
    return sigma_x
    
def sy(z0):
    a0 = 2.690
    a1 = 1.693
    a2 = 0.724
    a3 = 0.156
    sigma_y = a0 + a1*z0 + a2*z0**2 + a3*z0**3
    return sigma_y
     
def dudn0(X,Y,x0,y0,sigma_x,sigma_y):
    return lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
     
def dudx0(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(sqrt(2*pi)*sigma_x)
    return A*lamy(Y,y0,sigma_y)*(exp(-(X-0.5-x0)**2/(2*sigma_x**2))-exp(-(X+0.5-x0)**2/(2*sigma_x**2)))
    
def dudy0(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(sqrt(2*pi)*sigma_y)
    return A*lamx(X,x0,sigma_x)*(exp(-(Y-0.5-y0)**2/(2*sigma_y**2))-exp(-(Y+0.5-y0)**2/(2*sigma_y**2)))
   
def dudz0(X,Y,x0,y0,sigma_x,sigma_y,z0):
    return dudsx(X,Y,x0,y0,sigma_x,sigma_y)*dsxdz0(z0) + dudsy(X,Y,x0,y0,sigma_x,sigma_y)*dsydz0(z0)
 
def dsxdz0(z0):
    a1 = -0.934
    a2 = -0.325
    a3 = -0.025
    return a1 + 2*a2*z0 + 3*a3*z0**2
    
def dsydz0(z0):
    a1 = 1.693
    a2 = 0.724
    a3 = 0.156
    return a1 + 2*a2*z0 + 3*a3*z0**2
    
def dudsx(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(sqrt(2*pi)*sigma_x**2)
    return A*lamy(Y,y0,sigma_y)*((X-x0-0.5)*exp(-(X-0.5-x0)**2/(2*sigma_x**2))-(X-x0+0.5)*exp(-(X+0.5-x0)**2/(2*sigma_x**2)))
    
def dudsy(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(sqrt(2*pi)*sigma_y**2)
    return A*lamx(X,x0,sigma_x)*((Y-y0-0.5)*exp(-(Y-0.5-y0)**2/(2*sigma_y**2))-(Y-y0+0.5)*exp(-(Y+0.5-y0)**2/(2*sigma_y**2)))

def duds0(X,Y,x0,y0,sigma_x,sigma_y):
    return dudsx(X,Y,x0,y0,sigma_x,sigma_y)+dudsy(X,Y,x0,y0,sigma_x,sigma_y)
    
def jac1(X,Y,theta,cmos_params):
    x0,y0,z0,N0 = theta
    eta,texp,gain,offset,var = cmos_params
    sigma_x = sx(z0); sigma_y = sy(z0)
    i0 = N0*eta*gain*texp
    j_x0 = i0*dudx0(X,Y,x0,y0,sigma_x,sigma_y)
    j_y0 = i0*dudy0(X,Y,x0,y0,sigma_x,sigma_y)
    j_z0 = i0*dudz0(X,Y,x0,y0,sigma_x,sigma_y,z0)
    j_n0 = (i0/N0)*dudn0(X,Y,x0,y0,sigma_x,sigma_y)
    jac = np.array([j_x0, j_y0, j_z0, j_n0], dtype=np.float64)
    return jac
    
def jac2(adu,X,Y,theta,cmos_params):
    x0,y0,z0,N0 = theta
    eta,texp,gain,offset,var = cmos_params
    sigma_x = sx(z0); sigma_y = sy(z0)
    i0 = N0*eta*gain*texp
    lam = lamx(X,x0,sigma_x)*lamy(Y,y0,sigma_y)
    mu = i0*lam + var
    jac2 = 1 - adu/mu
    return jac2.flatten()

