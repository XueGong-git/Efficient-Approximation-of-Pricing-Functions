#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:43:28 2020

@author: s1998345
"""
import time
from scipy.interpolate import CubicSpline
import numpy as np
from random import seed
from numpy.random import randn
import math
# set the state of randn
seed(1)


#evaluate legendre function for x
def Mu(t,T,mu):
    cs = CubicSpline([0,0.5*T,T],mu)
    return cs(t)


#  mlmc_l = function for level l estimator 
def mlmc_l(M,l,N, alpha, mu, T, sigma, r0):
    nf = M**l
    nc = nf/M
    hf = T/nf
    hc = T/nc
    sums = np.zeros(4)
    
    for N1 in np.arange(0,N,10000):  #divide steps N into groups of 10000
        N2 = min(10000,N-N1)
        # GBM model
        lnr0 = math.log(r0)
        rc = np.ones(N2)*r0
        rf = rc
        lnrf = np.ones(N2)*lnr0
        lnrc = lnrf
        integralf = np.zeros(N2)
        integralc = integralf
        t=0
        if l == 0:
            dWf = math.sqrt(hf)*randn(N2)
            t = t + hf
            lnrf= lnrf + alpha*(math.log(Mu(t,T,mu))-lnrf)*hf + sigma*dWf
            rf = np.exp(lnrf)
            integralf = integralf + rf*hf
        else:
            for n in range(int(nc)): #coarse grid
                dWc = np.zeros(N2)
                for m in range(M): #fine grid
                    dWf = math.sqrt(hf)*randn(N2)
                    t = t + hf
                    dWc = dWc + dWf
                    lnrf  = lnrf + alpha*(math.log(Mu(t,T, mu))-lnrf)*hf + sigma*dWf
                    rf = np.exp(lnrf)
                    integralf = integralf + rf*hf
                lnrc = lnrc + alpha*(math.log(Mu(t,T, mu))-lnrc)*hc + sigma*dWc
                rc = np.exp(lnrc)
                integralc = integralc + rc*hc
        Pf = np.exp(-integralf) #price estimation using fine grid
        Pc = np.exp(-integralc) #price estimation using coarse grid
        if l == 0:
            Pc = 0
        sums[0] = sums[0] + sum(Pf-Pc)
        sums[1] = sums[1] + sum((Pf-Pc)**2)
        sums[2] = sums[2] + sum(Pf)
        sums[3] = sums[3] + sum(Pf**2)
    return(sums)
    
    
    
def mlmc(M,eps,extrap, alpha, mu, T, sigma, r0):
    start_time = time.time()

    L   = -1
    N   = 10000
    converged = 0
    suml=np.zeros((3,100))
    while converged == 0:
        L = L+1
        print(L)
        sums = mlmc_l(M,L,N, alpha, mu, T, sigma, r0)
        suml[0,L] = N
        suml[1,L] = sums[0] #sum(Pf-Pc) at level L
        suml[2,L] = sums[1] #sum((Pf-Pc).^2) level L
    
    
        # optimal sample sizes
    
        Vl = suml[2,:]/suml[0,:] - (suml[1,:]/suml[0,:])*(suml[1,:]/suml[0,:])
        #Nl = ceil( 2 * Nl * sum(Vl./Nl) / eps^2);
        Nl = np.ceil(2*np.sqrt(Vl[np.arange(L+1)]/(M**np.arange(L+1))) * np.sum(np.sqrt(Vl[np.arange(L+1)]*(M**np.arange(L+1)))) / eps**2)

    
        
        # update sample sums
    
        for l in range(L+1):
            if L == 0  :
                dNl = Nl-suml[0,l]
            else:
                dNl = Nl[l]-suml[0,l]
            if dNl>0:
                sums = mlmc_l(M,l,int(dNl), alpha, mu, T, sigma, r0)
                suml[0,l] = suml[0,l] + dNl;
                suml[1,l] = suml[1,l] + sums[0]
                suml[2,l] = suml[2,l] + sums[1]
    
            
        
        #test for convergence
        
        if (extrap==1):
            Range = 0
            if (L>1 and M**L>=16):
                con = M**Range*(suml[1,L+Range]/suml[0,L+Range]- (1/M)*suml[1,L+Range-1]/suml[0,L+Range-1] )
                converged = (np.max(abs(con)) < (M**2-1)*eps/math.sqrt(2)) or (M**L>1024) 
        else:
            Range = np.arange(-1,1)
            if (L>1 and M**L>=16):
                con = (1/(M**(-1*Range)))*suml[1,L+Range]/suml[0,L+Range]
                #converged = (max(abs(con)) < (M-1)*eps/2) or (M**L>1024)
                converged = (np.max(abs(con)) < (M-1)*eps/math.sqrt(2)) or (M**L>1024)
        
    # evaluate multi-timestep estimator

    P = sum(suml[1,0:L]/suml[0,0:L])
    if (extrap==1):
        P = P + ( suml[1,L]/suml[0,L] ) / (M-1)
    
    mlmc_cost = (1+1/M)*sum(Nl*M**np.arange(L+1))
    print("---MLMC runtime %s seconds ---" % (time.time() - start_time))

    return(P, Nl, mlmc_cost, con)

test = mlmc(M=4,eps=0.01,extrap=0, alpha=0.1, mu=[10,9,8], T=1, sigma=0.1, r0=1)