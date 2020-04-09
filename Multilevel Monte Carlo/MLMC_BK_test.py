
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:43:28 2020
@author: Gong Xue
"""
import pylab 
import time
from scipy.interpolate import CubicSpline
import numpy as np
from random import seed
from numpy.random import randn
import math
import matplotlib.pyplot as plt
# set the state of randn
seed(1)


#evaluate legendre function for x
def Mu(t,T,mu):
    cs = CubicSpline([0,0.5*T,T],mu)
    return cs(t)


#  mlmc_l = function for level l estimator 
def mlmc_l(M,l,N, alpha, mu, T, sigma, r0, L0):
    nf = M**(l+L0)
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
        if nf == 1:
            dWf = math.sqrt(hf)*randn(N2)
            lnrf= lnrf + alpha*(math.log(Mu(t,T,mu))-lnrf)*hf + sigma*dWf
            rf = np.exp(lnrf)
            integralf = integralf + rf*hf
            t = t + hf
        else:
            for n in range(int(nc)): #coarse grid
                dWc = np.zeros(N2)
                for m in range(M): #fine grid
                    dWf = math.sqrt(hf)*randn(N2)
                    dWc = dWc + dWf
                    lnrf  = lnrf + alpha*(math.log(Mu(t,T, mu))-lnrf)*hf + sigma*dWf
                    rf = np.exp(lnrf)
                    integralf = integralf + rf*hf
                    t = t + hf
                lnrc = lnrc + alpha*(math.log(Mu(t,T, mu))-lnrc)*hc + sigma*dWc
                rc = np.exp(lnrc)
                integralc = integralc + rc*hc
       # Pf=lnrf
        #Pc=lnrc
        Pf = np.exp(-integralf) #price estimation using fine grid
        Pc = np.exp(-integralc) #price estimation using coarse grid
        if l == 0:
            Pc = 0
        sums[0] = sums[0] + sum(Pf-Pc)
        sums[1] = sums[1] + sum((Pf-Pc)**2)
        sums[2] = sums[2] + sum(Pf)
        sums[3] = sums[3] + sum(Pf**2)
        #print(np.var(Pf))
        #print(np.var(Pf-Pc))
    return(sums)
    
    
    
def mlmc(M,eps,extrap, alpha, mu, T, sigma, r0, L0):
    start_time = time.time()




    #M=4
    #eps=0.005
    #extrap=1
    #alpha=3
    #mu=[0.01,0.25,0.05]
    #T=10
    #sigma=0.9
    #r0=0.01

    L   = -1
    N   = 10000
    converged = 0
    suml=np.zeros((5,100))
    while converged == 0:
        L = L+1
        print(L)
        sums = mlmc_l(M,L,N, alpha, mu, T, sigma, r0, L0)
        suml[0,L] = N
        suml[1,L] = sums[0] #sum(Pf-Pc) at level L
        suml[2,L] = sums[1] #sum((Pf-Pc).^2) level L
        # optimal sample sizes
    
        Vl = suml[2,np.arange(L+1)]/suml[0,np.arange(L+1)] - (suml[1,np.arange(L+1)]/suml[0,np.arange(L+1)])**2
        #Nl = ceil( 2 * Nl * sum(Vl./Nl) / eps^2);
        Nl = np.ceil(2*np.sqrt(Vl/(M**np.arange(L+1))) * np.sum(np.sqrt(Vl*(M**np.arange(L+1)))) / eps**2)

    
        
        # update sample sums
    
        for l in range(L+1):
            if L == 0  :
                dNl = Nl-suml[0,l]
            else:
                dNl = Nl[l]-suml[0,l]
            if dNl>0:
                sums = mlmc_l(M,l,int(dNl), alpha, mu, T, sigma, r0, L0)
                suml[0,l] = suml[0,l] + dNl;
                suml[1,l] = suml[1,l] + sums[0]
                suml[2,l] = suml[2,l] + sums[1]
                    
        
        #test for convergence
        
        if (extrap==1):
            Range = 0
            if (L>0 and M**(L+L0)>=16):
                con = M**Range*(suml[1,L+Range]/suml[0,L+Range]- (1/M)*suml[1,L+Range-1]/suml[0,L+Range-1] )
                converged = (np.max(abs(con)) < (M**2-1)*eps/math.sqrt(2)) or (M**(L+L0)>1024) 
        else:
            Range = np.arange(-1,1)
            if (L>0 and M**(L+L0)>=16):
                con = (1/(M**(-1*Range)))*suml[1,L+Range]/suml[0,L+Range]
                #converged = (max(abs(con)) < (M-1)*eps/2) or (M**L>1024)
                converged = (np.max(abs(con)) < (M-1)*eps/math.sqrt(2)) or (M**(L+L0)>1024)
        
    # evaluate multi-timestep estimator

    P = sum(suml[1,0:L]/suml[0,0:L])
    if (extrap==1):
        P = P + ( suml[1,L]/suml[0,L] ) / (M-1)
    cost = (1+1/M)*sum(Nl*M**np.arange(L+1))
    print("---MLMC runtime %s seconds ---" % (time.time() - start_time))


    return(P, Nl, cost)

#test1 = mlmc_l(M=4, l =0, N = 10000, alpha=0.9, mu=[0.01,0.25,0.05], T=5, sigma=0.9, r0=0.01, L0=0)
#test2 = mc(M=4,eps=0.0001, alpha=1.9871, mu=[1.30048,1.1365,1.26824], T=5.45182, sigma=0.699227, r0=1.34589)



M=4
N=1000000
alpha=5
mu=[0.02,0.03,0.04]
T=5
sigma=0.5
r0=0.03
L0=2


maxl = 6
del1 = np.zeros(maxl)
del2 = np.zeros(maxl)
var1 = np.zeros(maxl)
var2 = np.zeros(maxl)
N = 1000000
L = np.arange(maxl)
for l in L:
    print(l)
    sums = mlmc_l(M=M,l=l,N=N, alpha=alpha, mu=mu, T=T, sigma=sigma, r0=r0, L0 = 2)
    del1[l] = sums[2]/N #Pf
    del2[l]  = sums[0]/N #Pf-Pc
    var1[l]  = sums[3]/N-(sums[2]/N)**2 #Var[Pf]
    var2[l]  = sums[1]/N-(sums[0]/N)**2 #Var[Pf - Pc]




#plot 3 Nl
Eps = [ 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]
mlmc_cost = np.zeros((len(Eps),2))
std_cost = np.zeros((len(Eps),2))
for extrap in range(1):
    for i in range(len(Eps)):
        eps = Eps[i]
        mlmctest = mlmc(M=M,eps=eps,extrap=1, alpha=alpha, mu=mu, T=T, sigma=sigma, r0=r0, L0 = L0)
        Nl = mlmctest[1]

        l = len(Nl)
        print(Nl*M**(np.arange(l)+L0))
        mlmc_cost[i,extrap] = (1+1/M)*np.sum(Nl*M**(np.arange(l)+L0))
        Nmc = 2*var1[0:l]/eps**2

        std_cost[i,extrap] = np.sum((2*var1[0:l]/eps**2)*M**(np.arange(l)+L0))
print(mlmc_cost)
print(std_cost)



#plot complexity
eps2=np.multiply(Eps,Eps)
print(mlmc_cost[:, 0]*eps2)
print(std_cost[:, 0]*eps2)
line1= pylab.loglog(Eps,mlmc_cost[:, 0], label = 'MLMC')
line2= pylab.loglog(Eps,std_cost[:,0], label = 'MC')
#line3 = plt.loglog(Eps,mlmc_cost[:, 0], label = 'MLMC')
pylab.xlabel('eps')
pylab.ylabel('Cost'); #title(stitle)
pylab.legend()
plt.savefig('/home/s1998345/Documents/Efficient-Approximation-of-Pricing-Functions/MLMC plot/'+'MLMC_P_cost.png')



line1 = pylab.plot(L[1:]+2,np.log(var1[1:])/np.log(M), 'k*--', label = '$\hat{P}_l$')
line2 = pylab.plot(L[1:]+2,np.log(var2[1:])/np.log(M), 'b*-', label = '$\hat{P}_l- \hat{P}_{l-1}$')
line3 = pylab.plot(L[1:]+2,-2*(L[1:]+2)-4, 'r:', label = 'slope $=-2$')
pylab.xlabel('$l$')
pylab.ylabel('$log_M$ Variance'); #title(stitle)
pylab.legend()
plt.savefig('/home/s1998345/Documents/Efficient-Approximation-of-Pricing-Functions/MLMC plot/'+'MLMC_P_variance.png', dpi = 600, bbox_inches = "tight")



line1 = pylab.plot(L[1:]+2,np.log(abs(del1[1:]))/np.log(M), 'k*--',label = '$\hat{P}_l$')
line2 = pylab.plot(L[1:]+2,np.log(abs(del2[1:]))/np.log(M),'b*-', label = '$\hat{P}_l- \hat{P}_{l-1}$')
#line3 = pylab.plot(L[2:],np.log(abs(del2[2:]-del2[1:(maxl-1)]/M))/np.log(M), 'r--',label = 'Y_l-Y_{l-1}/M')
line4 = pylab.plot(L[1:]+2,-1*(L[1:]+2)-0.8, 'r:', label = 'slope $= -1$')
pylab.xlabel('$l$')
pylab.ylabel('$log_M |Mean|$'); 
pylab.legend()
plt.savefig('/home/s1998345/Documents/Efficient-Approximation-of-Pricing-Functions/MLMC plot/'+'MLMC_P_mean.png', dpi = 600, bbox_inches = "tight")
