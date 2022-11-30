#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:18:12 2022
@author: dos515
"""

"""
Using SciPy's odeint library to integrate functions - Plasma Sheath Problem
"""

from scipy.integrate import odeint # SciPy ODE integration
from numpy import linspace
import matplotlib.pyplot as plt  # Plotting library
import numpy as np

#Note: The equations being integrated have been normalized. 

def Poisson(start, x):
    # This function solves the normalized Poisson equation written as two first order ODEs
    #dphi_cap/dx_cap = -E_cap && dE_cap/dx_cap = 1/sqrt(1-2*phi_cap)-exp(phi_cap) 
    #cap refers to normalized values
#DD: Nice variable names 
    phicap= start[0]
    Ecap= start[1]
    
    # Calculate the derivatives
    dphicap_dxcap= -Ecap

#DD: The code might be a bit clearer if you do this in several steps, i.e. ni=...
# ne = ..., etc.

#DD: You've hard coded vs_hat = 1 here. You'll need to generalise this in future.
    dEcap_dxcap = (1/np.sqrt(1-2*phicap)) - np.exp(phicap)
    
    # Return the space derivatives in the same order as in the function input 
    return [dphicap_dxcap, dEcap_dxcap]


#DD: You comment is helpful to make it clear what these numbers represent
#but it could be clearer with named variables (phi0=....)
initial_cond = [0,0.001] # Starting values for phi and E at x =0 
x = linspace(0, 40, 100)
solution = odeint(Poisson, initial_cond, x)
phicap = solution[:,0]
Ecap = solution[:,1]

#Solving for the normalized current density: jcap = sqrt(1840/2pi)*exp(phicap) -1
jcap = np.zeros(len(phicap))
for i in range(len(phicap)):
#DD: I'd recommend avoiding magic numbers like 1840. It's much better to use
#a named variable such as mass_ratio or mi_over_me etc .    
    jcap[i] = (np.sqrt(1840/np.pi)*np.exp(phicap[i])) - 1
    
    
#%% - PLOTS

#Axes are labelled 'normalized to' based on quantity the original variable was divided by
    
#Plot normalized electric potential phi_cap as a function of distance x
plt.figure(figsize =(12,7),dpi=1000)
plt.plot(x,phicap, 'g',linewidth = 3)
plt.grid(True)  
plt.xlabel("Distance of plasma to sheath $\hat{x}$ \n [normalized to $\lambda_D$]", fontsize = 20)
plt.ylabel("Electric potential $\hat{\phi}$ \n [normalized to $T_e$]", fontsize =20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.xlim(0,40)
plt.title("$\hat{\phi}$(x)  \n [$\hat{\phi}_0$ = 0, $\hat{E}_0$ = 0.001, $\hat{v}_s$ = 1]", fontsize = 24)
plt.show()

#Plot normalized electric field E_cap as a function of distance x
plt.figure(figsize =(12,7),dpi=1000)
plt.plot(x,Ecap, 'b',linewidth = 3)
plt.grid(True)  
plt.xlabel("Distance of plasma to sheath $\hat{x}$ \n [normalized to $\lambda_D$]", fontsize = 20)
plt.ylabel("Electric field $\hat{E}$ \n [normalized to $T_e$/$\lambda_D$]", fontsize =20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.xlim(0,40)
plt.title("$\hat{E}$(x) \n [$\hat{\phi}_0$ = 0, $\hat{E}_0$ = 0.001, $\hat{v}_s$ = 1]", fontsize = 24)
plt.show()    

#Plot normalized current density j_cap as a function of distance x
plt.figure(figsize =(12,7),dpi=1000)
plt.plot(x,jcap, 'r', linewidth = 3)
plt.grid(True)
#DD: Nicely presented plots
plt.xlabel("Distance of plasma to sheath $\hat{x}$ \n [normalized to $\lambda_D$]", fontsize = 20)
plt.ylabel("Current density $\hat{j}$ \n [normalized to $en_sc_s$]", fontsize =20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.ylim(-5,25)
plt.xlim(0,40)
plt.title("$\hat{j}$(x) \n [for pure hydrogen plasma where $m_i$/$m_e$ = 1840]", fontsize = 24)
plt.show()

#%%

plt.figure(figsize =(12,7),dpi=1000)
plt.plot(x,phicap, 'g',linewidth = 3, label ='$\hat{\phi}(x)$')
plt.plot(x,Ecap, 'b',linewidth = 3, label ='$\hat{E}$(x)')
plt.plot(x,jcap, 'r', linewidth = 3, label ='$\hat{j}(x)$')
plt.grid(True)  
plt.xlabel("Distance of plasma to sheath $\hat{x}$ \n [normalized to $\lambda_D$]", fontsize = 20)
#plt.ylabel("Electric potential $\hat{\phi}$ \n [normalized to $T_e$]", fontsize =20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.xlim(0,40)
plt.title("$\hat{\phi}$(x), $\hat{E}$(x), $\hat{j}$(x)  \n [with $\hat{\phi}_0$ = 0, $\hat{E}_0$ = 0.001, $\hat{v}_s$ = 1] \n [for pure hydrogen plasma where $m_i$/$m_e$ = 1840]", fontsize = 24)
plt.legend(fontsize=20)
plt.show()