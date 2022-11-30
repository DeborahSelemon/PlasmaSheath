#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:41:15 2022

@author: dos515
"""
"""
Using SciPy's odeint library to integrate functions - Plasma Sheath Problem
"""

# Note: The equations being integrated have been normalized.

from scipy.integrate import odeint # SciPy ODE integration
from numpy import linspace
import matplotlib.pyplot as plt  # Plotting library
import numpy as np
from scipy.interpolate import interp1d

def Poisson(start, x, vs_cap):
    # This function solves the normalized Poisson equation written as two first order ODEs
    # dphi_cap/dx_cap = -E_cap && dE_cap/dx_cap = ni_cap - ne_cap = vs_cap/sqrt(vs_cap**2-2*phi_cap)-exp(phi_cap)
    # cap refers to normalized values
    phicap = start[0]
    Ecap = start[1]

    vi_cap = np.sqrt((vs_cap**2)-2*phicap) #Energy conservation
    ni_cap = vs_cap/vi_cap #Ion continuity
    ne_cap = np.exp(phicap) #Boltzmann electrons

    # Calculate the derivatives
    dphicap_dxcap = -Ecap #Electric field
    dEcap_dxcap = ni_cap - ne_cap #Poission's law

    # Return the space derivatives in the same order as in the function input
    return [dphicap_dxcap, dEcap_dxcap]

vs_cap = [1,1.5,2] #Values for ion velocity at start of sheath that we want to calculate for
initial_cond = [0, 0.001]  # Starting values for phi and E at x =0 respectively [i.e phi0 = 0, E0 = 0.001]
x = linspace(0, 40, 100) #Positions where we want to calculate phicap, Ecap and jcap
mi_me_ratio = 1840

plt.figure(figsize=(12, 7), dpi=1000) #Setting plot size

for i in range(len(vs_cap)):
    solution = odeint(Poisson, initial_cond, x, args = (vs_cap[i],))
    phicap_vals = solution[:, 0]
    Ecap_vals = solution[:, 1]

    # Solving for the normalized current density: jcap = sqrt(mi/me*2*pi)*exp(phicap) -1
    jcap_vals = (np.sqrt(mi_me_ratio/np.pi)*np.exp(phicap_vals)) - 1

    #Plotting the solutions together such that j=0 coincides at x=0
    #Interpolating for the value of xwall (position where j crosses zero for each vs solution)
    interp_func = interp1d(jcap_vals,x)
    xwall = interp_func(0)

    # Plot normalized current density j_cap as a function of distance x for given values of vs
    plt.plot(x-xwall, jcap_vals, linewidth=3, label = '$\hat{v}_s$ ='+'{vs}'.format(vs=vs_cap[i]))
    
#Other plot definitions
plt.grid(True)
plt.xlabel("Distance of plasma from wall $\hat{x}$ \n [normalized to $\lambda_D$]", fontsize=20)
plt.ylabel("Current density $\hat{j}$ \n [normalized to $en_sc_s$]", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title("$\hat{j}$(x) \n [for pure hydrogen plasma where $m_i$/$m_e$ = 1840]", fontsize=24)
plt.legend(fontsize=20)
plt.show()    

