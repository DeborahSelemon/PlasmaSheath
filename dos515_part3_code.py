#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov 12 16:07:07 2022

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

def Poisson(start, x, vs_cap, L_cap):
    # This function solves the normalized Poisson equation written as two first order ODEs
    # dphi_cap/dx_cap = -E_cap && dE_cap/dx_cap = ni_cap - ne_cap = vs_cap/sqrt(vs_cap**2-2*phi_cap)-exp(phi_cap)
    # cap refers to normalized values
    phi_cap = start[0]
    E_cap = start[1]
    vi_cap = start[2]

    ni_cap = vs_cap/vi_cap #Ion continuity
    ne_cap = np.exp(phi_cap) #Boltzmann electrons

    # Calculate the derivatives
    dphicap_dxcap = -E_cap #Electric field
    dEcap_dxcap = ni_cap - ne_cap #Poission's law
    dvicap_dxcap = E_cap/vi_cap - vi_cap/L_cap #Energy conservation

    # Return the space derivatives in the same order as in the function input
    return [dphicap_dxcap, dEcap_dxcap, dvicap_dxcap]

vs_cap = [1] #Values for ion velocity at start of sheath that we want to calculate for
L_cap = [0.1,1,10,100,1000,10000] #Values for length over which ion momentum is lost that we want to calculate for
initial_cond = [0, 0.001, 1]  # Starting values for phi,E and vi at x =0 respectively [i.e phi0 = 0, E0 = 0.001, vi = 1]
x = linspace(0, 40, 100) #Positions where we want to calculate phicap, Ecap and jcap
mi_me_ratio = 1840


fig, ax1 = plt.subplots() 
fig.set_size_inches([12,8])#Setting plot size

xwalls = [] #List to store position of wall for calculations with each value of L
vi_walls = [] #List to store value of ion velocty at wall for each value of L

for i in range(len(L_cap)):
    solution = odeint(Poisson, initial_cond, x, args = (vs_cap,L_cap[i],))
    phicap_vals = solution[:, 0]
    Ecap_vals = solution[:, 1]
    vi_vals = solution[:,2]

    # Solving for the normalized current density: jcap = sqrt(mi/me*2*pi)*exp(phicap) -1
    jcap_vals = (np.sqrt(mi_me_ratio/np.pi)*np.exp(phicap_vals)) - 1

    #Interpolating for the value of xwall (position where j crosses zero for each L solution)
    interp_func = interp1d(jcap_vals,x)
    xwall = interp_func(0)
    xwalls.append(xwall)
    
    #Interpolating for the value of vi_wall (ion velocity at the wall, where vi crosses zero for each L solution)
    interp_func_2 = interp1d(x-xwall,vi_vals)
    vi_wall = interp_func_2(0)
    vi_walls.append(vi_wall)

    # Plot normalized ion velocity vi_cap as a function of distance to wall x-xwall for given values of L
    ax1.plot(x-xwall,vi_vals,label = 'L ='+str(L_cap[i]))
    
#Other plot definitions
ax1.grid(True)
ax1.set_xlabel("Distance of plasma to wall \n [normalized to $\lambda_D$]", fontsize=20)
ax1.set_ylabel("Ion velocity $\hat{v}_i$ \n [normalized to $c_s$]", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax1.set_title("$\hat{v}_i$(x) \n [for pure hydrogen plasma where $m_i$/$m_e$ = 1840]", fontsize=24)
ax1.legend(fontsize=20, loc = 'upper right')
ax1.set_ylim(0,5)
ax1.set_xlim(-20,20)  

#Plotting ion velocity at wall as a function of collision length overlaid on first plot
ax2 = ax1.inset_axes([0.13,0.50,0.24,0.45]) #digits in bracket represent (xposition, yposition,width,height) of overliad axes
ax2.plot(L_cap,vi_walls)
ax2.set_xscale('log')
ax2.set_xlabel("Collision length, L", fontsize=15)
ax2.set_ylabel("Ion velocity $\hat{v}_i$ at wall \n [normalized to $c_s$]", fontsize=15)
ax2.set_yticks([0.5,1,1.5,2,2.5,3],fontsize=15)
ax2.set_xticks([0.1,1,10,100,1000,10000],fontsize=15)
