# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:42:51 2022


Purpose
-------
    A 0D chemostat model of redox reactions occuring in marine OMZs
    
    This model predicts the outcome of microbial populations competing for resources
    in low oxygen zones using the energetics and stoichiometries of the reactions they
    perform.
    
    Sources of competition are:
        - between denitrifying heterotrophs and aerobic heterotrophs for Organic matter
        - between ammonia oxidising archaea and anaerobic ammonium oxidisers for NH4
        - between nitrite oxidising bacteria, denitrifying heterotrophs and anaerobic ammonium oxidisers for NO2
        - between ammonia oxidising archaea, nitrite oxidising bacteria and aerobic heterotrophs for O2
    
    Possible symbioses are:
        - heterotrophs producing NH4 for ammonia oxidising archaea and anammox bacteria
        - ammonia oxidising archaea producing NO2 for denitrifiers
        - nitrite oxidising archaea producing NO3 for denitrifiers
        - denitrifiers producing NO2 for nitrite oxidising bacteria and anammox bacteria
        - anammox bactria producing NO3 for denitrifiers
        - phytoplankton in the deep chlorophyll max producing O2 for heterotrophs??


@author: Emily Zakem & Pearse Buchanan
"""


#%% imports

import sys
import os
import numpy as np
import xarray as xr
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten

# numerical packages
from numba import jit


# print versions of packages
print("python version =",sys.version[:5])
print("numpy version =", np.__version__)
print("xarray version =", xr.__version__)
print("pandas version =", pd.__version__)
print("netCDF4 version =", nc.__version__)
print("seaborn version =", sb.__version__)
print("matplotlib version =", sys.modules[plt.__package__].__version__)
print("cmocean version =", sys.modules[cmo.__package__].__version__)


#%% Set initial conditions and incoming concentrations to chemostat experiment

### Organic matter (S)
# range of org C flux from 2-8 uM C m-2 day-1 --> 0.26 - 1 uM N m-3 day-1, assuming 1 m3 box and flux into top of box
Sd0_exp = 1.5    # uM N  (dissolved; d)
Sp0_exp = 0.0    # uM N  (particulate; p)

### Oxygen
O20_exp = 5.0   # uM O2

### set pulsing terms
xpulse_int = np.arange(2,51,2)
xpulse_O2 = np.arange(0.2,5.1,0.2)

### model parameters
dil = 0.05  # dilution rate (1/day)
days = 1e4  # number of days to run chemostat
dt = 0.001  # timesteps per day (days)
timesteps = days/dt     # number of timesteps
out_at_day = 10.0       # output results this often (days)
nn_output = days/out_at_day     # number of entries for output


#%% initialise arrays for output

print((len(xpulse_int), len(xpulse_O2)))
# Nutrients
fin_O2 = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_Sd = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_Sp = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_NO3 = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_NO2 = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_NH4 = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_N2 = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan

# Biomasses
fin_bHet = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_bFac = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_b1Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_b2Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_b3Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_bAOO = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_bNOO = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_bAOX = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan

# Growth rates
fin_uHet = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_uFac = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_u1Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_u2Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_u3Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_uAOO = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_uNOO = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_uAOX = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan

# track facultative average respiration
fin_facaer = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan

# Rates
fin_rHet = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_rHetAer = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_rO2C = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_r1Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_r2Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_r3Den = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_rAOO = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_rNOO = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan
fin_rAOX = np.ones((len(xpulse_int), len(xpulse_O2))) * np.nan


#%% set traits of the different biomasses

os.chdir("C://Users/pearseb/Dropbox/PostDoc/my articles/Buchanan & Zakem - aerobic anaerobic competition/model and scripts/0D_redox_model")

#from traits_old import *
from traits_new import *


#%% calculate R*-stars for all microbes

from diffusive_o2_coefficient import po_coef
from O2_star import O2_star
from R_star import R_star


print("Oxygen")
print("Subsistence O2 to support bulk aerobic heterotrophy =", O2_star(dil, Qc_aer, diam_aer, dc, y_oO2*CN_aer))
print("Subsistence O2 to support bulk aerobic (facultative) heterotrophy =", O2_star(dil, Qc_aer, diam_aer, dc, y_oO2Fac*CN_aer))
print("Subsistence O2 to support bulk aerobic ammonia oxidation =", O2_star(dil, Qc_aoo, diam_aoo, dc, y_oAOO*CN_aoo))
print("Subsistence O2 to support bulk aerobic nitrite oxidation =", O2_star(dil, Qc_noo, diam_noo, dc, y_oNOO*CN_noo))

print("Organic matter")
print("Subsistence Norg to support bulk aerobic hetertrophy =", R_star(dil, K_s, VmaxS, y_oHet))
print("Subsistence Norg to support bulk aerobic (facultative) hetertrophy =", R_star(dil, K_s, VmaxS, y_oHetFac))
print("Subsistence Norg to support bulk anaerobic nitrate reducing hetertrophy =", R_star(dil, K_s, VmaxS, y_n1Den))
print("Subsistence Norg to support bulk anaerobic nitrite reducing hetertrophy =", R_star(dil, K_s, VmaxS, y_n2Den))

print("Ammonium")
print("Subsistence NH4 to support bulk aerobic ammonia oxidation =", R_star(dil, K_n_AOO, VmaxN_AOO, y_nAOO))
print("Subsistence NH4 to support bulk anaerobic ammonium oxidation (anammox) =", R_star(dil, K_nh4_AOX, VmaxNH4_AOX, y_nh4AOX))

print("Nitrite")
print("Subsistence NO2 to support bulk denitrification (NO2 --> N2) =", R_star(dil, K_n_Den, VmaxN_2Den, y_n2NO2))
print("Subsistence NO2 to support bulk aerobic nitrite oxidation =", R_star(dil, K_n_NOO, VmaxN_NOO, y_nNOO))
print("Subsistence NO2 to support bulk anaerobic ammonium oxidation (anammox) =", R_star(dil, K_no2_AOX, VmaxNO2_AOX, y_no2AOX))

print("Nitrate")
print("Subsistence NO3 to support bulk nitrate reduction by obligate anaerobe =", R_star(dil, K_n_Den, VmaxN_1Den, y_n1NO3))
print("Subsistence NO3 to support bulk nitrate reduction by facultative anaerobe =", R_star(dil, K_n_Den, VmaxN_1DenFac, y_n1NO3Fac))


#%% begin loop of experiments

from model import OMZredox
from line_plot import line_plot

for k in np.arange(len(xpulse_int)):
    for m in np.arange(len(xpulse_O2)):
        print(k,m, xpulse_int[k], xpulse_O2[m])
        
        # 1) Chemostat inputs
        in_Sd = Sd0_exp
        in_O2 = O20_exp
        in_Sp = 0.0  
        in_NO3 = 30.0
        in_NO2 = 0.0
        in_NH4 = 0.0
        
        # 2) Initial biomasses (set to 0.0 to exclude)
        in_bHet = 0.1
        in_bFac = 0.0
        in_b1Den = 0.1
        in_b2Den = 0.1
        in_b3Den = 0.0
        in_bAOO = 0.1
        in_bNOO = 0.1
        in_bAOX = 0.1
        
        # pulse conditions
        pulse_int = xpulse_int[k]
        pulse_Sd = 0.0
        pulse_bHet = 0.0
        pulse_bFac = 0.0
        pulse_O2 = xpulse_O2[m]
        
        # 3) Call main model
        results = OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
                           pulse_Sd, pulse_bHet, pulse_bFac, pulse_O2, pulse_int, \
                           po_aer, po_aoo, po_noo, \
                           VmaxS, K_s, \
                           VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_1DenFac, K_n_Den, \
                           VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, \
                           VmaxNH4_AOX, K_nh4_AOX, VmaxNO2_AOX, K_no2_AOX, \
                           y_oHet, y_oO2, y_oHetFac, y_oO2Fac, \
                           y_n1DenFac, y_n1NO3Fac, y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, \
                           y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
                           e_n2Den, e_n3Den, e_no3AOX, e_n2AOX, \
                           in_Sd, in_Sp, in_O2, in_NO3, in_NO2, in_NH4, \
                           in_bHet, in_bFac, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX)
                
        out_Sd = results[0]
        out_Sp = results[1]
        out_O2 = results[2]
        out_NO3 = results[3]
        out_NO2 = results[4]
        out_NH4 = results[5]
        out_N2 = results[6]
        out_bHet = results[7]
        out_bFac = results[8]
        out_b1Den = results[9]
        out_b2Den = results[10]
        out_b3Den = results[11]
        out_bAOO = results[12]
        out_bNOO = results[13]
        out_bAOX = results[14]
        out_uHet = results[15]
        out_uFac = results[16]
        out_u1Den = results[17]
        out_u2Den = results[18]
        out_u3Den = results[19]
        out_uAOO = results[20]
        out_uNOO = results[21]
        out_uAOX = results[22]
        out_facaer = results[23]
        out_rHet = results[24]
        out_rHetAer = results[25]
        out_rO2C = results[26]
        out_r1Den = results[27]
        out_r2Den = results[28]
        out_r3Den = results[29]
        out_rAOO = results[30]
        out_rNOO = results[31]
        out_rAOX = results[32]
        
        '''
        # 4) plot the results
        line_plot(nn_output, out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, 
                  out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, 
                  out_rHet, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX)
        '''
        
        # 5) Record solutions in initialised arrays
        fin_O2[k,m] = np.nanmean(out_O2[-200::])
        fin_Sd[k,m] = np.nanmean(out_Sd[-200::])
        fin_Sp[k,m] = np.nanmean(out_Sp[-200::])
        fin_NO3[k,m] = np.nanmean(out_NO3[-200::])
        fin_NO2[k,m] = np.nanmean(out_NO2[-200::])
        fin_NH4[k,m] = np.nanmean(out_NH4[-200::])
        fin_N2[k,m] = np.nanmean(out_N2[-200::])
        fin_bHet[k,m] = np.nanmean(out_bHet[-200::])
        fin_bFac[k,m] = np.nanmean(out_bFac[-200::])
        fin_b1Den[k,m] = np.nanmean(out_b1Den[-200::])
        fin_b2Den[k,m] = np.nanmean(out_b2Den[-200::])
        fin_b3Den[k,m] = np.nanmean(out_b3Den[-200::])
        fin_bAOO[k,m] = np.nanmean(out_bAOO[-200::])
        fin_bNOO[k,m] = np.nanmean(out_bNOO[-200::])
        fin_bAOX[k,m] = np.nanmean(out_bAOX[-200::])
        fin_uHet[k,m] = np.nanmean(out_uHet[-200::])
        fin_uFac[k,m] = np.nanmean(out_uFac[-200::])
        fin_u1Den[k,m] = np.nanmean(out_u1Den[-200::])
        fin_u2Den[k,m] = np.nanmean(out_u2Den[-200::])
        fin_u3Den[k,m] = np.nanmean(out_u3Den[-200::])
        fin_uAOO[k,m] = np.nanmean(out_uAOO[-200::])
        fin_uNOO[k,m] = np.nanmean(out_uNOO[-200::])
        fin_uAOX[k,m] = np.nanmean(out_uAOX[-200::])
        fin_facaer[k,m] = np.nanmean(out_facaer[-200::])
        fin_rHet[k,m] = np.nanmean(out_rHet[-200::])
        fin_rHetAer[k,m] = np.nanmean(out_rHetAer[-200::])
        fin_rO2C[k,m] = np.nanmean(out_rO2C[-200::])
        fin_r1Den[k,m] = np.nanmean(out_r1Den[-200::])
        fin_r2Den[k,m] = np.nanmean(out_r2Den[-200::])
        fin_r3Den[k,m] = np.nanmean(out_r3Den[-200::])
        fin_rAOO[k,m] = np.nanmean(out_rAOO[-200::])
        fin_rNOO[k,m] = np.nanmean(out_rNOO[-200::])
        fin_rAOX[k,m] = np.nanmean(out_rAOX[-200::])
        

print("Oxygen concentrations (nM) = ",fin_O2*1e3)
print("Organic N concentrations = ",fin_Sd)
print("NO3 concentrations = ",fin_NO3)
print("NO2 concentrations = ",fin_NO2)
print("NH4 concentrations = ",fin_NH4)
print("Rate of N2 production = ",(fin_rAOX*y_nh4AOX*e_n2AOX*0.5)+(fin_r2Den+fin_r3Den)*0.5)
print("Proportion of N2 produced by anammox = ", ((fin_rAOX*y_nh4AOX*e_n2AOX*0.5) / ((fin_rAOX*y_nh4AOX*e_n2AOX*0.5)+(fin_r2Den+fin_r3Den)*0.5)*100 ))

# 6. check conservation of mass if dilution rate is set to zero
if dil == 0.0:
    end_N = fin_Sd + fin_Sp + fin_NO3 + fin_NO2 + fin_NH4 + fin_N2 + \
            fin_bHet + fin_bFac + fin_b1Den + fin_b2Den + fin_b3Den + fin_bAOO + fin_bNOO + fin_bAOX
    ini_N = Sd0_exp + in_Sp + in_NO3 + in_NO2 + in_NH4 + \
            in_bHet + in_bFac + in_b1Den + in_b2Den + in_b3Den + in_bAOO + in_bNOO + in_bAOX
    for k in np.arange(len(Sd0_exp)):
        for m in np.arange(len(O20_exp)):
            print(" Checking conservation of N mass ")
            print(" Initial Nitrogen =", ini_N[k])
            print(" Final Nitrogen =", end_N[k,m])
            

del results
del out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2
del out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX
del out_uHet, out_uFac, out_u1Den, out_u2Den, out_u3Den, out_uAOO, out_uNOO, out_uAOX
del out_facaer, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX


#%% save the output to data folder

os.chdir("C://Users/pearseb/Dropbox/PostDoc/my articles/Buchanan & Zakem - aerobic anaerobic competition/data/0D_redox_model_output")

fname = 'newtraits_pulse'

np.savetxt(fname+'_O2.txt', fin_O2, delimiter='\t')
np.savetxt(fname+'_N2.txt', fin_N2, delimiter='\t')
np.savetxt(fname+'_NO3.txt', fin_NO3, delimiter='\t')
np.savetxt(fname+'_NO2.txt', fin_NO2, delimiter='\t')
np.savetxt(fname+'_NH4.txt', fin_NH4, delimiter='\t')
np.savetxt(fname+'_OM.txt', fin_Sd, delimiter='\t')

np.savetxt(fname+'_bHet.txt', fin_bHet, delimiter='\t')
np.savetxt(fname+'_bFac.txt', fin_bFac, delimiter='\t')
np.savetxt(fname+'_b1Den.txt', fin_b1Den, delimiter='\t')
np.savetxt(fname+'_b2Den.txt', fin_b2Den, delimiter='\t')
np.savetxt(fname+'_b3Den.txt', fin_b3Den, delimiter='\t')
np.savetxt(fname+'_bAOO.txt', fin_bAOO, delimiter='\t')
np.savetxt(fname+'_bNOO.txt', fin_bNOO, delimiter='\t')
np.savetxt(fname+'_bAOX.txt', fin_bAOX, delimiter='\t')

np.savetxt(fname+'_uHet.txt', fin_uHet, delimiter='\t')
np.savetxt(fname+'_uFac.txt', fin_uFac, delimiter='\t')
np.savetxt(fname+'_u1Den.txt', fin_u1Den, delimiter='\t')
np.savetxt(fname+'_u2Den.txt', fin_u2Den, delimiter='\t')
np.savetxt(fname+'_u3Den.txt', fin_u3Den, delimiter='\t')
np.savetxt(fname+'_uAOO.txt', fin_uAOO, delimiter='\t')
np.savetxt(fname+'_uNOO.txt', fin_uNOO, delimiter='\t')
np.savetxt(fname+'_uAOX.txt', fin_uAOX, delimiter='\t')

np.savetxt(fname+'_facaer.txt', fin_facaer, delimiter='\t')
np.savetxt(fname+'_rHet.txt', fin_rHet, delimiter='\t')
np.savetxt(fname+'_rHetAer.txt', fin_rHetAer, delimiter='\t')
np.savetxt(fname+'_r1Den.txt', fin_r1Den, delimiter='\t')
np.savetxt(fname+'_r2Den.txt', fin_r2Den, delimiter='\t')
np.savetxt(fname+'_r3Den.txt', fin_r3Den, delimiter='\t')
np.savetxt(fname+'_rAOO.txt', fin_rAOO, delimiter='\t')
np.savetxt(fname+'_rNOO.txt', fin_rNOO, delimiter='\t')
np.savetxt(fname+'_rAOX.txt', fin_rAOX, delimiter='\t')
np.savetxt(fname+'_rO2C.txt', fin_rO2C, delimiter='\t')


#%% plot the outcomes of the experiments

fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)


fig = plt.figure(figsize=(16,9))
gs = GridSpec(3, 3)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])
ax7 = plt.subplot(gs[2,0])
ax8 = plt.subplot(gs[2,1])
ax9 = plt.subplot(gs[2,2])

ax1.set_title('O2 (uM)', fontsize=fslab)
ax2.set_title('NO3 (uM)', fontsize=fslab)
ax3.set_title('NO2 (uM)', fontsize=fslab)
ax4.set_title('NH4 (uM)', fontsize=fslab)
ax5.set_title('N2 production (nM / day)', fontsize=fslab)
ax6.set_title('Anammox contribution (%)', fontsize=fslab)
ax7.set_title('Anaerobic heterotrophy (%)', fontsize=fslab)
ax8.set_title('Nitrite oxidation (nM / day)', fontsize=fslab)
ax9.set_title('Ammonia oxidation (nM / day)', fontsize=fslab)

nh4_n2_AOX = (e_n2AOX*0.5*y_nh4AOX)

p1 = ax1.contourf(xpulse_O2, xpulse_int, fin_O2, levels=np.arange(0,10.1,0.5), extend='max', cmap=colmap)
p2 = ax2.contourf(xpulse_O2, xpulse_int, fin_NO3, levels=np.arange(20,30.1,0.5), extend='both', cmap=colmap)
p3 = ax3.contourf(xpulse_O2, xpulse_int, fin_NO2, levels=np.arange(0,5.1,0.25), extend='max', cmap=colmap)
p4 = ax4.contourf(xpulse_O2, xpulse_int, fin_NH4, levels=np.arange(0,0.51,0.05), extend='max', cmap=colmap)
p5 = ax5.contourf(xpulse_O2, xpulse_int, (fin_rAOX * nh4_n2_AOX + fin_r2Den*0.5)*1e3, levels=np.arange(0,101,5), extend='max', cmap=colmap)
p6 = ax6.contourf(xpulse_O2, xpulse_int, fin_rAOX * nh4_n2_AOX / (fin_rAOX * nh4_n2_AOX + fin_r2Den*0.5) * 100, levels=np.arange(0,101,5), cmap=colmap)
p7 = ax7.contourf(xpulse_O2, xpulse_int, ((fin_rHet - fin_rHetAer)/fin_rHet) * 100, levels=np.arange(0,101,5), cmap=colmap)
p8 = ax8.contourf(xpulse_O2, xpulse_int, fin_rNOO*1e3, levels=np.arange(0,51,5), extend='max', cmap=colmap)
p9 = ax9.contourf(xpulse_O2, xpulse_int, fin_rAOO*1e3, levels=np.arange(0,51,5), extend='max', cmap=colmap)

# delineate import thresholds
col = 'firebrick'; lw = 1.0
ax1.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax2.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax3.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax4.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax5.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax6.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax7.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax8.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)
ax9.contour(xpulse_O2, xpulse_int, fin_O2*1e3, levels=[100], colors='k', linewidths=lw)

# get Babbin et al. 2020 Marine Chemistry data and find combination of Org and O2 supply that explains their data
os.chdir("C://Users/pearseb/Dropbox/PostDoc/my articles/Buchanan & Zakem - aerobic anaerobic competition/data")
babbin = pd.read_csv('Babbin2020.csv')
no2 = babbin['NO2(uM)'].quantile(q=0.25, interpolation='linear')
no2_ox = babbin['NO2_ox'].quantile(q=0.25, interpolation='linear')
n2_prod = babbin['N2_prod'].quantile(q=0.25, interpolation='linear')
per_anammox = babbin['%anammox'].quantile(q=0.25, interpolation='linear')
print(no2, no2_ox, n2_prod, per_anammox)

fin_NO2_babbin = fin_NO2 > no2
fin_NO2ox_babbin = fin_rNOO*1e3 > no2_ox
fin_N2prod_babbin = (fin_rAOX + fin_r2Den)*1e3 > n2_prod
fin_perAnammox_babbin = fin_rAOX / (fin_rAOX + fin_r2Den)*100 > per_anammox
babbin_conditions = fin_NO2_babbin * fin_N2prod_babbin * fin_perAnammox_babbin

c1 = ax1.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c2 = ax2.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c3 = ax3.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c4 = ax4.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c5 = ax5.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c6 = ax6.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c7 = ax7.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c8 = ax8.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)
c9 = ax9.contour(xpulse_O2, xpulse_int, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)

ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax3.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax4.tick_params(labelsize=fstic, labelbottom=False)
ax5.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax6.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax7.tick_params(labelsize=fstic)
ax8.tick_params(labelsize=fstic, labelleft=False)
ax9.tick_params(labelsize=fstic, labelleft=False)

ax1.set_ylabel('pulse frequency (days)', fontsize=fslab)
ax4.set_ylabel('pulse frequency (days)', fontsize=fslab)
ax7.set_ylabel('pulse frequency (days)', fontsize=fslab)
ax7.set_xlabel('O$_2$ pulse ($\mu$M)', fontsize=fslab)
ax8.set_xlabel('O$_2$ pulse ($\mu$M)', fontsize=fslab)
ax9.set_xlabel('O$_2$ pulse ($\mu$M)', fontsize=fslab)

cbar1 = fig.colorbar(p1, ax=ax1)
cbar2 = fig.colorbar(p2, ax=ax2)
cbar3 = fig.colorbar(p3, ax=ax3)
cbar4 = fig.colorbar(p4, ax=ax4)
cbar5 = fig.colorbar(p5, ax=ax5)
cbar6 = fig.colorbar(p6, ax=ax6)
cbar7 = fig.colorbar(p7, ax=ax7)
cbar8 = fig.colorbar(p8, ax=ax8)
cbar9 = fig.colorbar(p9, ax=ax9)


#%%

os.chdir("C://Users/pearseb/Dropbox/PostDoc/my articles/Buchanan & Zakem - aerobic anaerobic competition/figures")
fig.savefig('outcomes_'+fname+'.png', dpi=300)
fig.savefig('transparent/outcomes_'+fname+'.png', dpi=300, transparent=True)




