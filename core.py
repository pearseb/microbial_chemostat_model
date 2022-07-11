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
import netCDF4 as nc

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten

# numerical packages
from tqdm import tqdm 
from numba import jit


#%% print versions of packages

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
Sd0_exp = [0.5]    # uM N  (dissolved; d)
Sp0_exp = [0.0]    # uM N  (particulate; p)

### Oxygen
O20_exp = np.arange(1,10,2)   # uM O2

print("Ratios of Org:O2 suppply")
xx=0
for ii in np.arange(len(Sd0_exp)):
    for jj in np.arange(len(O20_exp)):
        xx += 1
        print("Exp %i ="%(xx), Sd0_exp[ii] / O20_exp[jj])

### model parameters
dil = 0.05  # dilution rate (1/day)
days = 1e4  # number of days to run chemostat
dt = 0.001  # timesteps per day (days)
timesteps = days/dt     # number of timesteps
out_at_day = 10.0       # output results this often (days)
nn_output = days/out_at_day     # number of entries for output


#%% initialise arrays for output

# Nutrients
fin_O2 = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_Sd = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_Sp = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_NO3 = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_NO2 = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_NH4 = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_N2 = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan

# Biomasses
fin_bHet = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_bFac = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_b1Den = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_b2Den = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_b3Den = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_bAOO = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_bNOO = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_bAOX = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan

# track facultative average respiration
fin_facaer = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan

# Rates
fin_rHet = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_rHetAer = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_rO2C = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_r1Den = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_r2Den = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_r3Den = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_rAOO = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_rNOO = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan
fin_rAOX = np.ones((len(Sd0_exp), len(O20_exp))) * np.nan


#%% calculate the diffusion limited O2 uptake for N-based biomass

def po_coef(diam, Qc, C_to_N):
    '''
    Calculates the maximum rate that O2 can be diffused into a cell

    Parameters
    ----------
    diam : Float
        equivalent spherical diatmeter of the microbe (um)
    Qc : Float
        carbon quota of a single cell (mol C / um^3)
    C_to_N : Float
        carbon to nitrogen ratio of the microbes biomass
    
    Returns
    -------
    po_coef : Float
        The maximum rate of O2 diffusion into the cell in units of m^3 / mmol N / day

    '''
    dc = 1.5776 * 1e-5      # cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
    dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day
    Vc = 4/3 * np.pi * (diam/2)**3  # volume of cell (um^3)
    Qn = Qc / C_to_N * Vc           # mol N / cell
    p1 = 4 * np.pi * dc * (diam*1e-6/2.0)   # convert diameter in um to meters because diffusion coefficient in is m^2/day
    pm = p1 / Qn
    po_coef = pm * 1e-3
    return po_coef

print(po_coef(0.5, 0.22*1e-12/12, 5))


#%% estimate yield from heterotrophy using stoichiometric theory and average processing times of the major ecoenzymes (Sinsabaugh et al. 2013 Ecology Letters)

Y_max = 0.6     # maximum possible growth efficiency measured in the field
B_CN = 4.5      # C:N of bacterial biomass (White et al. 2019)
OM_CN = 11.8    # C:N of labile dissolved organic matter (Letscher et al. 2015)
K_CN = 0.5      # C:N half-saturation coefficient (Sinsabaugh & Follstad Shah 2012 - Ecoenzymatic Stoichiometry and Ecological Theory)
EEA_CN = 1.123  # relative rate of enzymatic processing of complex C and complex N molecules into simple precursors for biosynthesis (Sinsabaugh & Follstad Shah 2012)

def yield_from_stoichiometry(Y_max, B_CN, OM_CN, K_CN, EEA_CN):
    '''
    Estimate the yield of bacterial heterotrophy
    
    Parameters
    ----------
    Y_max : Float
        The maximum yield
    B_CN : Float
        The C:N stoichiometric ratio of the biomass produced by heterotrophs
    OM_CN : Float
        The C:N stoichiometric ratio of the organic matter fuelling heterotrophy
    K_CN : Float
        Half-saturation coefficient for the assimialtion of C:N precursor molecules
    EEA_CN : Float
        Eco-Enzymatic Activity (EEA) rate of carbon and nitrogen processing

    Returns
    -------
    Float
        The yield of bacterial heterotrophy (0-->Y_Max) in mol Biomass C per mol Organic C

    '''
    
    S_CN = B_CN / (OM_CN * EEA_CN)
    return Y_max * S_CN / (S_CN + K_CN)

print(yield_from_stoichiometry(Y_max, B_CN, OM_CN, K_CN, EEA_CN))    
print(yield_from_stoichiometry(Y_max, B_CN, np.arange(5,31,1), K_CN, EEA_CN))    


#%% Set traits of the different biomasses

### 1.0 Uptake Parameterisations

# organic matter (uncertain)
VmaxS = 1.0     # mol orgN / mol cell / day
K_s = 0.1        # uM

''' 
Here we set the affinities of different microbes for their limiting resources
     Affinity = Vmax / K
     where Vmax is the maximum uptake rate of resource and K is the half-saturation coefficient
'''
# DIN
VmaxN_1Den = 50.8        # mol DIN / mol cell N / day at 20C
VmaxN_1DenFac = 50.8        # mol DIN / mol cell N / day at 20C
VmaxN_2Den = 50.8        # mol DIN / mol cell N / day at 20C
VmaxN_3Den = 50.8        # mol DIN / mol cell N / day at 20C
K_n_Den = 0.133         # uM for denitrification
VmaxN_AOO = 50.8    # mol DIN / mol cell N / day at 20C for AOA (Zakem et al 2018)
K_n_AOO = 0.10      # uM NH4 for AOA (Martens-Habbena et al. 2009; Horak et al. 2013; Newell et al. 2013)
VmaxN_NOO = 50.8    # mol DIN / mol cell N / day
K_n_NOO = 0.10      # uM NO2 for NOB (Sun et al. 2017)
VmaxNH4_AOX = 50.8    # mol DIN / mol cell N / day
VmaxNO2_AOX = 50.8    # mol DIN / mol cell N / day


### 2.0 yields (y) and products/excretions (e) for biomasses

# denominator coefficients for organic matter half-reactions (see Zakem et al. 2019 ISME) 
# these numbers are equivalent to the number of electrons per reaction 
#   (dO : to catabolise organic matter) dC
#   (dB : to synthesise biomass) dD
dO = 29.1   # assumes constant stoichiometry of all sinking organic matter of C6.6-H10.9-O2.6-N using equation: d = 4C + H - 2O -3N
dB = 20.0   # assumes constant stoichiometry of all heterotrophic bacteria of C5-H7-O2-N using equation: d = 4C + H - 2O -3N

# aerobic heterotrophy 
y_oHet = 0.14       # Organic matter yield for aerobic heterotrophy (mol B / mol organic N) - comes from Robinson et al. 2009
f_oHet = y_oHet * dB/dO  # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oO2 = (f_oHet/dB) / ((1.0-f_oHet)/4.0)  # yield of biomass per unit oxygen reduced

# nitrate reduction (NO3 --> NO2)
y_n1Den = y_oHet * 0.9          # we asssume that the yield of anaerobic respiration using NO3 is 90% of aerobic respiration
f_n1Den = y_n1Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n1NO3 = (f_n1Den/dB) / ((1.0-f_n1Den)/2.0) # yield of biomass per unit nitrate reduced (denominator of 2 because N is reduced by 2 electrons)
e_n1Den = 1.0 / y_n1NO3         # mols NO2 produced per mol biomass synthesised

# nitrite reduction (NO2 --> N2)
y_n2Den = y_oHet * 0.9          # we asssume that the yield of anaerobic respiration using NO2 is 90% of aerobic respiration
f_n2Den = y_n2Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n2NO2 = (f_n2Den/dB) / ((1.0-f_n2Den)/3.0) # yield of biomass per unit nitrite reduced
e_n2Den = 1.0 / y_n2NO2         # mols N2 produced per mol biomass synthesised

# Full denitrification (NO3 --> N2)
y_n3Den = y_oHet * 0.9          # we asssume that the yield of anaerobic respiration using NO2 is 90% of aerobic respiration
f_n3Den = y_n3Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n3NO3 = (f_n3Den/dB) / ((1.0-f_n3Den)/5.0) # yield of biomass per unit nitrate reduced
e_n3Den = 1.0 / y_n3NO3         # mols N2 produced per mol biomass synthesised

# Facultative heterotrophy (oxygen and nitrate reduction)
fac_penalty = 0.8
y_oHetFac = y_oHet * fac_penalty
f_oHetFac = y_oHetFac * dB/dO         # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oO2Fac = (f_oHetFac/dB) / ((1.0-f_oHetFac)/4.0)  # yield of biomass per unit oxygen reduced
y_n1DenFac = y_n1Den * fac_penalty
f_n1DenFac = y_n1DenFac * dB/dO
y_n1NO3Fac = (f_n1DenFac/dB) / ((1.0-f_n1DenFac)/2.0)

# Chemoautotrophic ammonia oxidation (NH3 --> NO2)
y_nAOO = 1/112.                 # mol N biomass per mol NH4 (Zakem et al. 2018)
y_oAOO = 1/162.                 # mol N biomass per mol O2 (Zakem et al. 2018)

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_nNOO = 1/334.                 # mol N biomass per mol NO2 (Zakem et al. 2018)
y_oNOO = 1/162.                 # mol N biomass per mol O2 (Zakem et al. 2018)

# Chemoautotrophic anammox (NH4 + NO2 --> NO3 + N2)
y_nh4AOX = 1/154.                # mol N biomass per mol NH4 (Zakem et al. 2019 ISME)
y_no2AOX = 1/216.                # mol N biomass per mol NO2 (Zakem et al. 2019 ISME)
e_n2AOX = 163.*2                 # mol N (as N2) formed per mol biomass N synthesised
e_no3AOX = 43.                   # mol NO3 formed per mol biomass N synthesised

# diffusive coefficient for oxygen
po_aer = po_coef(0.5, 0.22*1e-12/12, 5)
po_aoo = po_aer
po_noo = po_aer

### Check useages of N in reactions
print("")
print("moles OrgN used in aerobic heterotrophy =",1/y_oHet)
print("moles OrgN used in facultative aerobic heterotrophy =",1/y_oHetFac)
print("moles OrgN used in facultative denitrification =",1/y_n1DenFac)
print("moles OrgN used in denitrification (NO3 --> NO2) =",1/y_n1Den)
print("moles OrgN used in denitrification (NO2 --> N2) =",1/y_n2Den)
print("moles OrgN used in denitrification (NO3 --> N2) =",1/y_n3Den)
print("moles NH4 used in ammonia oxidation =",1/y_nAOO)
print("moles NO2 used in nitrite oxidation =",1/y_nNOO)
print("moles NH4+NO2 used in anammox =",(1/y_nh4AOX + 1/y_no2AOX), "and produced as Biomass, NO3 and N2 in anammox =",(e_no3AOX + e_n2AOX + 1))

### Check usages of oxygen in reactions
print("")
print("moles O2 used in aerobic heterotrophy =",1/y_oO2)
print("moles O2 used in facultative aerobic heterotrophy =",1/y_oO2Fac)
print("moles O2 used in ammonia oxidation =",1/y_oAOO)
print("moles O2 used in nitrite oxidation =",1/y_oNOO)


#%% Updated traits

dO = 29.1   # assumes constant stoichiometry of all sinking organic matter of C6.6-H10.9-O2.6-N using equation: d = 4C + H - 2O -3N
dB = 18.0   # assumes constant stoichiometry of all heterotrophic bacteria of C4.5-H7-O2-N using equation: d = 4C + H - 2O -3N


Y_max = 0.6     # maximum possible growth efficiency measured in the field
B_CN = 4.5      # C:N of bacterial biomass (White et al. 2019)
OM_CN = 6.6    # C:N of labile dissolved organic matter (Letscher et al. 2015)
K_CN = 0.5      # C:N half-saturation coefficient (Sinsabaugh & Follstad Shah 2012 - Ecoenzymatic Stoichiometry and Ecological Theory)
EEA_CN = 1.123  # relative rate of enzymatic processing of complex C and complex N molecules into simple precursors for biosynthesis (Sinsabaugh & Follstad Shah 2012)

# aerobic heterotrophy 
y_oHet = yield_from_stoichiometry(Y_max, B_CN, OM_CN, K_CN, EEA_CN) / B_CN * OM_CN  # OM_CN / B_CN converts to mol BioN per mol OrgN 
f_oHet = y_oHet * dB/dO  # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oO2 = (f_oHet/dB) / ((1.0-f_oHet)/4.0)  # yield of biomass per unit oxygen reduced
mumax_Het = 0.5     # Rappé et al., 2002
VmaxS = mumax_Het / y_oHet  # mol Org N (mol BioN)-1 per day

den_penalty = 0.9

# nitrate reduction (NO3 --> NO2)
y_n1Den = y_oHet * den_penalty          # we asssume that the yield of anaerobic respiration using NO3 is 90% of aerobic respiration
f_n1Den = y_n1Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n1NO3 = (f_n1Den/dB) / ((1.0-f_n1Den)/2.0) # yield of biomass per unit nitrate reduced (denominator of 2 because N is reduced by 2 electrons)
e_n1Den = 1.0 / y_n1NO3         # mols NO2 produced per mol biomass synthesised
VmaxN_1Den = mumax_Het * den_penalty / y_n1NO3        # mol DIN / mol cell N / day at 20C
K_n_Den = 4.0                     # 4 – 25 µM NO2 for denitrifiers (Almeida et al. 1995)

# nitrite reduction (NO2 --> N2)
y_n2Den = y_oHet * den_penalty          # we asssume that the yield of anaerobic respiration using NO2 is 90% of aerobic respiration
f_n2Den = y_n2Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n2NO2 = (f_n2Den/dB) / ((1.0-f_n2Den)/3.0) # yield of biomass per unit nitrite reduced
e_n2Den = 1.0 / y_n2NO2         # mols N2 produced per mol biomass synthesised
VmaxN_2Den = mumax_Het * den_penalty / y_n2NO2        # mol DIN / mol cell N / day at 20C

# Full denitrification (NO3 --> N2)
y_n3Den = y_oHet * den_penalty          # we asssume that the yield of anaerobic respiration using NO2 is 90% of aerobic respiration
f_n3Den = y_n3Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n3NO3 = (f_n3Den/dB) / ((1.0-f_n3Den)/5.0) # yield of biomass per unit nitrate reduced
e_n3Den = 1.0 / y_n3NO3         # mols N2 produced per mol biomass synthesised
VmaxN_3Den = mumax_Het * den_penalty / y_n3NO3        # mol DIN / mol cell N / day at 20C

# Facultative heterotrophy (oxygen and nitrate reduction)
fac_penalty = 0.8
y_oHetFac = y_oHet * fac_penalty
f_oHetFac = y_oHetFac * dB/dO         # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oO2Fac = (f_oHetFac/dB) / ((1.0-f_oHetFac)/4.0)  # yield of biomass per unit oxygen reduced
y_n1DenFac = y_n1Den * fac_penalty
f_n1DenFac = y_n1DenFac * dB/dO
y_n1NO3Fac = (f_n1DenFac/dB) / ((1.0-f_n1DenFac)/2.0)
VmaxN_1DenFac = mumax_Het * fac_penalty / y_n1NO3Fac        # mol DIN / mol cell N / day at 20C

# Chemoautotrophic ammonia oxidation (NH3 --> NO2)
y_nAOO = 0.0245         # mol N biomass per mol NH4 (Bayer et al. 2022; Zakem et al. 2022)
d_AOO = 4*4 + 7 - 2*2 -3*1  # number of electrons produced
f_AOO = y_nAOO / (6*(1/d_AOO - y_nAOO/d_AOO))         # fraction of electrons going to biomass synthesis from electron donor (NH4) (Zakem et al. 2022)
y_oAOO = f_AOO/d_AOO / ((1-f_AOO)/4.0)                # mol N biomass per mol O2 (Bayer et al. 2022; Zakem et al. 2022)
mumax_AOO = 0.8            # per day (Qin et al. 2015)
VmaxN_AOO = mumax_AOO / y_nAOO

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_nNOO = 0.0126         # mol N biomass per mol NO2 (Bayer et al. 2022)
d_NOO = 3.4*4 + 7 - 2*2 - 3*1
f_NOO = (y_nNOO * d_NOO) /2          # fraction of electrons going to biomass synthesis from electron donor (NO2) (Zakem et al. 2022)
y_oNOO = 4*f_NOO*(1-f_NOO)/d_NOO         # mol N biomass per mol O2 (Bayer et al. 2022)
mumax_NOO = 0.96        # per day (Spieck et al. 2014)
VmaxN_NOO = mumax_NOO / y_nNOO

# Chemoautotrophic anammox (NH4 + NO2 --> NO3 + N2)
y_nh4AOX = 1./70.0                  # mol N biomass per mol NH4 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
y_no2AOX = 1./81.0                  # mol N biomass per mol NO2 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
e_n2AOX = 139                       # mol N (as N2) formed per mol biomass N synthesised ***Rounded to nearest whole number
e_no3AOX = 11                       # mol NO3 formed per mol biomass N synthesised ***Rounded to nearest whole number
mumax_AOX = 0.17                    # Okabe et al. 2021 ISME
VmaxNH4_AOX = mumax_AOX / y_nh4AOX
VmaxNO2_AOX = mumax_AOX / y_no2AOX


### Check useages of N in reactions
print("")
print("moles OrgN used in aerobic heterotrophy =",1/y_oHet)
print("moles OrgN used in facultative aerobic heterotrophy =",1/y_oHetFac)
print("moles OrgN used in facultative denitrification =",1/y_n1DenFac)
print("moles OrgN used in denitrification (NO3 --> NO2) =",1/y_n1Den)
print("moles OrgN used in denitrification (NO2 --> N2) =",1/y_n2Den)
print("moles OrgN used in denitrification (NO3 --> N2) =",1/y_n3Den)
print("moles NH4 used in ammonia oxidation =",1/y_nAOO)
print("moles NO2 used in nitrite oxidation =",1/y_nNOO)
print("moles NH4+NO2 used in anammox =",(1/y_nh4AOX + 1/y_no2AOX), "and produced as Biomass, NO3 and N2 in anammox =",(e_no3AOX + e_n2AOX + 1))

### Check usages of oxygen in reactions
print("")
print("moles O2 used in aerobic heterotrophy =",1/y_oO2)
print("moles O2 used in facultative aerobic heterotrophy =",1/y_oO2Fac)
print("moles O2 used in ammonia oxidation =",1/y_oAOO)
print("moles O2 used in nitrite oxidation =",1/y_oNOO)


#%% calculate R*-stars for all microbes

def o2_star(loss, Qc, diam, dc, y_oO2):
    '''
    Calculates the subsistence O2 concentration required to sustain biomass at equilibrium

    Parameters
    ----------
    loss : Float
        rate at which biomass is lost (1 / day)
    Qc : TYPE
        carbon quota of a single cell (mol C / um^3)
    diam : Float
        equivalent spherical diatmeter of the microbe (um)
    dc : Float
        rate of diffusive oxygen supply (m^2 / day)
    y_oO2 : TYPE
        biomass yield per O2 molecule reduced (mol N / mol O2)

    Returns
    -------
    Float
        Subsistence O2 concentration

    '''
    return (loss * Qc * (diam/2)**2) / (3 * dc * 1e-12 * y_oO2)

def R_star(loss, K_r, Vmax, y_r):
    '''
    

    Parameters
    ----------
    loss : TYPE
        rate of loss of biomass
    K_r : TYPE
        half-saturation coefficient for uptake
    Vmax : TYPE
        Maximum uptake rate of molecule
    y_r : TYPE
        Yield of biomass creation (mol BioN per mol compound)

    Returns
    -------
    TYPE
        R* = the subsistence concentration of the compound in question required to sustain biomass

    '''
    return (loss * K_r) / (Vmax * y_r - loss)


# loss rates (day-1)
loss = dil  # assume loss rates the same for all microbes?

# cell volumes (um^3)
vol_aer = 0.05  # based on SAR11 (Giovannoni 2017 Ann. Rev. Marine Science)
vol_den = vol_aer
vol_aoo = np.pi * (0.2*0.5)**2 * 0.8    # [rods] N. maritimus SCM1 in Table S1 from Hatzenpichler 2012 App. Envir. Microbiology
vol_noo = np.pi * (0.3*0.5)**2 * 3      # [rods] based on average cell diameter and length of Nitrospina gracilis (Spieck et al. 2014 Sys. Appl. Microbiology)
vol_aox = 4/3 * np.pi * (0.8*0.5)**3    # [cocci] diameter of 0.8 um based on Candidatus Scalindua (Wu et al. 2020 Water Science and Tech.)
vol_moa = 4/3 * np.pi * (2*0.5)**3      # [cocci] diameter of 1-3 um based on Candidatus Methanoperedens (Haroon et al. 2013 Nature)
vol_mob = np.pi * (0.375*0.5)**2 * 1    # [rods] diameter 0.375 um and length 1 um based on Candidatus Methylomirabilis oxyfera (Ettwig et al. 2010 Nature)

# cell sizes (um), where vol = 4/3 * pi * r^3
diam_aer = ( 3 * vol_aer / (4 * np.pi) )**(1./3)*2
diam_den = ( 3 * vol_den / (4 * np.pi) )**(1./3)*2
diam_aoo = ( 3 * vol_aoo / (4 * np.pi) )**(1./3)*2
diam_noo = ( 3 * vol_noo / (4 * np.pi) )**(1./3)*2
diam_aox = ( 3 * vol_aox / (4 * np.pi) )**(1./3)*2
diam_moa = ( 3 * vol_moa / (4 * np.pi) )**(1./3)*2
diam_mob = ( 3 * vol_mob / (4 * np.pi) )**(1./3)*2

# cellular C:N of microbes
CN_aer = 4.5    # White et al. 2019
CN_den = CN_aer
CN_aoo = 4.0    # Bayer et al. 2022
CN_noo = 3.4    # Bayer et al. 2022
CN_aox = 5.0    # Lotti et al. 2014

# g carbon per cell (assumes 0.1 g DW/WW for all microbial types as per communication with Marc Strous)
Ccell_aer = 0.1 * (12*CN_aer / (12*CN_aer + 7 + 16*2 + 14)) / (1e12 / vol_aer)
Ccell_den = 0.1 * (12*CN_den / (12*CN_den + 7 + 16*2 + 14)) / (1e12 / vol_den)
Ccell_aoo = 0.1 * (12*CN_aoo / (12*CN_aoo + 7 + 16*2 + 14)) / (1e12 / vol_aoo)
Ccell_noo = 0.1 * (12*CN_noo / (12*CN_noo + 7 + 16*2 + 14)) / (1e12 / vol_noo)
Ccell_aox = 0.1 * (12*CN_aox / (12*CN_aox + 8.7 + 16*1.55 + 14)) / (1e12 / vol_aox)

# cell quotas (mol C / um^3)
Qc_aer = Ccell_aer / vol_aer / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_den = Ccell_den / vol_den / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_aoo = Ccell_aoo / vol_aoo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_noo = Ccell_noo / vol_noo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_aox = Ccell_aox / vol_aox / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019

# diffusive oxygen coefficient
dc = 1.5776 * 1e-5      # cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day
po_aer = po_coef(diam_aer, Qc_aer, CN_aer)
po_den = po_coef(diam_den, Qc_den, CN_den)
po_aoo = po_coef(diam_aoo, Qc_aoo, CN_aoo)
po_noo = po_coef(diam_noo, Qc_noo, CN_noo)
po_aox = po_coef(diam_aox, Qc_aox, CN_aox)

# K_r values
K_s
K_n_AOO = 0.1   # Martens-Habbena et al. 2009 Nature
K_n_NOO = 0.1   # Reported from OMZ (Sun et al. 2017) and oligotrophic conditions (Zhang et al. 2020)
K_nh4_AOX = 0.45    # Awata et al. 2013 for Scalindua
K_no2_AOX = 0.45   # Awata et al. 2013 for Scalindua

# estimate R* and O2* for the different groups to predict which kinds of microbes 
# will be competitive when certain resources are limiting

print("Oxygen")
print("Subsistence O2 to support bulk aerobic heterotrophy =", o2_star(loss, Qc_aer, diam_aer, dc, y_oO2*CN_aer))
print("Subsistence O2 to support bulk aerobic (facultative) heterotrophy =", o2_star(loss, Qc_aer, diam_aer, dc, y_oO2Fac*CN_aer))
print("Subsistence O2 to support bulk aerobic ammonia oxidation =", o2_star(loss, Qc_aoo, diam_aoo, dc, y_oAOO*CN_aoo))
print("Subsistence O2 to support bulk aerobic nitrite oxidation =", o2_star(loss, Qc_noo, diam_noo, dc, y_oNOO*CN_noo))

print("Organic matter")
print("Subsistence Norg to support bulk aerobic hetertrophy =", R_star(loss, K_s, VmaxS, y_oHet))
print("Subsistence Norg to support bulk aerobic (facultative) hetertrophy =", R_star(loss, K_s, VmaxS, y_oHetFac))
print("Subsistence Norg to support bulk anaerobic nitrate reducing hetertrophy =", R_star(loss, K_s, VmaxS, y_n1Den))
print("Subsistence Norg to support bulk anaerobic nitrite reducing hetertrophy =", R_star(loss, K_s, VmaxS, y_n2Den))

print("Ammonium")
print("Subsistence NH4 to support bulk aerobic ammonia oxidation =", R_star(loss, K_n_AOO, VmaxN_AOO, y_nAOO))
print("Subsistence NH4 to support bulk anaerobic ammonium oxidation (anammox) =", R_star(loss, K_nh4_AOX, VmaxNH4_AOX, y_nh4AOX))

print("Nitrite")
print("Subsistence NO2 to support bulk denitrification (NO2 --> N2) =", R_star(loss, K_n_Den, VmaxN_2Den, y_n2NO2))
print("Subsistence NO2 to support bulk aerobic nitrite oxidation =", R_star(loss, K_n_NOO, VmaxN_NOO, y_nNOO))
print("Subsistence NO2 to support bulk anaerobic ammonium oxidation (anammox) =", R_star(loss, K_no2_AOX, VmaxNO2_AOX, y_no2AOX))

print("Nitrate")
print("Subsistence NO3 to support bulk nitrate reduction by obligate anaerobe =", R_star(loss, K_n_Den, VmaxN_1Den, y_n1NO3))
print("Subsistence NO3 to support bulk nitrate reduction by facultative anaerobe =", R_star(loss, K_n_Den, VmaxN_1DenFac, y_n1NO3Fac))


#%% define the model

@jit(nopython=True)
def OMZredox(timesteps, nn_output, dt, dil, out_at_day, pulse_Sd, pulse_bHet, pulse_bFac, pulse_O2, pulse_int, \
             po_aer, po_aoo, po_noo, VmaxS, K_s, VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_1DenFac, K_n_Den, VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, VmaxNH4_AOX, VmaxNO2_AOX,\
             y_oHet, y_oO2, y_oHetFac, y_oO2Fac, y_n1DenFac, y_n1NO3Fac, y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
             in_Sd, in_Sp, in_O2, in_NO3, in_NO2, in_NH4, in_bHet, in_bFac, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX):
    '''
    ______________________________________________________
    0D Chemostat model for aerobic and anaerobic processes
    ------------------------------------------------------
    
    INPUTS
    ------
    
        timesteps   :   number of timesteps in run
        nn_output   :   number of times we save output
        dt          :   number of timesteps per day
        dil         :   dilution rate of chemostat
        out_at_day  :   day interval to record output
        pulse_OM    :   instant injection of OM into the model (mmol)
        pulse_O2    :   instant injection of O2 into the model (mmol)
        pulse_int   :   interval of pulsing (days)
                        
        
        po_coef     :   diffusion-limited O2 uptake (m^3 / mmol N / day)
        VmaxS       :   maximum uptake rate of organic matter (mol orgN / mol N-cell / day )
        K_s         :   half saturation coefficient for organic matter uptake
        VmaxN_1Den  :   maximum uptake rate of NO3 by nitrate reducing bacteria (mol NO3 / mol N-cell / day )
        VmaxN_2Den  :   maximum uptake rate of NO2 by nitrite reducing bacteria (mol NO3 / mol N-cell / day )
        VmaxN_3Den  :   maximum uptake rate of NO3 by fully denitrifying bacteria (mol NO3 / mol N-cell / day )
        VmaxN_1DenFac : maximum uptake rate of NO3 by nitrate reducing facultative bacteria (mol NO3 / mol N-cell / day )
        K_n_Den     :   half saturation coefficient for DIN uptake by heterotrophic bacteria
        VmaxN_AOO   :   maximum uptake rate of NH4 by ammonia oxidisers (mol NH4 / mol N-cell / day )
        K_n_AOO     :   half saturation coefficient for NH4 uptake by ammonia oxidisers
        VmaxN_NOO   :   maximum uptake rate of NO2 by nitrite oxidisers (mol NO2 / mol N-cell / day )
        K_n_NOO     :   half saturation coefficient for NO2 uptake by nitrite oxidisers
        VmaxNH4_AOX :   maximum uptake rate of NH4 by anammox bacteria (mol NH4 / mol N-cell / day )
        VmaxNO2_AOX :   maximum uptake rate of NO2 by anammox bacteria (mol NO2 / mol N-cell / day )
        K_nh4_AOX   :   half saturation coefficient for NH4 uptake by anaerobic ammonium oxidisers
        K_no2_AOX   :   half saturation coefficient for NO2 uptake by anaerobic ammonium oxidisers
                
        y_oHet      :   mol N-biomass / mol N-organics consumed
        y_oO2       :   mol N-biomass / mol O2 reduced
        y_oHetFac   :   mol N-biomass / mol N-organics consumed
        y_oO2Fac    :   mol N-biomass / mol O2 reduced
        y_n1DenFac  :   mol N-biomass / mol N-organics consumed
        y_n1NO3Fac  :   mol N-biomass / mol NO3 reduced
        y_n1Den     :   mol N-biomass / mol N-organics consumed
        y_n1NO3     :   mol N-biomass / mol NO3 reduced (NO3 --> NO2)
        y_n2Den     :   mol N-biomass / mol N-organics consumed
        y_n2NO2     :   mol N-biomass / mol NO2 reduced (NO2 --> N2)
        y_n3Den     :   mol N-biomass / mol N-organics consumed
        y_n3NO3     :   mol N-biomass / mol NO3 reduced (NO3 --> N2)
        y_nAOO      :   mol N-biomass / mol NH4 oxidised
        y_oAOO      :   mol N-biomass / mol O2 reduced
        y_nNOO      :   mol N-biomass / mol NO2 oxidised
        y_oNOO      :   mol N-biomass / mol O2 reduced
        y_nh4AOX    :   mol N-biomass / mol NH4 oxidised
        y_no2AOX    :   mol N-biomass / mol NO2 reduced
                
        in_Sd       :   initial Sd concentration
        in_Sp       :   initial Sp concentration
        in_O2       :   initial O2 concentration
        in_NO3      :   initial NO3 concentration
        in_NO2      :   initial NO2 concentration
        in_NH4      :   initial NH4 concentration
        in_bHet     :   initial Het concentration of biomass
        in_bFac     :   initial Fac concentration of biomass
        in_b1Den    :   initial 1Den concentration of biomass
        in_b2Den    :   initial 2Den concentration of biomass
        in_b3Den    :   initial 3Den concentration of biomass
        in_bAOO     :   initial AOO concentration of biomass
        in_bNOO     :   initial NOO concentration of biomass
        in_bAOX     :   initial AOX concentration of biomass
        
        
    OUTPUTS
    -------
    
        m_Sd       :   modelled Sd concentration
        m_Sp       :   modelled Sp concentration
        m_O2       :   modelled O2 concentration
        m_NO3      :   modelled NO3 concentration
        m_NO2      :   modelled NO2 concentration
        m_NH4      :   modelled NH4 concentration
        m_bHet     :   modelled Het concentration of biomass
        m_bFac     :   modelled Fac concentration of biomass
        m_b1Den    :   modelled 1Den concentration of biomass
        m_b2Den    :   modelled 2Den concentration of biomass
        m_b3Den    :   modelled 3Den concentration of biomass
        m_bAOO     :   modelled AOO concentration of biomass
        m_bNOO     :   modelled NOO concentration of biomass
        m_bAOX     :   modelled AOX concentration of biomass
    '''
    
    # transfer initial inputs to model variables
    m_Sd = in_Sd
    m_Sp = in_Sp
    m_O2 = in_O2
    m_NO3 = in_NO3
    m_NO2 = in_NO2
    m_NH4 = in_NH4
    m_N2 = 0.0
    m_bHet = in_bHet
    m_bFac = in_bFac
    m_b1Den = in_b1Den
    m_b2Den = in_b2Den
    m_b3Den = in_b3Den
    m_bAOO = in_bAOO
    m_bNOO = in_bNOO
    m_bAOX = in_bAOX
    
    # set the output arrays 
    out_Sd = np.ones((int(nn_output)+1)) * np.nan
    out_Sp = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NO3 = np.ones((int(nn_output)+1)) * np.nan
    out_NO2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_N2 = np.ones((int(nn_output)+1)) * np.nan
    out_bHet = np.ones((int(nn_output)+1)) * np.nan
    out_bFac = np.ones((int(nn_output)+1)) * np.nan
    out_b1Den = np.ones((int(nn_output)+1)) * np.nan
    out_b2Den = np.ones((int(nn_output)+1)) * np.nan
    out_b3Den = np.ones((int(nn_output)+1)) * np.nan
    out_bAOO = np.ones((int(nn_output)+1)) * np.nan
    out_bNOO = np.ones((int(nn_output)+1)) * np.nan
    out_bAOX = np.ones((int(nn_output)+1)) * np.nan
    out_facaer = np.ones((int(nn_output)+1)) * np.nan
    out_rHet = np.ones((int(nn_output)+1)) * np.nan
    out_rHetAer = np.ones((int(nn_output)+1)) * np.nan
    out_rO2C = np.ones((int(nn_output)+1)) * np.nan
    out_r1Den = np.ones((int(nn_output)+1)) * np.nan
    out_r2Den = np.ones((int(nn_output)+1)) * np.nan
    out_r3Den = np.ones((int(nn_output)+1)) * np.nan
    out_rAOO = np.ones((int(nn_output)+1)) * np.nan
    out_rNOO = np.ones((int(nn_output)+1)) * np.nan
    out_rAOX = np.ones((int(nn_output)+1)) * np.nan
    
    # set the array for recording average activity of facultative anaerobes
    interval = int((1/dt * out_at_day))
    facaer = np.ones((interval)) * np.nan
    
    # record the initial conditions
    i = 0
    out_Sd[i] = m_Sd 
    out_Sp[i] = m_Sp
    out_O2[i] = m_O2
    out_NO3[i] = m_NO3
    out_NO2[i] = m_NO2 
    out_NH4[i] = m_NH4 
    out_N2[i] = m_N2
    out_bHet[i] = m_bHet
    out_bFac[i] = m_bFac
    out_b1Den[i] = m_b1Den
    out_b2Den[i] = m_b2Den
    out_b3Den[i] = m_b3Den 
    out_bAOO[i] = m_bAOO
    out_bNOO[i] = m_bNOO
    out_bAOX[i] = m_bAOX
    
    # begin the loop
    for t in np.arange(1,timesteps+1,1):
        
        # uptake rates (p)
        p_O2_aer = po_aer * m_O2                            # mol O2 / day
        p_O2_aoo = po_aoo * m_O2                            # mol O2 / day
        p_O2_noo = po_noo * m_O2                            # mol O2 / day
        p_Sp = VmaxS * m_Sp / (K_s + m_Sp)                  # mol Org / day
        p_Sd = VmaxS * m_Sd / (K_s + m_Sd)                  # mol Org / day
        p_FacNO3 = VmaxN_1DenFac * m_NO3 / (K_n_Den + m_NO3)             # mol NO3 / day
        p_1DenNO3 = VmaxN_1Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_2DenNO2 = VmaxN_2Den * m_NO2 / (K_n_Den + m_NO2)               # mol NO2 / day
        p_3DenNO3 = VmaxN_3Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_NH4_AOO = VmaxN_AOO * m_NH4 / (K_n_AOO + m_NH4)   # mol NH4 / day
        p_NO2_NOO = VmaxN_NOO * m_NO2 / (K_n_NOO + m_NO2)   # mol NO2 / day
        p_NH4_AOX = VmaxNH4_AOX * m_NH4 / (K_nh4_AOX + m_NH4)     # mol NH4 / day
        p_NO2_AOX = VmaxNO2_AOX * m_NO2 / (K_no2_AOX + m_NO2)     # mol NO2 / day
        
        # growth rates (u)
        u_Het = np.fmax(0.0, np.fmin(p_Sd * y_oHet, p_O2_aer * y_oO2))          # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
        u_FacO2 = np.fmax(0.0, np.fmin(p_Sd * y_oHetFac, p_O2_aer * y_oO2Fac))  # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
        u_FacNO3 = np.fmax(0.0, np.fmin(p_Sd * y_n1DenFac, p_FacNO3 * y_n1NO3Fac)) # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_1Den = np.fmax(0.0, np.fmin(p_Sd * y_n1Den, p_1DenNO3 * y_n1NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_2Den = np.fmax(0.0, np.fmin(p_Sd * y_n2Den, p_2DenNO2 * y_n2NO2))         # mol Org / day * mol Biomass / mol Org || mol NO2 / day * mol Biomass / mol NO2
        u_3Den = np.fmax(0.0, np.fmin(p_Sd * y_n3Den, p_3DenNO3 * y_n3NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_AOO = np.fmax(0.0, np.fmin(p_NH4_AOO * y_nAOO, p_O2_aoo * y_oAOO))    # mol NH4 / day * mol Biomass / mol NH4 || mol O2 / day * mol Biomass / mol O2
        u_NOO = np.fmax(0.0, np.fmin(p_NO2_NOO * y_nNOO, p_O2_noo * y_oNOO))    # mol NO2 / day * mol Biomass / mol NO2 || mol O2 / day * mol Biomass / mol O2
        u_AOX = np.fmax(0.0, np.fmin(p_NO2_AOX * y_no2AOX, p_NH4_AOX * y_nh4AOX)) # mol NO2 / day * mol Biomass / mol NO2 || mol NH4 / day * mol Biomass / mol NH4
        
        ### BIOMASS TYPES & THEIR STOICHIOMETRIES (already encoded within yields, which are inputs)
        # 1. Aerobic heterotrophy ( 7.1-OM + 47-O2 --> B + 42-CO2 + 6.1-NH4)
        # 2. Facultative anaerobes ( 8.9-OM + 60-O2 --> B + 53-CO2 + 7.9-NH4)
        # 3. Denitrification ( 7.9-OM + 105-NO3 --> B + 47-CO2 + 6.9-NH4 + 105-NO2)
        # 4. Denitrification ( 7.9-OM + 70-NO2 --> B + 47-CO2 + 6.9-NH4 + 35-N2)
        # 5. Denitrification ( 7.9-OM + 42-NO3 --> B + 47-CO2 + 6.9-NH4 + 21-N2)
        # 6. Ammonia oxidation ( 112-NH4 + 5-CO2 + 162-O2 --> B + 111-NO2)
        # 7. Nitrite oxidation ( 334-NO2 + 5-CO2 + 162-O2 --> B + 334-NO3)
        # 8. Anammox ( 154-NH4 + 216-NO2 + 5-CO2 --> B + 42-NO3 + 163-N2)
        
        ###
        ### Change in state variables per day (ddt)
        ###
        
        # deal with facultative bacteria first
        if u_FacO2 >= u_FacNO3:
            u_Fac = u_FacO2
            ddt_Sd_Fac = u_Fac * m_bFac / y_oHetFac
            ddt_O2_Fac = u_Fac * m_bFac / y_oO2Fac
            ddt_NO3_Fac = 0.0
            ddt_NH4_Fac = u_Fac * m_bFac * (1./y_oHetFac - 1)
            facaer[int(t % interval)] = 1.0
        else:
            u_Fac = u_FacNO3
            ddt_Sd_Fac = u_Fac * m_bFac / y_n1DenFac
            ddt_O2_Fac = 0.0
            ddt_NO3_Fac = u_Fac * m_bFac / y_n1NO3Fac
            ddt_NH4_Fac = u_Fac * m_bFac * (1./y_n1DenFac - 1)
            facaer[int(t % interval)] = 0.0
            
        
        ### rates
        aer_heterotrophy = u_Het * m_bHet / y_oHet      \
                           + ddt_Sd_Fac
        heterotrophy = u_Het * m_bHet / y_oHet      \
                       + ddt_Sd_Fac                 \
                       + u_1Den * m_b1Den / y_n1Den \
                       + u_2Den * m_b2Den / y_n2Den \
                       + u_3Den * m_b3Den / y_n3Den       
        oxy_consumption = u_Het * m_bHet / y_oO2    \
                          + ddt_O2_Fac              \
                          + u_AOO * m_bAOO / y_oAOO \
                          + u_NOO * m_bNOO / y_oNOO
        den_nar = u_1Den * m_b1Den / y_n1NO3        \
                  + ddt_NO3_Fac
        den_nir = u_2Den * m_b2Den / y_n2NO2
        den_full = u_3Den * m_b3Den / y_n3NO3
        ammonia_ox = u_AOO * m_bAOO / y_nAOO
        nitrite_ox = u_NOO * m_bNOO / y_nNOO
        anammox_nh4 = u_AOX * m_bAOX / y_nh4AOX
        anammox_no2 = u_AOX * m_bAOX / y_no2AOX
        anammox_no3 = u_AOX * m_bAOX * e_no3AOX
        
        # Dissolved organic matter (consumed by 1, 2, 3, 4, 5)
        ddt_Sd = dil * (in_Sd - m_Sd) - heterotrophy
                 
        # Particulate organic matter (consumed by 1, 2, 3, 4, 5)
        ddt_Sp = dil * (in_Sp - m_Sp)   # particle associated POM not currently included in model
        
        # Dissolved oxygen (consumed by 1, 2, 6, 7)
        ddt_O2 = dil * (in_O2 - m_O2) - oxy_consumption
                 
        # Nitrate (consumed by 2, 3, 5) (produced by 7, 8)
        ddt_NO3 = dil * (in_NO3 - m_NO3)        \
                 - den_nar                      \
                 - den_full                     \
                 + nitrite_ox                   \
                 + anammox_no3        
        
        # Nitrite (consumed by 4, 7, 8) (produced by 2, 3, 6)
        ddt_NO2 = dil * (in_NO2 - m_NO2)        \
                 - den_nir                      \
                 - nitrite_ox                   \
                 - anammox_no2                  \
                 + den_nar                      \
                 + u_AOO * m_bAOO * (1./y_nAOO - 1)   # because it uses one mol of NH4 for biomass synthesis         
        
        # Ammonium (consumed by 6, 7, 8) (produced by 1, 2, 3, 4, 5)
        ddt_NH4 = dil * (in_NH4 - m_NH4)        \
                 - ammonia_ox                   \
                 - anammox_nh4                  \
                 - u_NOO * m_bNOO               \
                 + u_Het * m_bHet * (1./y_oHet - 1)    \
                 + ddt_NH4_Fac                         \
                 + u_1Den * m_b1Den * (1./y_n1Den - 1) \
                 + u_2Den * m_b2Den * (1./y_n2Den - 1) \
                 + u_3Den * m_b3Den * (1./y_n3Den - 1)    
        
        # Dinitrogen gas (produced by 3, 4, 8)
        ddt_N2 = dil * (-m_N2)                  \
                 + u_2Den * m_b2Den * e_n2Den   \
                 + u_3Den * m_b3Den * e_n3Den   \
                 + u_AOX * m_bAOX * e_n2AOX
        
        # Biomass of aerobic heterotrophs
        ddt_bHet = dil * (-m_bHet)              \
                   + u_Het * m_bHet 
        
        # Biomass of facultative heterotrophs
        ddt_bFac = dil * (-m_bFac)              \
                   + u_Fac * m_bFac 
        
        # Biomass of nitrate denitrifiers
        ddt_b1Den = dil * (-m_b1Den)            \
                   + u_1Den * m_b1Den
        
        # Biomass of nitrite denitrifiers
        ddt_b2Den = dil * (-m_b2Den)            \
                   + u_2Den * m_b2Den
        
        # Biomass of full denitrifiers
        ddt_b3Den = dil * (-m_b3Den)            \
                   + u_3Den * m_b3Den 
        
        # Biomass of ammonia oxidising archaea
        ddt_bAOO = dil * (-m_bAOO)              \
                   + u_AOO * m_bAOO 
        
        # Biomass of nitrite oxidising bacteria
        ddt_bNOO = dil * (-m_bNOO)              \
                   + u_NOO * m_bNOO
        
        # Biomass of anammox bacteria
        ddt_bAOX = dil * (-m_bAOX)              \
                   + u_AOX * m_bAOX 
        
        
        # apply changes to state variables normalised by timestep (dt = timesteps per day)
        m_Sd = m_Sd + ddt_Sd * dt
        m_Sp = m_Sp + ddt_Sp * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NO3 = m_NO3 + ddt_NO3 * dt
        m_NO2 = m_NO2 + ddt_NO2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_N2 = m_N2 + ddt_N2 * dt
        m_bHet = m_bHet + ddt_bHet * dt
        m_bFac = m_bFac + ddt_bFac * dt
        m_b1Den = m_b1Den + ddt_b1Den * dt
        m_b2Den = m_b2Den + ddt_b2Den * dt
        m_b3Den = m_b3Den + ddt_b3Den * dt
        m_bAOO = m_bAOO + ddt_bAOO * dt
        m_bNOO = m_bNOO + ddt_bNOO * dt
        m_bAOX = m_bAOX + ddt_bAOX * dt
        
        # pulse OM and O2 into the chemostat
        if (t % int((1/dt * pulse_int))) == 0:
            m_Sd = m_Sd + pulse_Sd
            m_bHet = m_bHet + pulse_bHet
            m_bFac = m_bFac + pulse_bFac
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_Sd[i] = m_Sd 
            out_Sp[i] = m_Sp
            out_O2[i] = m_O2
            out_NO3[i] = m_NO3
            out_NO2[i] = m_NO2 
            out_NH4[i] = m_NH4 
            out_N2[i] = m_N2
            out_bHet[i] = m_bHet
            out_bFac[i] = m_bFac
            out_b1Den[i] = m_b1Den
            out_b2Den[i] = m_b2Den
            out_b3Den[i] = m_b3Den 
            out_bAOO[i] = m_bAOO
            out_bNOO[i] = m_bNOO
            out_bAOX[i] = m_bAOX
            out_facaer[i] = np.nanmean(facaer)
            out_rHet[i] = heterotrophy
            out_rHetAer[i] = aer_heterotrophy
            out_rO2C[i] = oxy_consumption
            out_r1Den[i] = den_nar
            out_r2Den[i] = den_nir
            out_r3Den[i] = den_full
            out_rAOO[i] = ammonia_ox
            out_rNOO[i] = nitrite_ox
            out_rAOX[i] = anammox_nh4
            
    return out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, out_facaer, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX



#%% plot the results

def plot_results(nn_output, out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, out_rHet, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX):
    fig = plt.figure(figsize=(16,6))
    gs = GridSpec(1,4)

    ax1 = plt.subplot(gs[0,0])
    plt.title('External substrates')
    plt.plot(np.arange(nn_output+1), out_Sd, color='k', linestyle='-', label='Sd ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_Sp, color='grey', linestyle='-', label='Sp ($\mu$M)')
    plt.plot(np.arange(nn_output+1), out_O2*1e3, color='firebrick', linestyle='-', label='O$_2$ (nM)')
    plt.legend()

    ax2 = plt.subplot(gs[0,1])
    plt.title('Nitrogen species')
    plt.plot(np.arange(nn_output+1), out_NO3, color='k', linestyle='-', label='NO$_3$')
    plt.plot(np.arange(nn_output+1), out_NO2, color='firebrick', linestyle='-', label='NO$_2$')
    plt.plot(np.arange(nn_output+1), out_NH4, color='goldenrod', linestyle='-', label='NH$_4$')
    plt.plot(np.arange(nn_output+1), out_N2, color='royalblue', linestyle='-', label='N$_2$')
    totalN = out_NO3 + out_NO2 + out_NH4 + out_N2 + out_bHet + out_b1Den + out_b2Den + out_b3Den + out_bFac + out_bAOO + out_bNOO + out_bAOX + out_Sd + out_Sp
    plt.plot(np.arange(nn_output+1), totalN, color='k', linestyle=':', linewidth=2.0, label="Total N")
    plt.legend()

    ax3 = plt.subplot(gs[0,2])
    plt.title('Biomasses (uM N)')
    plt.plot(np.arange(nn_output+1), out_bHet, color='k', linestyle='-', label='Het')
    plt.plot(np.arange(nn_output+1), out_bFac, color='k', linestyle='--', label='Fac')
    plt.plot(np.arange(nn_output+1), out_b1Den, color='firebrick', linestyle='-', label='1Den')
    plt.plot(np.arange(nn_output+1), out_b2Den, color='firebrick', linestyle='--', label='2Den')
    plt.plot(np.arange(nn_output+1), out_b3Den, color='firebrick', linestyle=':', label='3Den')
    plt.plot(np.arange(nn_output+1), out_bAOO, color='goldenrod', linestyle='-', label='AOO')
    plt.plot(np.arange(nn_output+1), out_bNOO, color='royalblue', linestyle='-', label='NOO')
    plt.plot(np.arange(nn_output+1), out_bAOX, color='forestgreen', linestyle='-', label='AOX')
    plt.legend()
    
    ax4 = plt.subplot(gs[0,3])
    plt.title('Rates (uM / day)')
    plt.plot(np.arange(nn_output+1), out_rHet, color='k', linestyle='-', label='Heterotrophy')
    plt.plot(np.arange(nn_output+1), out_rO2C, color='k', linestyle='--', label='O2 loss')
    plt.plot(np.arange(nn_output+1), out_r1Den, color='firebrick', linestyle='-', label='NO3 --> NO2')
    plt.plot(np.arange(nn_output+1), out_r2Den, color='firebrick', linestyle='--', label='NO2 --> N2')
    plt.plot(np.arange(nn_output+1), out_r3Den, color='firebrick', linestyle=':', label='NO3 --> N2')
    plt.plot(np.arange(nn_output+1), out_rAOO, color='goldenrod', linestyle='-', label='NH4 --> NO2')
    plt.plot(np.arange(nn_output+1), out_rNOO, color='royalblue', linestyle='-', label='NO2 --> NO3')
    plt.plot(np.arange(nn_output+1), out_rAOX, color='forestgreen', linestyle='-', label='NH4 + 1.16NO2 --> 0.99N2')
    plt.legend(loc='center right', bbox_to_anchor=(2.1,0.5), ncol=1)
    
    plt.subplots_adjust(right=0.825, left=0.05)
    
    ax1.set_yscale('log'); ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    
    return fig


#%% begin loop of experiments

%%time

for k in np.arange(len(Sd0_exp)):
    for m in np.arange(len(O20_exp)):
        print(k,m)
        
        # 1) Chemostat inputs
        in_Sd = Sd0_exp[k]
        in_O2 = O20_exp[m]
        in_Sp = 0.0  
        in_NO3 = 30.0
        in_NO2 = 0.0
        in_NH4 = 0.25
        
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
        pulse_int = 300
        pulse_Sd = 0.0
        pulse_bHet = 0.00
        pulse_bFac = 0.00
        pulse_O2 = 0.0
        
        # 3) Call main model
        (out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, out_facaer, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX) = OMZredox(timesteps, nn_output, dt, dil, out_at_day, pulse_Sd, pulse_bHet, pulse_bFac, pulse_O2, pulse_int, \
                                                                                                                                                                                                                                                                 po_aer, po_aoo, po_noo, VmaxS, K_s, VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_1DenFac, K_n_Den, VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, VmaxNH4_AOX, VmaxNO2_AOX, \
                                                                                                                                                                                                                                                                 y_oHet, y_oO2, y_oHetFac, y_oO2Fac, y_n1DenFac, y_n1NO3Fac, y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
                                                                                                                                                                                                                                                                 in_Sd, in_Sp, in_O2, in_NO3, in_NO2, in_NH4, in_bHet, in_bFac, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX)
        
        # 4) plot the results
        plot_results(nn_output, out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, out_rHet, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX)
        
        # 5) Record solutions in initialised arrays
        fin_O2[k,m] = np.nanmean(out_O2[-50::])
        fin_Sd[k,m] = np.nanmean(out_Sd[-50::])
        fin_Sp[k,m] = np.nanmean(out_Sp[-50::])
        fin_NO3[k,m] = np.nanmean(out_NO3[-50::])
        fin_NO2[k,m] = np.nanmean(out_NO2[-50::])
        fin_NH4[k,m] = np.nanmean(out_NH4[-50::])
        fin_N2[k,m] = np.nanmean(out_N2[-50::])
        fin_bHet[k,m] = np.nanmean(out_bHet[-50::])
        fin_bFac[k,m] = np.nanmean(out_bFac[-50::])
        fin_b1Den[k,m] = np.nanmean(out_b1Den[-50::])
        fin_b2Den[k,m] = np.nanmean(out_b2Den[-50::])
        fin_b3Den[k,m] = np.nanmean(out_b3Den[-50::])
        fin_bAOO[k,m] = np.nanmean(out_bAOO[-50::])
        fin_bNOO[k,m] = np.nanmean(out_bNOO[-50::])
        fin_bAOX[k,m] = np.nanmean(out_bAOX[-50::])
        fin_facaer[k,m] = np.nanmean(out_facaer[-50::])
        fin_rHet[k,m] = np.nanmean(out_rHet[-50::])
        fin_rHetAer[k,m] = np.nanmean(out_rHetAer[-50::])
        fin_rO2C[k,m] = np.nanmean(out_rO2C[-50::])
        fin_r1Den[k,m] = np.nanmean(out_r1Den[-50::])
        fin_r2Den[k,m] = np.nanmean(out_r2Den[-50::])
        fin_r3Den[k,m] = np.nanmean(out_r3Den[-50::])
        fin_rAOO[k,m] = np.nanmean(out_rAOO[-50::])
        fin_rNOO[k,m] = np.nanmean(out_rNOO[-50::])
        fin_rAOX[k,m] = np.nanmean(out_rAOX[-50::])
        

print("Oxygen concentrations (nM) = ",fin_O2*1e3)
print("Organic N concentrations = ",fin_Sd)
print("NO3 concentrations = ",fin_NO3)
print("NO2 concentrations = ",fin_NO2)
print("NH4 concentrations = ",fin_NH4)
print("Rate of N2 production = ",(fin_rAOX*y_nh4AOX*e_n2AOX*0.5)+(fin_r2Den+fin_r3Den)*0.5)
print("Proportion of N2 produced by anammox = ", ((fin_rAOX*y_nh4AOX*e_n2AOX*0.5) / ((fin_rAOX*y_nh4AOX*e_n2AOX*0.5)+(fin_r2Den+fin_r3Den)*0.5)*100 ))

# 6. check conservation of mass if dilution rate is set to zero
if dil == 0.0:
    end_N = fin_Sd + fin_Sp + fin_NO3 + fin_NO2 + fin_NH4 + fin_N2 + fin_bHet + fin_bFac + fin_b1Den + fin_b2Den + fin_b3Den + fin_bAOO + fin_bNOO + fin_bAOX
    ini_N = in_Sd + in_Sp + in_NO3 + in_NO2 + in_NH4 + in_bHet + in_bFac + in_b1Den + in_b2Den + in_b3Den + in_bAOO + in_bNOO + in_bAOX
    for k in np.arange(len(Sd0_exp)):
        for m in np.arange(len(O20_exp)):
            print(" Checking conservation of N mass ")
            print(" Initial Nitrogen =", ini_N)
            print(" Final Nitrogen =", end_N[k,m])
            
        

#%% plot the Org:O2

fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)


O20, Sd0 = np.meshgrid(O20_exp, Sd0_exp)

fig = plt.figure(figsize=(8,6))
gs = GridSpec(1,1)
ax1 = plt.subplot(gs[0])
ax1.tick_params(labelsize=fstic)
p1 = plt.contourf(O20_exp, Sd0_exp, Sd0/O20, levels=np.arange(0,2.01,0.1), cmap=colmap)
ax1.set_xlabel('Oxygen supply rate ($\mu$M / day)', fontsize=fslab)
ax1.set_ylabel('Organic N supply rate ($\mu$M / day)', fontsize=fslab)

cbar = plt.colorbar(p1)
cbar.ax.set_ylabel('Org N : O$_2$ supply ratio', fontsize=fslab)

os.chdir("C://Users/pearseb/Dropbox/PostDoc/my articles/Buchanan & Zakem - aerobic anaerobic competition/figures")
fig.savefig('orgN_O2_supply.png', dpi=300)
fig.savefig('transparent/orgN_O2_supply.png', dpi=300, transparent=True)


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

p1 = ax1.contourf(O20_exp, Sd0_exp, fin_O2, levels=np.arange(0,10.1,0.5), extend='max', cmap=colmap)
p2 = ax2.contourf(O20_exp, Sd0_exp, fin_NO3, levels=np.arange(20,30.1,0.5), extend='both', cmap=colmap)
p3 = ax3.contourf(O20_exp, Sd0_exp, fin_NO2, levels=np.arange(0,5.1,0.25), extend='max', cmap=colmap)
p4 = ax4.contourf(O20_exp, Sd0_exp, fin_NH4, levels=np.arange(0,0.51,0.05), extend='max', cmap=colmap)
p5 = ax5.contourf(O20_exp, Sd0_exp, (fin_rAOX + fin_r2Den)*1e3, levels=np.arange(0,101,5), extend='max', cmap=colmap)
p6 = ax6.contourf(O20_exp, Sd0_exp, fin_rAOX / (fin_rAOX + fin_r2Den) * 100, levels=np.arange(0,101,5), cmap=colmap)
p7 = ax7.contourf(O20_exp, Sd0_exp, ((fin_rHet - fin_rHetAer)/fin_rHet) * 100, levels=np.arange(0,101,5), cmap=colmap)
p8 = ax8.contourf(O20_exp, Sd0_exp, fin_rNOO*1e3, levels=np.arange(0,21,1), extend='max', cmap=colmap)
p9 = ax9.contourf(O20_exp, Sd0_exp, fin_rAOO*1e3, levels=np.arange(0,21,1), extend='max', cmap=colmap)

# delineate import thresholds
col = 'firebrick'; lw = 2.0
#c1 = ax1.contour(O20_exp, Sd0_exp, fin_O2, levels=[0.012], colors=col, linewidths=lw)
#c2 = ax2.contour(O20_exp, Sd0_exp, fin_NO3, levels=[30], colors=col, linewidths=lw)
#c3 = ax3.contour(O20_exp, Sd0_exp, fin_NO2, levels=[0.188], colors=col, linewidths=lw)
#c4 = ax4.contour(O20_exp, Sd0_exp, fin_NH4, levels=[0.188], colors=col, linewidths=lw)
#c5 = ax5.contour(O20_exp, Sd0_exp, (fin_rAOX + fin_r2Den)*1e3, levels=[50], colors=col, linewidths=lw)
#c6 = ax6.contour(O20_exp, Sd0_exp, fin_rAOX / (fin_rAOX + fin_r2Den) * 100, levels=[30], colors=col, linewidths=lw)

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
babbin_conditions = fin_NO2_babbin * fin_NO2ox_babbin * fin_N2prod_babbin * fin_perAnammox_babbin

c1 = ax1.contour(O20_exp, Sd0_exp, babbin_conditions, levels=[0.5], colors=col, linewidths=lw)


ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax3.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax4.tick_params(labelsize=fstic, labelbottom=False)
ax5.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax6.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax7.tick_params(labelsize=fstic)
ax8.tick_params(labelsize=fstic, labelleft=False)
ax9.tick_params(labelsize=fstic, labelleft=False)

ax1.set_ylabel('N$_{org}$ supply ($\mu$M / day)', fontsize=fslab)
ax4.set_ylabel('N$_{org}$ supply ($\mu$M / day)', fontsize=fslab)
ax7.set_ylabel('N$_{org}$ supply ($\mu$M / day)', fontsize=fslab)
ax7.set_xlabel('O$_2$ supply ($\mu$M / day)', fontsize=fslab)
ax8.set_xlabel('O$_2$ supply ($\mu$M / day)', fontsize=fslab)
ax9.set_xlabel('O$_2$ supply ($\mu$M / day)', fontsize=fslab)

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
fig.savefig('outcomes_oldtraits.png', dpi=300)
fig.savefig('transparent/outcomes_oldtraits.png', dpi=300, transparent=True)




