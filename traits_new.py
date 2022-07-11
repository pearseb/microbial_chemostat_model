# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:31:11 2022

Purpose
-------
    Old traits used in Zakem et al. 2019 ISME

@author: pearseb
"""

import numpy as np
from diffusive_o2_coefficient import po_coef
from yield_from_stoichiometry import yield_stoich


### 1.0 max growth rates (per day)
mumax_Het = 0.5     # Rappé et al., 2002
mumax_AOO = 0.5     # Wutcher et al. 2006; Horak et al. 2013; Shafiee et al. 2019; Qin et al. 2015
mumax_NOO = 0.96    # Spieck et al. 2014)
mumax_AOX = 0.2     # Okabe et al. 2021 ISME | Lotti et al. 2014


### 2.0 diffusive oxygen requirements based on cell diameters and carbon contents

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


### 2.0 Yields (y), products (e) and maximum uptake rate (Vmax)
###     Vmax = ( mu_max * Quota ) / yield 
###     we remove the need for the Quota term by considering everything in mols N per mol N-biomass, such that
###     Vmax = mu_max / yield (units of mol N per mol N-biomass per day)

dO = 29.1   # assumes constant stoichiometry of all sinking organic matter of C6.6-H10.9-O2.6-N using equation: d = 4C + H - 2O -3N
dB = 18.0   # assumes constant stoichiometry of all heterotrophic bacteria of C4.5-H7-O2-N using equation: d = 4C + H - 2O -3N

# solve for the yield of heterotrophic bacteria using approach of Sinsabaugh et al. 2013
Y_max = 0.6     # maximum possible growth efficiency measured in the field
B_CN = 4.5      # C:N of bacterial biomass (White et al. 2019)
OM_CN = 6.6     # C:N of labile dissolved organic matter (Letscher et al. 2015)
K_CN = 0.5      # C:N half-saturation coefficient (Sinsabaugh & Follstad Shah 2012 - Ecoenzymatic Stoichiometry and Ecological Theory)
EEA_CN = 1.123  # relative rate of enzymatic processing of complex C and complex N molecules into simple precursors for biosynthesis (Sinsabaugh & Follstad Shah 2012)

# aerobic heterotrophy 
y_oHet = yield_stoich(Y_max, B_CN, OM_CN, K_CN, EEA_CN) / B_CN * OM_CN  # OM_CN / B_CN converts to mol BioN per mol OrgN 
f_oHet = y_oHet * dB/dO  # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oO2 = (f_oHet/dB) / ((1.0-f_oHet)/4.0)  # yield of biomass per unit oxygen reduced
VmaxS = mumax_Het / y_oHet  # mol Org N (mol BioN)-1 per day


den_penalty = 0.9

# nitrate reduction (NO3 --> NO2)
y_n1Den = y_oHet * den_penalty          # we asssume that the yield of anaerobic respiration using NO3 is 90% of aerobic respiration
f_n1Den = y_n1Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n1NO3 = (f_n1Den/dB) / ((1.0-f_n1Den)/2.0) # yield of biomass per unit nitrate reduced (denominator of 2 because N is reduced by 2 electrons)
e_n1Den = 1.0 / y_n1NO3         # mols NO2 produced per mol biomass synthesised
VmaxN_1Den = mumax_Het * den_penalty / y_n1NO3        # mol DIN / mol cell N / day at 20C

# nitrite reduction (NO2 --> N2)
y_n2Den = y_oHet * den_penalty          # we asssume that the yield of anaerobic respiration using NO2 is 90% of aerobic respiration
f_n2Den = y_n2Den * dB/dO       # fraction of electrons used for biomass synthesis
y_n2NO2 = (f_n2Den/dB) / ((1.0-f_n2Den)/2.0) # yield of biomass per unit nitrite reduced
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
VmaxN_AOO = mumax_AOO / y_nAOO

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_nNOO = 0.0126         # mol N biomass per mol NO2 (Bayer et al. 2022)
d_NOO = 3.4*4 + 7 - 2*2 - 3*1
f_NOO = (y_nNOO * d_NOO) /2          # fraction of electrons going to biomass synthesis from electron donor (NO2) (Zakem et al. 2022)
y_oNOO = 4*f_NOO*(1-f_NOO)/d_NOO         # mol N biomass per mol O2 (Bayer et al. 2022)
VmaxN_NOO = mumax_NOO / y_nNOO

# Chemoautotrophic anammox (NH4 + NO2 --> NO3 + N2)
y_nh4AOX = 1./70.0                  # mol N biomass per mol NH4 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
y_no2AOX = 1./81.0                  # mol N biomass per mol NO2 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
e_n2AOX = 139                       # mol N (as N2) formed per mol biomass N synthesised ***Rounded to nearest whole number
e_no3AOX = 11                       # mol NO3 formed per mol biomass N synthesised ***Rounded to nearest whole number
VmaxNH4_AOX = mumax_AOX / y_nh4AOX
VmaxNO2_AOX = mumax_AOX / y_no2AOX


### 3.0 Half-saturation coefficients
K_s = 0.1           # organic nitrogen (uncertain) uM
K_n_Den = 4.0       # 4 – 25 µM NO2 for denitrifiers (Almeida et al. 1995)
K_n_AOO = 0.1       # Martens-Habbena et al. 2009 Nature
K_n_NOO = 0.1       # Reported from OMZ (Sun et al. 2017) and oligotrophic conditions (Zhang et al. 2020)
K_nh4_AOX = 0.45    # Awata et al. 2013 for Scalindua
K_no2_AOX = 0.45    # Awata et al. 2013 for Scalindua actually finds a K_no2 of 3.0 uM, but this excludes anammox completely in our experiments



### Check useages of N in reactions
print("")
print("moles OrgN used in aerobic heterotrophy =",1/y_oHet)
print("moles OrgN used in facultative aerobic heterotrophy =",1/y_oHetFac)
print("moles OrgN used in facultative denitrification =",1/y_n1DenFac)
print("moles OrgN used in denitrification (NO3 --> NO2) =",1/y_n1Den)
print("moles OrgN used in denitrification (NO2 --> N2) =",1/y_n2Den)
print("moles OrgN used in denitrification (NO3 --> N2) =",1/y_n3Den)
print("moles NO3 used in denitrification (NO3 --> NO2) =",1/y_n1NO3)
print("moles NO2 used in denitrification (NO2 --> N2) =",1/y_n2NO2)
print("moles NO3 used in denitrification (NO3 --> N2) =",1/y_n3NO3)
print("moles NH4 used in ammonia oxidation =",1/y_nAOO)
print("moles NO2 used in nitrite oxidation =",1/y_nNOO)
print("moles NH4+NO2 used in anammox =",(1/y_nh4AOX + 1/y_no2AOX), "and produced as Biomass, NO3 and N2 in anammox =",(e_no3AOX + e_n2AOX + 1))

### Check usages of oxygen in reactions
print("")
print("moles O2 used in aerobic heterotrophy =",1/y_oO2)
print("moles O2 used in facultative aerobic heterotrophy =",1/y_oO2Fac)
print("moles O2 used in ammonia oxidation =",1/y_oAOO)
print("moles O2 used in nitrite oxidation =",1/y_oNOO)
