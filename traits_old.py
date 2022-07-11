# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:31:11 2022

Purpose
-------
    Old traits used in Zakem et al. 2019 ISME

@author: pearseb
"""

from diffusive_o2_coefficient import po_coef

### 1.0 Uptake Parameterisations

# organic matter (uncertain)
VmaxS = 1.0     # mol orgN / mol cell / day
K_s = 0.1        # uM

# DIN
VmaxN_1Den = 50.8        # mol DIN / mol cell N / day at 20C
VmaxN_1DenFac = 50.8        # mol DIN / mol cell N / day at 20C
VmaxN_2Den = 50.8        # mol DIN / mol cell N / day at 20C
VmaxN_3Den = 50.8        # mol DIN / mol cell N / day at 20C
K_n_Den = 0.133         # uM for denitrification
VmaxN_AOO = 50.8    # mol DIN / mol cell N / day at 20C for AOA (Zakem et al 2018)
K_n_AOO = 0.133      # uM NH4 for AOA (Martens-Habbena et al. 2009; Horak et al. 2013; Newell et al. 2013)
VmaxN_NOO = 50.8    # mol DIN / mol cell N / day
K_n_NOO = 0.133      # uM NO2 for NOB (Sun et al. 2017)
VmaxNH4_AOX = 50.8    # mol DIN / mol cell N / day
VmaxNO2_AOX = 50.8    # mol DIN / mol cell N / day
K_nh4_AOX = 0.133
K_no2_AOX = 0.133


### interesting aside here... the max growth rate is 7 per day for aerobic heterotrophs
### assuming that mu_max = Vmax * yield

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


### 3.0 diffusive oxygen requirements based on cell diameters and carbon contents

# cell sizes (um), where vol = 4/3 * pi * r^3
diam_aer = 0.5
diam_den = diam_aer
diam_aoo = diam_aer
diam_noo = diam_aer
diam_aox = diam_aer

# cellular C:N of microbes
CN_aer = 5.0
CN_den = CN_aer
CN_aoo = CN_aer
CN_noo = CN_aer
CN_aox = CN_aer

# cell quotas (mol C / um^3)
Qc_aer = 0.22 * 1e-12 / 12.0
Qc_den = Qc_aer
Qc_aoo = Qc_aer
Qc_noo = Qc_aer
Qc_aox = Qc_aer

# diffusive oxygen coefficient
dc = 1.5776 * 1e-5      # cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day

po_aer = po_coef(diam_aer, Qc_aer, CN_aer)
po_den = po_coef(diam_den, Qc_den, CN_den)
po_aoo = po_coef(diam_aoo, Qc_aoo, CN_aoo)
po_noo = po_coef(diam_noo, Qc_noo, CN_noo)
po_aox = po_coef(diam_aox, Qc_aox, CN_aox)
