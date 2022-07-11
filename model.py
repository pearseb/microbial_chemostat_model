# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:28:17 2022

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

### imports
import numpy as np
from numba import jit

@jit(nopython=True)
def OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
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
             in_bHet, in_bFac, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX):
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
    out_uHet = np.ones((int(nn_output)+1)) * np.nan
    out_uFac = np.ones((int(nn_output)+1)) * np.nan
    out_u1Den = np.ones((int(nn_output)+1)) * np.nan
    out_u2Den = np.ones((int(nn_output)+1)) * np.nan
    out_u3Den = np.ones((int(nn_output)+1)) * np.nan
    out_uAOO = np.ones((int(nn_output)+1)) * np.nan
    out_uNOO = np.ones((int(nn_output)+1)) * np.nan
    out_uAOX = np.ones((int(nn_output)+1)) * np.nan
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
            out_uHet[i] = u_Het
            out_uFac[i] = u_Fac
            out_u1Den[i] = u_1Den
            out_u2Den[i] = u_2Den
            out_u3Den[i] = u_3Den 
            out_uAOO[i] = u_AOO
            out_uNOO[i] = u_NOO
            out_uAOX[i] = u_AOX
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
            
    return [out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, \
            out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, \
            out_uHet, out_uFac, out_u1Den, out_u2Den, out_u3Den, out_uAOO, out_uNOO, out_uAOX, \
            out_facaer, out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX]
