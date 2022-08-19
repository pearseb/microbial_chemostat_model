# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:51:33 2022

@author: pearseb
"""

import numpy as np

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten


def line_plot(nn_output, out_Sd, out_Sp, out_O2, out_NO3, out_NO2, out_NH4, out_N2, \
              out_bHet, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_bAOO, out_bNOO, out_bAOX, \
              out_rHet, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_rAOO, out_rNOO, out_rAOX):
    
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
    plt.plot(np.arange(nn_output+1), out_rO2C, color='k', linestyle='--', label='O$_2$ loss')
    plt.plot(np.arange(nn_output+1), out_r1Den, color='firebrick', linestyle='-', label='NO$_3$ --> NO$_2$')
    plt.plot(np.arange(nn_output+1), out_r2Den, color='firebrick', linestyle='--', label='NO$_2$ --> N$_2$')
    plt.plot(np.arange(nn_output+1), out_r3Den, color='firebrick', linestyle=':', label='NO$_3$ --> N$_2$')
    plt.plot(np.arange(nn_output+1), out_rAOO, color='goldenrod', linestyle='-', label='NH$_4$ --> NO$_2$')
    plt.plot(np.arange(nn_output+1), out_rNOO, color='royalblue', linestyle='-', label='NO$_2$ --> NO$_3$')
    plt.plot(np.arange(nn_output+1), out_rAOX, color='forestgreen', linestyle='-', label='Anammox (NH$_4$ loss)')
    plt.legend(loc='center right', bbox_to_anchor=(1.8,0.5), ncol=1)
    
    plt.subplots_adjust(right=0.825, left=0.05)
    
    ax1.set_yscale('log'); ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    
    return fig