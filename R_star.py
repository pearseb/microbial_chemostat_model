# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:45:06 2022

Purpose
-------
    Calculate R* (subsistence concentration of nutient for microbial population)

@author: pearseb
"""


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