# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:45:06 2022

Purpose
-------
    Calculate O2* (subsistence O2 of microbial population)

@author: pearseb
"""


def O2_star(loss, Qc, diam, dc, y_oO2):
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