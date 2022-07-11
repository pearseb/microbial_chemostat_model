# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:34:28 2022

Purpose
-------
    Calculate the maximum rate that O2 can be diffused into a cell based on its traits

@author: pearseb
"""

import numpy as np

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
