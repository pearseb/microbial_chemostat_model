# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:36:40 2022

Purpose
-------
    Find the yield of heterotrophy based on the approach of Sinsabaugh et al. 2013

@author: pearseb
"""

def yield_stoich(Y_max, B_CN, OM_CN, K_CN, EEA_CN):
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