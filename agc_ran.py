#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25/09/2024: * "hann_func" fonksiyonunda değişiklik yapıldı.
            * "my_agc" fonksiyonunda görülen yanlışlık düzeltildi. Buna göre
              genliği düşük olan sinyaller yükseltilip, belirli bir genlikten
              (mid_level_db) yüksek olanlar ise daha az yükseltiliyor. 
              
16/02/2024: * "my_agc" ve ilgili fonksiyonlar eklendi.

15/02/2024: * İkinci derleme parametreleri satırı eklendi. 

Created on Wed Feb 14 11:27:01 2024

@author: tansu
"""
import numpy as np 

# pythran export agc(complex128[], float, float, float, float, float)
# pythran export agc(float64[], float, float, float, float, float)
def agc(data, alpha, beta, set_point, max_gain, initial):
    out = []    
    amp = initial
    gain = 1
    for x in data:
        item = np.abs(x)
        if item != 0:
            if item > amp:
                amp = (item-amp)*alpha + amp
            else:
                amp = (item-amp)*beta + amp
            gain = set_point/amp            
        else:
            gain = 1            
        out.append(x*gain)
    return np.array(out), amp

# pythran export fast_agc(complex128[], float, float, float, float)
# pythran export fast_agc(float64[], float, float, float, float)
def fast_agc(data, alpha, set_point, max_gain, initial):
    gain = initial
    out = []
    for x in data:
        item = np.abs(x)
        new_val = x*gain
        out.append(new_val)
        gain += (set_point-item)*alpha
        if gain > max_gain:
            gain = max_gain
    return np.array(out), gain

# def hann_func(n, size):
#     return (0.5 - 0.5*np.cos(2*np.pi*n/size))

# pythran export hann_func(float, float) 
def hann_func(n, size):
    return (0.5 + 0.5*np.cos(2*np.pi*n/size))

# def my_agc(x, min_level_db, max_level_db, min_gain, max_gain):
#     tlevel = 10*np.log10(np.mean(np.abs(x)**2))
#     if tlevel > max_level_db:
#         tlevel = max_level_db
#     if tlevel < min_level_db:
#         tlevel = min_level_db
#     size = max_level_db - min_level_db
#     delta_gain = max_gain - min_gain
#     return delta_gain*hann_func(tlevel, size) + min_gain

# pythran export my_agc(complex128[], float, float, float, float)
def my_agc(x, min_level_db, max_level_db, min_gain, max_gain):  
    tlevel = 10*np.log10(np.mean(np.abs(x)**2))
    if tlevel > max_level_db:
        tlevel = max_level_db
    if tlevel < min_level_db:
        tlevel = min_level_db
    size = max_level_db - min_level_db
    mid_level_db = (max_level_db + min_level_db)/2
    delta_gain = max_gain - min_gain
    return delta_gain*hann_func(tlevel-mid_level_db, size) + min_gain

# pythran export leveler(float, float, float)
def leveler(setpoint, alpha, initial):
    return (setpoint-initial)*alpha+initial