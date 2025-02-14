#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21/03/2024: * Komplex çarpımlar için "moving_average_fast_cmpx" eklendi.

Created on Tue Feb 20 11:42:54 2024

@author: tansu
"""
import numpy as np

# pythran export moving_average(float[], int, float[])
def moving_average(data, wsize, initials):
    dsize = len(data)
    y = np.zeros(dsize)               
    buffer = np.zeros(dsize+wsize-1)
    buffer[:(wsize-1)] = initials
    buffer[-(dsize):] = data
    
    # Bu alt kısım "valid" konvolüsyona eşdeğer.
    y[0] = np.sum(buffer[:wsize]) # İlk değer..
    for i in range(dsize-1): # ve diğerleri..
        y[i+1] = y[i]-buffer[i]+buffer[i+wsize] 
    
    initials = buffer[-(wsize-1):]
    return y/wsize, initials

# Bunu radyo programında kullanmak için buffer ve başlangıç değerlerinin
# dışarıdan önceden hazırlanıp parametre olarak girildiği durum için hazırladım.
# pythran export moving_average_fast(float[], float[])
# pythran export moving_average_fast(float32[], float32[])
def moving_average_fast(buffer, window):
    bsize = len(buffer)
    wsize = len(window)
    dsize = bsize-wsize+1
    y = np.zeros(dsize)                       
    # Bu alt kısım "valid" konvolüsyona eşdeğer.
    y[0] = np.sum(buffer[:wsize]) # İlk değer..
    for i in range(dsize-1): # ve diğerleri..
        y[i+1] = y[i]-buffer[i]+buffer[i+wsize]         
    return y/wsize

# pythran export moving_average_fast_cmpx(complex128[], complex128[])
# pythran export moving_average_fast_cmpx(complex128[], float[])
def moving_average_fast_cmpx(buffer, window):
    bsize = len(buffer)
    wsize = len(window)
    dsize = bsize-wsize+1
    y = np.zeros(dsize).astype(np.complex128)                       
    # Bu alt kısım "valid" konvolüsyona eşdeğer.
    y[0] = np.sum(buffer[:wsize]) # İlk değer..
    for i in range(dsize-1): # ve diğerleri..
        y[i+1] = y[i]-buffer[i]+buffer[i+wsize]         
    return y/wsize
