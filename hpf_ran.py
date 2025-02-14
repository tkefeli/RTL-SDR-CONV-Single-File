#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15/02/2024: * Üçüncü derleme parametreleri satırı eklendi..

Created on Wed Feb 14 13:14:15 2024

@author: tansu
"""
import numpy as np
# pythran export hpf(float64[], float, float, float)
# pythran export hpf(complex128[], float, float, float)
# pythran export hpf(complex128[], complex, complex, float)
def hpf(data, x=0.0, y=0.0, alpha=0.9):
    outdata = []
    for item in data:
        out=alpha*(item-x+y)
        x = item
        y = out
        outdata.append(out)
    return np.array(outdata), x, y