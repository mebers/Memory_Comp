# -*- coding: utf-8 -*-

import lalsimulation as lalsim
import h5py
from pycbc import waveform 
import lal
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j



# Evaluating G function.
def G_ang_int(l1,l2,l3,m1,m2,m3):
    
    eval = (-1)**(m1+m2)*np.sqrt(((2.*l1+1.)*(2.*l2+1)*(2.*l3+1))/(4*np.pi)) \
                *wigner_3j(l1,l2,l3,0,-2,2)*wigner_3j(l1,l2,l3,-m1,m2,-m3)
    return eval

# print G_ang_int(2,2,2,0,0,-2)

