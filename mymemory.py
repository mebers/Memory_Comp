# -*- coding: utf-8 -*-

import lalsimulation as lalsim
import h5py
from gwtools import harmonics
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
import utils



# Evaluating G function.
def G_ang_int(l1,l2,l3,m1,m2,m3):
    
    eval = (-1)**(m1+m2)*np.sqrt(((2.*l1+1.)*(2.*l2+1)*(2.*l3+1))/(4*np.pi)) \
                *wigner_3j(l1,l2,l3,0,-2,2)*wigner_3j(l1,l2,l3,-m1,m2,-m3)
    return eval

# print G_ang_int(2,2,2,0,0,-2)

def h20_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, 0, f_ref, phi_ref=phi_ref)    

    h22 = mode_dict['h_l2m2']
    #Compute gradient
    h22_dot = np.gradient(h22,dt)
    h22_dot_c = np.conjugate(h22_dot)
    prod_hdot = h22_dot*h22_dot_c

    dh20mem = dist_mpc*10**6*utils.PC_SI/utils.C_SI*1./np.sqrt(24.)*2.*G_ang_int(2,2,2,0,2,2)*prod_hdot
    # Hereditary integral
    h20mem = np.cumsum(dh20mem)*dt
    hmem = h20mem*harmonics.sYlm(-2, 2, 0, inclination, np.pi/2)


    return t, hmem
    
# Generate a waveform
dt = 1./4096.
dist_mpc = 100.
f_low = 20.
f_ref = 20.

q = 1.
chi1 = np.array([0.6, -0.3, 0.1])
chi2 = np.array([0.3, 0.6, -0.1])
M = 70.
inclination = np.pi/2
phi_ref = np.pi/4

approximant = 'NRSur7dq4'

"""
t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
            M, dist_mpc, 0, f_ref, phi_ref=phi_ref)    
h22 = mode_dict['h_l2m2']
h22_dot = np.gradient(h22,dt)
print(h22_dot*h22_dot*dist_mpc*10**6*utils.PC_SI)
exit()
"""
# time, h = h+ -1j * hx
#for (key,values) in mode_dict.items() :
#   print(key)
t, h20 = h20_dom_mem(approximant, q, [0., 0., 0.0], [0.0, 0.0, 0.0], dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='non-spinning, equal mass')

t, h20 = h20_dom_mem(approximant, 3.2, [0., 0., 0.0], [0.0, 0.0, 0.0], dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='non-spinning, q=3.2')

t, h20 = h20_dom_mem(approximant, q, [0., 0., 0.8], [0.0, 0.0, 0.8], dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='aligned spins, equal mass')

t, h20 = h20_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='precessing, equal mass')

t, h20 = h20_dom_mem(approximant, 3.4, chi1, chi2, dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='precessing, q=3.4')

#plt.plot(t,np.imag(h20))
plt.tight_layout()
plt.legend(loc=2)
plt.xlabel(r'$t$')
plt.ylabel(r'$h_+^\mathrm{mem}$')
plt.savefig('memwaveform.pdf')
plt.close()