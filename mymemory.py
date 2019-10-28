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

def h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, 0, f_ref, phi_ref)    

    h22 = mode_dict['h_l2m2']

    #Compute gradient
    h22_dot = np.gradient(h22,dt)
    h22_dot_c = np.conjugate(h22_dot)
    prod_hdot = h22_dot*h22_dot_c

    dh20mem = dist_mpc*10**6*utils.PC_SI/utils.C_SI*1./np.sqrt(24.)*2.*G_ang_int(2,2,2,0,2,2)*prod_hdot

    # Hereditary integral
    h20mem = np.cumsum(dh20mem)*dt
    hmem = h20mem*harmonics.sYlm(-2, 2, 0, inclination, phi_ref)

    return t, hmem

def h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
            M, dist_mpc, f_low, f_ref,phi_ref)  

    #mode_dict_dot = np.gradient(mode_dict,dt)
    mode_dict.update({mode: np.gradient(mode_dict[mode],dt) for mode in mode_dict.keys()})

    dh20mem_p = np.zeros(len(t))
    dh20mem_c = np.zeros(len(t))
    const = dist_mpc*10**6*utils.PC_SI/utils.C_SI*1./np.sqrt(24.)
    
    for llp in range(2,5):
        for llpp in range(2,5):
            for mp in range(-llp,llp+1):
                for mpp in range(-llpp,llpp+1):
                    
                    if mp != mpp:
                        continue
                    if G_ang_int(2,llp,llpp,0,mp,mpp)==0:
                        continue
                    
                    hdotp = mode_dict['h_l%dm%d'%(llp, mp)]
                    hdotp_conj = np.conjugate(mode_dict['h_l%dm%d'%(llpp, mpp)])
                    prod_hdot_p = np.array(np.real(hdotp*hdotp_conj))
                    prod_hdot_c = np.array(np.imag(hdotp*hdotp_conj))
                        
                    dh20mem_p = dh20mem_p + const*G_ang_int(2,llp,llpp,0,mp,mpp)*prod_hdot_p
                    dh20mem_c = dh20mem_c + const*G_ang_int(2,llp,llpp,0,mp,mpp)*prod_hdot_c

                    print(llp,llpp,mp,mpp)
                    #imode = mode_dict['h_l%dm%d'%(ll, m)]
    
    h20mem_p = np.cumsum(dh20mem_p)*dt   
    h20mem_c = np.cumsum(dh20mem_c)*dt
    hmem_p = h20mem_p*harmonics.sYlm(-2, 2, 0, inclination, np.pi/2)
    hmem_c = h20mem_c*harmonics.sYlm(-2, 2, 0, inclination, np.pi/2)

    return t, hmem_p, hmem_c

    
# Generate a waveform
dt = 1./4096.
dist_mpc = 100.
f_low = 20.
f_ref = 20.

q = 1.9
chi1 = np.array([0., 0., 0.5])
chi2 = np.array([0., 0., 0.4])
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

# time, h = h+ -1j * hx
#for (key,values) in mode_dict.items() :
#   print(key)
t, h20 = h_dom_mem(approximant, q, [0., 0., 0.0], [0.0, 0.0, 0.0], dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='non-spinning, equal mass')

t, h20 = h_dom_mem(approximant, 3.2, [0., 0., 0.0], [0.0, 0.0, 0.0], dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='non-spinning, q=3.2')

t, h20 = h_dom_mem(approximant, q, [0., 0., 0.8], [0.0, 0.0, 0.8], dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='aligned spins, equal mass')

t, h20 = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='precessing, equal mass')

t, h20 = h_dom_mem(approximant, 3.4, chi1, chi2, dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
plt.plot(t,np.real(h20),label='precessing, q=3.4')
"""


t, hmemdom = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)
tt, hmem, hmemc = h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, 0,f_ref, phi_ref,inclination)

plt.plot(t,np.real(hmemdom),label='Dominant memory')
plt.plot(tt,np.real(hmem),label='memory in h_plus')
plt.plot(tt,np.real(hmemc),label='memory in h_cross')


plt.tight_layout()
plt.legend(loc=2)
plt.xlabel(r'$t$')
plt.ylabel(r'$h^\mathrm{mem}$')
plt.savefig('memdom_vs_mem20.pdf')
plt.close()