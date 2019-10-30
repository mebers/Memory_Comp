# -*- coding: utf-8 -*-
#############################################################################
##      Filename: mymemory.py
##      Author: Michael Ebersold
##      Created: 18-10-2019
##      Description: Compute memory from oscillatory waveforms
#############################################################################

import lalsimulation as lalsim
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
import utils
from timeit import default_timer as timer



# Evaluating G function.
def G_ang_int(l1,l2,l3,m1,m2,m3):
    #Evalute angular integral using equation (10) in memory paper
    
    eval = (-1.)**(m1+m2)*np.sqrt(((2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.))/(4.*np.pi)) \
                *wigner_3j(l1,l2,l3,0,-2,2)*wigner_3j(l1,l2,l3,-m1,m2,-m3)
    
    return float(eval)


def h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    if approximant == ('NRSur7dq4' or 'NRSur7dq2'):
        t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, f_low, f_ref, phi_ref)
        
        h22 = mode_dict['h_l2m2']
        
    else: # For waveforms that have only the dominant mode
        t, h = utils.generate_LAL_waveform(approximant, q, chi1, chi2, dt, M, \
                dist_mpc, f_low, f_ref, inclination, phi_ref)
        
        h22 = h/utils.sYlm(-2, 2, 2, inclination, np.pi/2-phi_ref)
    

    #Compute gradient
    h22_dot = np.gradient(h22,dt)
    h22_dot_c = np.conjugate(h22_dot)
    prod_hdot = h22_dot*h22_dot_c

    dh20mem = dist_mpc*10**6*utils.PC_SI/utils.C_SI*1./np.sqrt(24.)*2.*G_ang_int(2,2,2,0,2,2)*prod_hdot

    # Hereditary integral
    h20mem = np.cumsum(dh20mem)*dt
    hmem = h20mem*utils.sYlm(-2, 2, 0, inclination, np.pi/2-phi_ref)

    return t, np.real(hmem), np.imag(hmem)


def h_memory20(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    if approximant == ('NRSur7dq4' or 'NRSur7dq2'):
        t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, f_low, f_ref,phi_ref)
    else:
        print('Use the surrogate waveforms "NRSur7dq2" or "NRSur7dq4", for other waveform models \
                    the memory computation is not working yet')
        exit()
    
    # Take time derivative of oscillatory hlm modes
    mode_dict.update({mode: np.gradient(mode_dict[mode],dt) for mode in mode_dict.keys()})

    dh20mem_p = np.zeros(len(t))
    dh20mem_c = np.zeros(len(t))
    const = dist_mpc*10**6*utils.PC_SI/utils.C_SI*1./np.sqrt(24.)
    
    llmax = 4
    for llp in range(2,llmax+1):
        for llpp in range(2,llmax+1):
            for mp in range(-llp,llp+1):
                for mpp in range(-llpp,llpp+1):
                    
                    if mp != mpp:
                        continue
                    if G_ang_int(2,llp,llpp,0,mp,mpp)==0:
                        continue
                    
                    #start = timer()
                    hdotp = mode_dict['h_l%dm%d'%(llp, mp)]
                    hdotp_conj = np.conjugate(mode_dict['h_l%dm%d'%(llpp, mpp)])

                    prod_hdot_p = np.array(np.real(hdotp*hdotp_conj))
                    prod_hdot_c = np.array(np.imag(hdotp*hdotp_conj))

                    dh20mem_p = dh20mem_p + const*G_ang_int(2,llp,llpp,0,mp,mpp)*prod_hdot_p
                    dh20mem_c = dh20mem_c + const*G_ang_int(2,llp,llpp,0,mp,mpp)*prod_hdot_c
                    
                    #end = timer()
                    #print('Time for mode %d %d %d %d'%(llp,llpp,mp,mpp))
                    #print(end - start)
                    print('Combination %d %d %d %d %d %d done'%(ll,llp,llpp,m,mp,mpp))


    
    h20mem_p = np.cumsum(dh20mem_p)*dt   
    h20mem_c = np.cumsum(dh20mem_c)*dt
    hmem_p = h20mem_p*utils.sYlm(-2, 2, 0, inclination, np.pi/2-phi_ref)
    hmem_c = h20mem_c*utils.sYlm(-2, 2, 0, inclination, np.pi/2-phi_ref)

    return t, hmem_p, hmem_c


def h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    if approximant == ('NRSur7dq4' or 'NRSur7dq2'):
        t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, f_low, f_ref,phi_ref)
    else:
        print('Use the surrogate waveforms "NRSur7dq2" or "NRSur7dq2", for other waveform models \
                    the memory computation is not working yet')
        exit()
    
    # Take time derivative of oscillatory hlm modes
    mode_dict.update({mode: np.gradient(mode_dict[mode],dt) for mode in mode_dict.keys()})
    
    llmax = 4
    memp_mode_dict = {}
    memc_mode_dict = {}
    
    for ll in range(2,llmax+1):
        const = dist_mpc*10**6*utils.PC_SI/utils.C_SI*np.sqrt(np.math.factorial(ll-2)/float(np.math.factorial(ll+2)))

        for m in range(-ll,ll+1):
            dhmem_p = np.zeros(len(t))
            dhmem_c = np.zeros(len(t))
            
            for llp in range(2,llmax+1):
                for llpp in range(2,llmax+1):
                    for mp in range(-llp,llp+1):
                        for mpp in range(-llpp,llpp+1):
                    
                            if m != mp - mpp:
                                continue
                            if G_ang_int(ll,llp,llpp,m,mp,mpp)==0:
                                continue
                    
                            #start = timer()
                            hdotp = mode_dict['h_l%dm%d'%(llp, mp)]
                            hdotp_conj = np.conjugate(mode_dict['h_l%dm%d'%(llpp, mpp)])

                            prod_hdot_p = np.array(np.real(hdotp*hdotp_conj))
                            prod_hdot_c = np.array(np.imag(hdotp*hdotp_conj))

                            dhmem_p = dhmem_p + const*G_ang_int(ll,llp,llpp,m,mp,mpp)*prod_hdot_p
                            dhmem_c = dhmem_c + const*G_ang_int(ll,llp,llpp,m,mp,mpp)*prod_hdot_c
                            
                            #end = timer()
                            #print('Time for mode %d %d %d %d %d %d'%(ll,llp,llpp,m,mp,mpp))
                            #print(end - start)
                            print('Combination %d %d %d %d %d %d done'%(ll,llp,llpp,m,mp,mpp))

            memp_mode_dict['h_l%dm%d'%(ll, m)] = dhmem_p
            memc_mode_dict['h_l%dm%d'%(ll, m)] = dhmem_c
            
    memp_mode_dict.update({mode: np.cumsum(memp_mode_dict[mode])*dt for mode in memp_mode_dict.keys()})
    memc_mode_dict.update({mode: np.cumsum(memc_mode_dict[mode])*dt for mode in memc_mode_dict.keys()})
    hmem_p = np.zeros(len(t))
    hmem_c = np.zeros(len(t))
    
    for ll in range(2,llmax+1):
        for m in range(-ll,ll+1):
            hmem_p = hmem_p + np.real((memp_mode_dict['h_l%dm%d'%(ll, m)]+1j*memc_mode_dict['h_l%dm%d'%(ll, m)]) \
                                      *utils.sYlm(-2, ll, m, inclination, np.pi/2-phi_ref))
            hmem_c = hmem_c - np.imag((memp_mode_dict['h_l%dm%d'%(ll, m)]+1j*memc_mode_dict['h_l%dm%d'%(ll, m)]) \
                                      *utils.sYlm(-2, ll, m, inclination, np.pi/2-phi_ref))

    return t, hmem_p, hmem_c
    

# Generate a waveform
dt = 1./(4096)
dist_mpc = 100
f_low = 20.
f_ref = 20.

q = 1.8
chi1 = np.array([0., 0., 0.5])
chi2 = np.array([0., 0., 0.4])
M = 70
inclination = np.pi/2
phi_ref = np.pi/4

approximant = 'SEOBNRv4'
t, hmemdom, hmemdomc = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)
plt.plot(t,np.real(hmemdom),label='SEOBNRv4')

approximant = 'IMRPhenomD'
t, hmemdom, hmemdomc = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)
plt.plot(t,np.real(hmemdom),label='IMRPhenomD')

approximant = 'NRSur7dq2'
t, hmemdom, hmemdomc = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)
plt.plot(t,np.real(hmemdom),label='NRSur7dq2')

approximant = 'NRSur7dq4'
t, hmemdom, hmemdomc = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)
plt.plot(t,np.real(hmemdom),label='NRSur7dq4')

#t, hmemdom, hmemdomc = h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)
#tt, hmem20, hmemc20 = h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)
#ttt, hmem, hmemc = h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)

#plt.plot(t,np.real(hmemdom),label='Dominant memory')
#plt.plot(tt,hmem20,label=r'memory in $h_+^{20}$')
#plt.plot(tt,hmemc20,label=r'memory in $h_\times^{20}$')
#plt.plot(ttt,hmem,label=r'memory in $h_+$')
#plt.plot(ttt,hmemc,label=r'memory in $h_\times$')

plt.legend(loc=2)
plt.xlabel(r'$t$')
plt.ylabel(r'$h^\mathrm{mem}$')
plt.tight_layout()
plt.savefig('memory_test.pdf')
plt.close()