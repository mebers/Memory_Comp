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
from sympy.physics.wigner import wigner_3j
import utils


# Evaluating G function.
def G_ang_int(l1,l2,l3,m1,m2,m3):
    #Evalute angular integral using equation (10) in memory paper
    
    eval = (-1.)**(m1+m2)*np.sqrt(((2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.))/(4.*np.pi)) \
                *wigner_3j(l1,l2,l3,0,2,-2)*wigner_3j(l1,l2,l3,-m1,m2,-m3)
    
    return float(eval)


def h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    if approximant in ('NRSur7dq4','NRSur7dq2'):
        t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, f_low, f_ref, phi_ref)
        
        h22 = mode_dict['h_l2m2']
        
    else: # For waveforms that have only the dominant mode
        t, h = utils.generate_LAL_waveform(approximant, q, chi1, chi2, dt, M, \
                dist_mpc, f_low, f_ref, inclination, phi_ref)
        
        h22 = h/utils.sYlm(-2, 2, 2, inclination, np.pi/2-phi_ref)/np.sqrt(2)
    
    #Compute gradient
    h22_dot = np.gradient(h22,dt)
    h22_dot_c = np.conjugate(h22_dot)
    prod_hdot = h22_dot*h22_dot_c

    dh20mem = dist_mpc*10**6*utils.PC_SI/utils.C_SI*1./np.sqrt(24.)*2.*G_ang_int(2,2,2,0,2,2)*prod_hdot

    # Hereditary integral
    h20mem = np.cumsum(dh20mem)*dt
    hmem = h20mem*utils.sYlm(-2, 2, 0, inclination, np.pi/2-phi_ref)

    return t, hmem


def h_memory20(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref, inclination, filepath=None):
    
    if approximant in ('NRSur7dq4','NRSur7dq2'):
        t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, f_low, f_ref,phi_ref)
        
    # Choose_TD_modes of NR waveform does not work yet...
    elif approximant == 'NR_hdf5':
        t, mode_dict, q, s1x, s1y, s1z, s2x, s2y, s2z, f_low = utils.load_lvcnr_data(filepath, \
                    'all', M, dt, inclination, phi_ref, dist_mpc, 0.)
        print('mass ratio: %d'%q)
        print('spin 1: [%d, %d, %d]'%(s1x,s1y,s1z))
        print('spin 1: [%d, %d, %d]'%(s2x,s2y,s2z))
        print('f_low: %d'%f_low)
        
        
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
                    print('Combination %d %d %d %d done'%(llp,llpp,mp,mpp))
    
    h20mem_p = np.cumsum(dh20mem_p)*dt   
    h20mem_c = np.cumsum(dh20mem_c)*dt

    hmem_p = np.real((h20mem_p + 1j*h20mem_c)*utils.sYlm(-2, 2, 0, inclination, np.pi/2-phi_ref))
    hmem_c = (-1.)*np.imag((h20mem_p + 1j*h20mem_c)*utils.sYlm(-2, 2, 0, inclination, np.pi/2-phi_ref))
    
    hmem = hmem_p - 1.j*hmem_c
    return t, hmem


def h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination):
    
    # For the surrogates directly take the individual modes
    if approximant in ('NRSur7dq4','NRSur7dq2','NRHybSur3dq8','SEOBNRv4PHM'):
        t, mode_dict = utils.generate_LAL_modes(approximant, q, chi1, chi2, dt, \
                M, dist_mpc, f_low, f_ref,phi_ref)

        # For the HybridSurrogate, -m modes are not directly created
        if approximant == 'NRHybSur3dq8':
            for ll in range(2,5):
                for m in range(-ll,0):
                    # The (4,1) and (4,0) modes are not created by this model
                    if (ll==4 and m == -1):
                        mode_dict['h_l%dm%d'%(ll, 0)] = np.zeros(len(t))
                        mode_dict['h_l%dm%d'%(ll, -m)] = np.zeros(len(t))
                        mode_dict['h_l%dm%d'%(ll, m)] = np.zeros(len(t))
                    else:
                        mode_dict['h_l%dm%d'%(ll, m)] = (-1)**ll*np.conjugate(mode_dict['h_l%dm%d'%(ll, (-1*m))])
            
        # Choose time zero where the maximum of the summed individual modes is
        hlmtot = np.zeros(len(t))
        for mode in mode_dict.keys():
            hlmtot = hlmtot + np.array(mode_dict[mode])
        # Find t0 as the max of of all modes
        t0 = int(np.where(abs(hlmtot) == np.amax(abs(hlmtot)))[0])
        t = t - t[t0]

    # Choose_TD_modes of NR waveform does not work yet...    
    elif approximant == 'NR_hdf5':
        t, mode_dict, q, s1x, s1y, s1z, s2x, s2y, s2z, f_low = utils.load_lvcnr_data(filepath, \
                    'all', M, dt, inclination, phi_ref, dist_mpc, 0.)
        print('mass ratio: %d'%q)
        print('spin 1: [%d, %d, %d]'%(s1x,s1y,s1z))
        print('spin 1: [%d, %d, %d]'%(s2x,s2y,s2z))
        print('f_low: %d'%f_low)
        
    else:
        print('Use the surrogate waveforms "NRSur7dq2", "NRSur7dq4" or the precessing effective one body model with higher modes "SEOBNRv4PHM", for other waveform models you can use the function h_dom_mem(), which computes the memory due to the h22 mode')
        exit()
    
    # Take time derivative of oscillatory hlm modes
    mode_dict.update({mode: np.gradient(mode_dict[mode],dt) for mode in mode_dict.keys()})
    
    lmax = 4 # To which l the memory contributions are computed
    llmax = 4 # To which ll oscillatory modes are taken into account
    memp_mode_dict = {}
    memc_mode_dict = {}
    
    for ll in range(2,lmax+1):
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
                    
                            hdotp = mode_dict['h_l%dm%d'%(llp, mp)]
                            hdotp_conj = np.conjugate(mode_dict['h_l%dm%d'%(llpp, mpp)])

                            prod_hdot_p = np.array(np.real(hdotp*hdotp_conj))
                            prod_hdot_c = np.array(np.imag(hdotp*hdotp_conj))

                            dhmem_p = dhmem_p + const*G_ang_int(ll,llp,llpp,m,mp,mpp)*prod_hdot_p
                            dhmem_c = dhmem_c + const*G_ang_int(ll,llp,llpp,m,mp,mpp)*prod_hdot_c
                            
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
            hmem_c = hmem_c + np.imag((memp_mode_dict['h_l%dm%d'%(ll, m)]+1j*memc_mode_dict['h_l%dm%d'%(ll, m)]) \
                                      *utils.sYlm(-2, ll, m, inclination, np.pi/2-phi_ref))
            
            hmem = hmem_p - 1.j*hmem_c
    
    return t, hmem
