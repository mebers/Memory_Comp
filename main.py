# -*- coding: utf-8 -*-
#############################################################################
##      Filename: mymemory.py
##      Author: Michael Ebersold
##      Created: 18-10-2019
##      Description: Get memory from oscillatory waveforms
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import utils
import memory


# Generate a waveform
# Choose waveform approximant (Computation is optimized for the surrogates ('NRSur7dq4'))
# For other waveform models only the h22 components is taken into account -> memory is typically underestimated
# Support for NR waveforms will come in the future
approximant = 'NRSur7dq4'

M = 60.  # Total mass in solar masses
q = 1.0   # mass ratio m1/m2
chi1 = np.array([0.0, 0.0, 0.])   # Dimensionless spin vector of black hole 1
chi2 = np.array([0.0, 0., 0.])   # Dimensionless spin vector of black hole 2

inclination = np.pi/2   # Inclination angle (0 is face-on, np.pi/2 is edge-on)
phi_ref = 0.            # reference phase angle, only goes into the Ylms
dist_mpc = 100.         # Distance to the binary in Mpc

# Choose starting and reference frequency of the waveform
# For the surrogates and NR waveforms: 0 will use the full waveform
f_low = 30.
f_ref = f_low

# Set the timestep, should be small enough especially for highly oscillatory signals (small masses)
dt = 1./(4*4096)


# h_dom_mem(-): Compute the memory in the h20 mode sourced solely by the h22 mode
# Returns: time array, memory in the plus polarization -1j*memory in the cross polarization
# Has to be used for other waveform models than the surrogates.

#t, hmemdom = memory.h_dom_mem(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)

# h_memory20(-): Compute the memory in the h20 mode sourced by all oscillatory modes
# Returns: time array, memory in the plus polarization -1j*memory in the cross polarization

#tt, hmem20, hmemc20 = memory.h_memory20(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)

# h_memory(-): Compute the memory in all modes, sourced by all oscillatory modes
# Returns: time array, memory in the plus polarization -1j*memory in the cross polarization

ttt, hmem = memory.h_memory(approximant, q, chi1, chi2, dt, M, dist_mpc, f_low,f_ref, phi_ref,inclination)


# To generate the oscillatory waveform, call:
tosc, hosc = utils.generate_LAL_waveform(approximant, q, chi1, chi2, dt, M, dist_mpc, \
                    f_low, f_ref, inclination, phi_ref, ellMax=None, single_mode=None)

# Make plots
plt.figure(figsize=(9,4))
plt.rcParams.update({'font.size': 12})
#plt.plot(t,np.real(hmemdom),label=r'memory from $h_{22}$')

#plt.plot(tosc, np.real(hosc),label=r'$h_+$')
plt.plot([tosc[0]-0.01,tosc[-1]+0.01],[0,0], 'k',linewidth=0.5)
plt.plot(tosc,np.real(hosc),'b',linewidth=1.0,label=r'oscillatory waveform')

plt.plot(ttt,np.real(hmem),'r',linewidth=1.0,label=r'memory contribution')

plt.plot(tosc, np.real(hosc)+np.real(hmem),'g',linewidth=1.0,label=r'full waveform')

#plt.plot(ttt,(-1)*np.imag(hmem),label=r'memory in $h_\times$')


plt.title('Gravitational wave strain vs. time')
plt.legend(loc=(0.1,0.2))
plt.xlabel(r'$t$')
plt.ylabel(r'$h_+(t)$')
plt.tight_layout()
plt.savefig('test.pdf')
