import numpy as np
import matplotlib.pyplot as P
import os
import h5py

import lal
import lalsimulation as lalsim
from lal import MSUN_SI, MTSUN_SI, PC_SI, C_SI

#-----------------------------------------------------------------
def generate_random_params(qMax, Mmin, chiMax, Mmax=300):

    q = np.random.uniform(1, qMax)

    chi1mag = np.random.uniform(0, chiMax)
    chi1th = np.random.uniform(0, np.pi)
    chi1ph = np.random.uniform(0, 2*np.pi)
    chi1 = [chi1mag * np.sin(chi1th) * np.cos(chi1ph), \
            chi1mag * np.sin(chi1th) * np.sin(chi1ph), \
            chi1mag * np.cos(chi1th), \
            ]

    chi2mag = np.random.uniform(0, chiMax)
    chi2th = np.random.uniform(0, np.pi)
    chi2ph = np.random.uniform(0, 2*np.pi)
    chi2 = [chi2mag * np.sin(chi2th) * np.cos(chi2ph), \
            chi2mag * np.sin(chi2th) * np.sin(chi2ph), \
            chi2mag * np.cos(chi2th), \
            ]

    M = np.random.uniform(Mmin, Mmax)
    inclination = np.random.uniform(0, np.pi)
    phi_ref = np.random.uniform(0, 2*np.pi)

    return q, chi1, chi2, M, inclination, phi_ref

#-----------------------------------------------------------------
def set_single_mode(params, l, m):
    """ Sets modes in params dict.
        Only adds (l,m) and (l,-m) modes.
    """

    # First, create the 'empty' mode array
    ma=lalsim.SimInspiralCreateModeArray()

    # add (l,m) and (l,-m) modes
    lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
    lalsim.SimInspiralModeArrayActivateMode(ma, l, -m)

    #then insert your ModeArray into the LALDict params with
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, ma)

    return params

#-----------------------------------------------------------------
def load_lvcnr_data(filepath, mode, M, dt, inclination, phiRef, \
    dist_mpc, f_low=0):
    """ If f_low = 0, uses the entire NR data. The actual f_low will be
    returned.
    """

    NRh5File = h5py.File(filepath, 'r')

    # set mode for NR data
    params_NR = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertNumRelData(params_NR, filepath)

    if mode != 'all':
        params_NR = set_single_mode(params_NR, mode[0], mode[1])

    # Metadata parameters masses:
    m1 = NRh5File.attrs['mass1']
    m2 = NRh5File.attrs['mass2']
    m1SI = m1 * M/(m1 + m2) * MSUN_SI
    m2SI = m2 * M/(m1 + m2) * MSUN_SI

    distance = dist_mpc* 1.0e6 * PC_SI
    f_ref = f_low
    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(f_ref, M, \
        filepath)
    s1x = spins[0]
    s1y = spins[1]
    s1z = spins[2]
    s2x = spins[3]
    s2y = spins[4]
    s2z = spins[5]

    # If f_low == 0, update it to the start frequency so that the surrogate
    # gets the right start frequency
    if f_low == 0:
        f_low = NRh5File.attrs['f_lower_at_1MSUN']/M
    f_ref = f_low
    f_low = f_ref

    # Generating the NR waveform
    approx = lalsim.NR_hdf5
    hp, hc = lalsim.SimInspiralChooseTDWaveform(m1SI, m2SI, s1x, s1y, s1z,
               s2x, s2y, s2z, distance, inclination, phiRef, 0.0, 0.0, 0.0,
               dt, f_low, f_ref, params_NR, approx)

    h = np.array(hp.data.data - 1.j*hc.data.data)
    t = dt *np.arange(len(h))

    q = m1SI/m2SI

    NRh5File.close()

    return t, h, q, s1x, s1y, s1z, s2x, s2y, s2z, f_low, f_ref

#-----------------------------------------------------------------
def generate_LAL_waveform(approximant, q, chiA0, chiB0, dt, M, \
    dist_mpc, f_low, f_ref, inclination=0, phi_ref=0., ellMax=None, \
    single_mode=None):

    distance = dist_mpc* 1.0e6 * PC_SI
    approxTag = lalsim.SimInspiralGetApproximantFromString(approximant)

    # component masses of the binary
    m1_kg =  M*MSUN_SI*q/(1.+q)
    m2_kg =  M*MSUN_SI/(1.+q)

    if single_mode is not None and ellMax is not None:
        raise Exception("Specify only one of single_mode or ellMax")

    dictParams = lal.CreateDict()
    # If ellMax, load all modes with ell<=ellMax
    if ellMax is not None:
        ma=lalsim.SimInspiralCreateModeArray()
        for ell in range(2, ellMax+1):
            lalsim.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
        lalsim.SimInspiralWaveformParamsInsertModeArray(dictParams, ma)
    elif single_mode is not None:
    # If a single_mode is given, load only that mode (l,m) and (l,-m)
        dictParams = set_single_mode(dictParams, single_mode[0], single_mode[1])

    hp, hc = lalsim.SimInspiralChooseTDWaveform(\
        m1_kg, m2_kg, chiA0[0], chiA0[1], chiA0[2], \
        chiB0[0], chiB0[1], chiB0[2], \
        distance, inclination, phi_ref, 0, 0, 0,\
        dt, f_low, f_ref, dictParams, approxTag)

    h = np.array(hp.data.data - 1.j*hc.data.data)
    t = dt *np.arange(len(h))

    return t, h

#-----------------------------------------------------------------
def generate_dynamics(approximant, q, chiA0, chiB0, dt, M, \
    f_low, f_ref, phi_ref=0.):

    approxTag = lalsim.SimInspiralGetApproximantFromString(approximant)

    # component masses of the binary
    m1_kg =  M*MSUN_SI*q/(1.+q)
    m2_kg =  M*MSUN_SI/(1.+q)

    orbphase, quat0, quat1, quat2, quat3, chiAx, chiAy, \
        chiAz, chiBx, chiBy, chiBz = lalsim.PrecessingNRSurDynamics(phi_ref, \
        dt, m1_kg, m2_kg, f_low, f_ref, chiA0[0], chiA0[1], chiA0[2], \
        chiB0[0], chiB0[1], chiB0[2], approxTag)

    t = dt *np.arange(len(orbphase.data))

    quat = np.array([quat0.data, quat1.data, quat2.data, quat3.data])
    chiA = np.array([chiAx.data, chiAy.data, chiAz.data]).T
    chiB = np.array([chiBx.data, chiBy.data, chiBz.data]).T

    return t, orbphase.data, quat, chiA, chiB



#-----------------------------------------------------------------
def generate_LAL_modes(approximant, q, chiA0, chiB0, dt, M, \
    dist_mpc, f_low, f_ref, phi_ref, ellMax=None):

    distance = dist_mpc* 1.0e6 * PC_SI

    approxTag = lalsim.SimInspiralGetApproximantFromString(approximant)

    # component masses of the binary
    m1_kg =  M*MSUN_SI*q/(1.+q)
    m2_kg =  M*MSUN_SI/(1.+q)

    dictParams = lal.CreateDict()
    if ellMax is not None:
        ma=lalsim.SimInspiralCreateModeArray()
        for ell in range(2, ellMax+1):
            lalsim.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
        lalsim.SimInspiralWaveformParamsInsertModeArray(dictParams, ma)

    lmax = 5    # This in unused
    hmodes = lalsim.SimInspiralChooseTDModes(phi_ref, dt, m1_kg, m2_kg, \
        chiA0[0], chiA0[1], chiA0[2], chiB0[0], chiB0[1], chiB0[2], \
        f_low, f_ref, distance, dictParams, lmax, approxTag)

    t = np.arange(len(hmodes.mode.data.data)) * dt
    mode_dict = {}
    while hmodes is not None:
        mode_dict['h_l%dm%d'%(hmodes.l, hmodes.m)] = hmodes.mode.data.data
        hmodes = hmodes.next
    return t, mode_dict
