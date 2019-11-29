import numpy as np
import matplotlib.pyplot as P
import os
import h5py
import lal
import lalsimulation as lalsim
from lal import MSUN_SI, MTSUN_SI, PC_SI, C_SI

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
    lmax = 5
    # Generating the NR waveform
    approx = lalsim.NR_hdf5

    hmodes = lalsim.SimInspiralChooseTDModes(phiRef, dt, m1SI, m2SI, \
            s1x, s1y, s1z, s2x, s2y, s2z, \
            f_low, f_ref, distance, params_NR, lmax, approx)

    t = np.arange(len(hmodes.mode.data.data)) * dt
    
    mode_dict = {}
    while hmodes is not None:
        mode_dict['h_l%dm%d'%(hmodes.l, hmodes.m)] = hmodes.mode.data.data
        hmodes = hmodes.next
    return t, mode_dict, q, s1x, s1y, s1z, s2x, s2y, s2z, f_low, f_ref

    '''
    approx = lalsim.NR_hdf5
    hp, hc = lalsim.SimInspiralChooseTDWaveform(m1SI, m2SI, s1x, s1y, s1z,
               s2x, s2y, s2z, distance, inclination, phiRef, 0.0, 0.0, 0.0,
               dt, f_low, f_ref, params_NR, approx)
    h = np.array(hp.data.data - 1.j*hc.data.data)
    t = dt *np.arange(len(h))

    q = m1SI/m2SI

    NRh5File.close()

    return t, h, q, s1x, s1y, s1z, s2x, s2y, s2z, f_low, f_ref
    '''
    
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
def generate_LAL_FDwaveform(approximant, q, chiA0, chiB0, df, M, \
    dist_mpc, f_min, f_max, f_ref, inclination=0, phi_ref=0., ellMax=None, \
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

    hp, hc = lalsim.SimInspiralChooseFDWaveform(\
        m1_kg, m2_kg, chiA0[0], chiA0[1], chiA0[2], \
        chiB0[0], chiB0[1], chiB0[2], \
        distance, inclination, phi_ref, 0, 0, 0,\
        df, f_min, f_max, f_ref, dictParams, approxTag)

    h = np.array(hp.data.data - 1.j*hc.data.data)
    f = np.arange(f_min,f_min+df*len(h),df)

    return f, h

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

#-----------------------------------------------------------------
#Spin-weighted spherical harmonic modes
def fac(n):
   result = 1
   for i in range(2, n+1):
      result *= i
   return result

def dlms(l, m, s, Theta):
    
    sq = np.sqrt(fac(l+m)*fac(l-m)*fac(l+s)*fac(l-s))
    d = 0.
    for k in range(max(0,m-s),min(l+m,l-s)+1):
        d = d + (-1.)**k*np.sin(Theta/2.)**(2.*k+s-m)*np.cos(Theta/2.)**(2.*l+m-s-2.*k)/(fac(k)*fac(l+m-k)*fac(l-s-k)*fac(s-m+k))
    return sq*d

def sYlm(s,l,m,Theta,Phi):
    
    res = (-1.)**(-s)*np.sqrt((2.*l+1)/(4*np.pi))*dlms(l,m,-s,Theta)
    
    if res==0:
        return 0.
    else:
        return complex(res*np.cos(m*Phi), res*np.sin(m*Phi))

#-----------------------------------------------------------------
#Other version of Spin-weighted spherical harmonic modes
# coefficient function
def Cslm(s, l, m):
    return np.sqrt(l*l*(4.0*l*l - 1.0)/((l*l - m*m)*(l*l - s*s)))


# recursion function
def s_lambda_lm(s, l, m, x):

    Pm = np.power(-0.5, m)

    if (m !=  s): Pm = Pm*np.power(1.0+x, (m-s)*1.0/2)
    if (m != -s): Pm = Pm*np.power(1.0-x, (m+s)*1.0/2)
   
    Pm = Pm * np.sqrt(fac(2*m + 1)*1.0/(4.0*np.pi*fac(m+s)*fac(m-s)))
   
    if (l == m):
        return Pm
   
    Pm1 = (x + s*1.0/(m+1))*Cslm(s, m+1, m)*Pm
   
    if (l == m+1):
        return Pm1
    else:
        for n in range (m+2, l+1):
            Pn = (x + s*m*1.0/(n*(n-1.0)))*Cslm(s, n, m)*Pm1 - Cslm(s, n, m)*1.0/Cslm(s, n-1, m)*Pm
            Pm = Pm1
            Pm1 = Pn  
        return Pn


def Ylm(ss, ll, mm, theta, phi):
   
    Pm = 1.0

    l = ll
    m = mm
    s = ss

    if (l < 0):
        return 0
    if (abs(m) > l or l < abs(s)):
        return 0

    if (abs(mm) < abs(ss)):
        s=mm
        m=ss
        if ((m+s) % 2):
            Pm  = -Pm

    if (m < 0):
        s=-s
        m=-m
        if ((m+s) % 2):
            Pm  = -Pm

    result = Pm * s_lambda_lm(s, l, m, np.cos(theta))

    return complex(result*np.cos(mm*phi), result*np.sin(mm*phi))

#-----------------------------------------------------------------
