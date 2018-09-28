#!/usr/bin/python3

import numpy as np

delta = 0.1e-3

C = 300e-12
gL = 30e-9
Vt = 20e-3
El = -70e-3
Rp = 2e-3

RpT = int(Rp/delta)

def LIF(V, I):
	'''Returns the value of f given v(t), i(t)
	'''
	return (-gL * (V-El) + I)/C

def LIF_tc():
	return gL * (Vt - El)
def LIF_ssv():
	return El

def initalize_V(N, M):
	V = np.zeros((N, M))
	V[:, 0] = LIF_ssv()
	return V

def get_voltage_ref_spikes(V_, I, refraction):

    # import pdb; pdb.set_trace()

    k1 = LIF(V_, I[:, 0])
    k2 = LIF(V_ + delta*k1, I[:, 1])

    V = V_ + delta*(k1+k2)/2

    nrmask = (refraction == 0).astype(int)
    V = El*(1-nrmask) + V*nrmask

    smask = (V >= Vt).astype(int)
    V = El*smask + V*(1-smask)

    refraction = (refraction-1)*(1-nrmask) + RpT*smask*nrmask

    spikers = np.where(smask)[0]
    return V, refraction, spikers
