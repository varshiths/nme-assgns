#!/usr/bin/python3

import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt


T = 500e-3
delta = 0.1e-3

M = int(T/delta)
N = 10
alpha = 0.1

C = 300e-12
gL = 30e-9
Vt = 20e-3
El = -70e-3

# El = -70
# Vt = 20
# gL = 30e-6
# C = 300e-9


def LIF(V, I):
	'''Returns the value of f given v(t), i(t)
	'''
	return (-gL * (V-El) + I)/C

def LIF_tc():
	return gL * (Vt - El)
def LIF_ssv():
	return El

def initalize_current():
	inp = np.ones((N, M))

	ic = LIF_tc()

	for k in range(1, N+1):
		inp[k-1, :] = (1+k*alpha)*ic

	return inp

def initalize_voltage(i):
	out = i.copy()
	out[:, 0] = LIF_ssv()
	return out

def runge_kutta_2_and_reset(Y, func):
	for i in range(1, Y.shape[1]):
		
		k1 = func(Y[:, i-1], i-1)
		k2 = func(Y[:, i-1] + delta*k1, i)

		Y[:, i] = Y[:, i-1] + delta*(k1+k2)/2

		reset = (Y[:, i] >= Vt).astype(int)
		Y[:, i] = El*reset + Y[:, i]*(1-reset)

	return Y

def main():

	# pdb.set_trace()
	
	I = initalize_current()
	V = initalize_voltage(I)
	V = runge_kutta_2_and_reset(V, func=lambda V, t: LIF(V, I[:, t]))

	base_currents = I[:, 0]
	tps = np.argwhere((V == El).astype(int))

	adels = []
	for i in range(N):
		# pdb.set_trace()
		tpss = (tps[tps[:, 0] == i][:, 1]).astype(float)
		adels.append(np.mean(np.ediff1d(tpss)) * delta)

	plt.figure(0)
	for i in range(2, N, 2):
		plt.plot(np.arange(M)*delta, V[i-1], label="Neuron: %d"%i)
	plt.legend(loc='upper right', shadow=True)

	plt.figure(1)
	plt.plot(base_currents, adels)

	plt.show()

if __name__ == '__main__':
	main()