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
	# print(ic)

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

	# nitialise based on `\alpha` and `k`
	I = initalize_current()
	V = initalize_voltage(I)
	# final values in matrix V
	V = runge_kutta_2_and_reset(V, func=lambda V, t: LIF(V, I[:, t]))

	# compute the frequency of the spikes
	base_currents = I[:, 0]
	tps = np.argwhere((V == El).astype(int))

	adels = []
	for i in range(N):
		tpss = (tps[tps[:, 0] == i][:, 1]).astype(float)
		adels.append(np.mean(np.ediff1d(tpss)) * delta)

	for i in range(2, N, 2):
		plt.figure()
		plt.title("Potential of Neuron %d"%i)
		plt.plot(np.arange(M)*delta, V[i-1])
		plt.xlabel("time (s)")
		plt.ylabel("voltage (V)")
		plt.savefig("Q1.n%d.png"%i)

	plt.figure()
	plt.title("Average Time Period")
	plt.plot(base_currents, adels)
	plt.ylabel("time period (s)")
	plt.xlabel("current (A)")
	plt.savefig("Q1.avg.png")

	# plt.show()

if __name__ == '__main__':
	main()