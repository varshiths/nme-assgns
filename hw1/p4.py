#!/usr/bin/python3

from enum import Enum
import numpy as np
import pdb as pdb

from scipy.optimize import newton as NR

import matplotlib.pyplot as plt

SI = False

C       =    1 * 1e-6
ENa     =   50 * 1e-3
EK      =  -77 * 1e-3
El      =  -55 * 1e-3
gNa     =  120 * 1e-3
gK      =   36 * 1e-3
gl      =  0.3 * 1e-3

def HH_V(V, n, m, h, I):
    return ( -iNa(V, m, h) -iK(V, n) -il(V) + I ) / C
def iNa(V, m, h):
    return gNa * m**3 * h * (V-ENa)
def iK(V, n):
    return gK * n**4 * (V-EK)
def il(V):
    return gl * (V-El)

def HH_n(V, n):
    return alpha_n(V)*(1-n) - beta_n(V)*(n)
def HH_m(V, m):
    return alpha_m(V)*(1-m) - beta_m(V)*(m)
def HH_h(V, h):
    return alpha_h(V)*(1-h) - beta_h(V)*(h)

def inm(x):
    # return x
    return 1000 * x
def alpha_n(V):
    V = inm(V)
    return ( 0.01*(V+55) )/( 1 - np.exp( -(V+55)/10 ) )
def alpha_m(V):
    V = inm(V)
    return ( 0.1*(V+40) )/( 1 - np.exp( -(V+40)/10 ) )
def alpha_h(V):
    V = inm(V)
    return 0.07*np.exp( -0.05*(V+65) )

def beta_n(V):
    V = inm(V)
    return 0.125*np.exp( -(V+65)/80 )
def beta_m(V):
    V = inm(V)
    return 4.0*np.exp( -0.0556*(V+65) )
def beta_h(V):
    V = inm(V)
    return 1.0 / ( 1 + np.exp( -0.1 * (V+35) ) )


def sse_V(V):
    '''
    Equation whose equation is steady state value for f(V) = 0
    '''
    n = HH_ss_n(V)
    m = HH_ss_m(V)
    h = HH_ss_h(V)
    return HH_V(V, n, m, h, 0)

val_ss_V = None
def HH_ss_V():
    global val_ss_V
    if val_ss_V is not None:
        return val_ss_V

    initV = 0.0
    val_ss_V = NR(sse_V, initV)
    return val_ss_V

'''
All three functions below
If V is None, return at steady state V
else return m, n, p at the value V assuming steady state
'''
def HH_ss_n(V=None):
    if V is None:
        V = HH_ss_V()
    return alpha_n(V) / ( alpha_n(V) + beta_n(V) )
def HH_ss_m(V=None):
    if V is None:
        V = HH_ss_V()
    return alpha_m(V) / ( alpha_m(V) + beta_m(V) )
def HH_ss_h(V=None):
    if V is None:
        V = HH_ss_V()
    return alpha_h(V) / ( alpha_h(V) + beta_h(V) )

N = 1
Io          = 15    * 1e-6
FT          = 300  * 1e-3
T           = 30    * 1e-3
delta       = 0.01  * 1e-3
M = int(FT/delta)
HM = int(T/delta)

def initalize_current():
    currents = np.zeros((N, M))
    # currents[:, :] = Io
    currents[:, 2*HM:3*HM] = Io
    return currents

def initalize_V_n_m_h():
    V = np.zeros((N, M))
    n = np.zeros((N, M))
    m = np.zeros((N, M))
    h = np.zeros((N, M))

    V[:, 0] = HH_ss_V()
    n[:, 0] = HH_ss_n()
    m[:, 0] = HH_ss_m()
    h[:, 0] = HH_ss_h()
    return V, n, m, h

def euler_sim(V, n, m, h, I):

    for i in range(1, M):

        tV = V[:, i-1]
        tn = n[:, i-1]
        tm = m[:, i-1]
        th = h[:, i-1]
        tI = I[:, i-1]

        # if i % 10000 == 0:
            # print("Euler: %d/%d, values: "%(i, M), tV, tn, tm, th, tI)

        V[:, i] = tV + delta * HH_V(tV, tn, tm, th, tI)
        n[:, i] = tn + delta * HH_n(tV, tn)
        m[:, i] = tm + delta * HH_m(tV, tm)
        h[:, i] = th + delta * HH_h(tV, th)     

    return V, n, m, h

def ion_currents(V, n, m, h):
    return iNa(V, m, h), iK(V, n), il(V)
def power(V, cNa, cK, cl, I):
    pT = V * ( -cNa -cK -cl + I )
    pNa = cNa * ( V-ENa )
    pK  = cK  * ( V-EK  )
    pl  = cl  * ( V-El  )
    return pNa, pK, pl, pT

def integrate(power):
    
    E = np.zeros(N)
    for i in range(M):
        E += power[:, i-1]
    return E

def energy(rNa, rK, rl, rT):

    area = 1e-8
    eNa     = integrate(rNa) * area
    eK      = integrate(rK) * area
    el      = integrate(rl) * area
    eT      = integrate(rT) * area
    return eNa, eK, el, eT

def main():

    currents = initalize_current()
    V, n, m, h = initalize_V_n_m_h()

    # print("Steady State Values")
    # print("V: ", V[0, 0])
    # print("n: ", n[0, 0])
    # print("m: ", m[0, 0])
    # print("h: ", h[0, 0])

    # import pdb; pdb.set_trace()
    V, n, m, h = euler_sim( V, n, m, h, currents )
    cNa, cK, cl = ion_currents(V, n, m, h)

    rNa, rK, rl, rT = power(V, cNa, cK, cl, currents)
    eNa, eK, el, eT = energy(rNa, rK, rl, rT)

    # print("Energy Values")
    # print("Na: ", eNa)
    # print("K: ", eK)
    # print("l: ", el)
    # print("T: ", eT)

    plt.figure()
    plt.title("Membrane Potential")
    plt.plot(np.arange(M)*delta, V[0])
    plt.xlabel("time (s)")
    plt.ylabel("voltage (V)")
    plt.savefig("Q4.1.png")

    plt.figure()
    plt.title("Ion Currents")
    plt.plot(np.arange(M)*delta, cNa[0], label="Na")
    plt.plot(np.arange(M)*delta, cK[0], label="K")
    plt.plot(np.arange(M)*delta, cl[0], label="l")
    plt.legend(loc='upper right', shadow=True)
    plt.xlabel("time (s)")
    plt.ylabel("current (A)")
    plt.savefig("Q4.2.png")

    plt.figure()
    plt.title("Power Dissipated")
    plt.plot(np.arange(M)*delta, rT[0], label="Capacitor(membrane)")
    plt.plot(np.arange(M)*delta, rNa[0], label="Na")
    plt.plot(np.arange(M)*delta, rK[0], label="K")
    plt.plot(np.arange(M)*delta, rl[0], label="l")
    plt.legend(loc='upper right', shadow=True)
    plt.xlabel("time (s)")
    plt.ylabel("power (watts)")
    plt.savefig("Q4.3.png")

    # plt.figure()
    # plt.plot(np.arange(M)*delta, n[0], label="n")
    # plt.plot(np.arange(M)*delta, m[0], label="m")
    # plt.plot(np.arange(M)*delta, h[0], label="h")
    # plt.legend(loc='upper right', shadow=True)
    # plt.savefig("Q4.3.png")



if __name__ == '__main__':
    main()
