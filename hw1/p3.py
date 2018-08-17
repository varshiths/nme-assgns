#!/usr/bin/python3

from enum import Enum
import numpy as np
import pdb as pdb

from scipy.optimize import newton as NR

import matplotlib.pyplot as plt

class Model(Enum):
    RS = 1
    IB = 2
    CH = 3

configs = {
    Model.RS : {
        "C"     : 200   * 1e-12  ,
        "gL"    : 10    * 1e-9   ,
        "El"    : -70   * 1e-3   ,
        "Vt"    : -50   * 1e-3   ,
        "delT"  : 2     * 1e-3   ,
        "a"     : 2     * 1e-9   ,
        "tw"    : 30    * 1e-3   ,
        "b"     : 0     * 1e-12  ,
        "Vr"    : -58   * 1e-3   ,
    },
    Model.IB : {
        "C"     : 130   * 1e-12  ,
        "gL"    : 18    * 1e-9   ,
        "El"    : -58   * 1e-3   ,
        "Vt"    : -50   * 1e-3   ,
        "delT"  : 2     * 1e-3   ,
        "a"     : 4     * 1e-9   ,
        "tw"    : 150   * 1e-3   ,
        "b"     : 120   * 1e-12  ,
        "Vr"    : -50   * 1e-3   ,
    },
    Model.CH : {
        "C"     : 200   * 1e-12  ,
        "gL"    : 10    * 1e-9   ,
        "El"    : -58   * 1e-3   ,
        "Vt"    : -50   * 1e-3   ,
        "delT"  : 2     * 1e-3   ,
        "a"     : 2     * 1e-9   ,
        "tw"    : 120   * 1e-3   ,
        "b"     : 100   * 1e-12  ,
        "Vr"    : -46   * 1e-3   ,
    },
}

T = 500e-3
delta = 0.1e-3
M = int(T/delta)

def AEF_V(U, V, I, model):
    '''Returns the value of f given v(t), i(t)
    for equation involving voltage 
    '''
    _p = configs[model]
    return ( -_p["gL"]*(V-_p["El"]) + _p["gL"]*_p["delT"]*np.exp((V-_p["Vt"])/_p["delT"]) - U + I )/_p["C"]

def AEF_U(U, V, model):
    '''Returns the value of f given v(t), i(t)
    for equation involving U value function 
    '''
    _p = configs[model]
    return (_p["a"]*(V-_p["El"])-U)/_p["tw"]

ssV = {}
def AEF_ssV(model):
    global ssV
    if model in ssV.keys():
        return ssV[model]

    _p = configs[model]
    ssV_f = lambda x: AEF_V( _p["a"]*(x-_p["El"]), x, 0, model )

    initV = 0
    val = NR(ssV_f, 0, tol=1e-8)
    ssV[model] = val
    return val

def AEF_ssU(model):
    _p = configs[model]
    return _p["a"] * ( AEF_ssV(model) - _p["El"] )

def initalize_models_and_current():
    # ctypes = np.array([250e-12]*3 + [350e-12]*3 + [450e-12]*3)
    # models = [Model.RS, Model.IB, Model.CH]*3
    
    ctypes = np.array([250e-12, 350e-12, 450e-12]*3)
    models = [Model.RS]*3 + [Model.IB]*3 + [Model.CH]*3

    models = list(zip(models, ctypes))

    cvals = np.tile(ctypes, (M, 1)).transpose()
    currents = { x: y for x, y in zip(models, cvals)}
    return models, currents

def initalize_U_V(models):
    N = len(models)
    U = np.zeros((N, M))
    V = np.zeros((N, M))
    initUs = [ AEF_ssU(model) for model in models ]
    initVs = [ AEF_ssV(model) for model in models ]
    U[:, 0] = initUs
    V[:, 0] = initVs
    return U, V

def euler_sim_and_reset(U, V, f, g, models):

    def f_and_g(*args):
        return f(*args), g(*args)

    for i in range(1, M):
        for mn, model in enumerate(models):
            
            k, l = f_and_g(U[mn, i-1], V[mn, i-1], i-1, model)
            U[mn, i] = U[mn, i-1] + delta*k
            V[mn, i] = V[mn, i-1] + delta*l

            reset = (V[mn, i] >= 0).astype(int)
            rest_U_add = configs[model[0]]["b"]
            rest_V = configs[model[0]]["Vr"]

            U[mn, i] = rest_U_add*reset + U[mn, i]
            V[mn, i] = rest_V*reset + V[mn, i]*(1-reset)

    return U, V

def main():

    # for model in [Model.RS, Model.IB, Model.CH]:
    #     print(model)
    #     print("U ", AEF_ssU(model))
    #     print("V ", AEF_ssV(model))

    models, currents = initalize_models_and_current()
    jmodels = [x for x, y in models]
    U, V = initalize_U_V(jmodels)
    
    def get_curr(I, t):
        if isinstance(t, int):
            return I[t]
        else:
            v1 = I[int(t)]
            v2 = I[int(t)+1]
            weight = t - int(t)
            return v2*weight + v1*(1-weight)

    U, V = euler_sim_and_reset(
        U, V,
        f=lambda U, V, t, model: AEF_U(U, V, model[0]),
        g=lambda U, V, t, model: AEF_V(U, V, get_curr(currents[model], t), model[0]),
        models=models,
        )

    for mn, model in enumerate(models):
        # if mn % 3 == 0:
        plt.figure()
        plt.title( "Model: %s Current: %s"%(model[0], currents[model][0]) )
        plt.plot(np.arange(M)*delta, V[mn])
        plt.xlabel("time (s)")
        plt.ylabel("voltage (V)")
        # plt.legend(loc='upper right', shadow=True)
        # if mn % 3 == 2:
        plt.savefig("Q3.%d.png"%(mn))

    # plt.show()

if __name__ == '__main__':
    main()
    # pass