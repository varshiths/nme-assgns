#!/usr/bin/python3

from enum import Enum
import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt

class Model(Enum):
    RS = 1
    IB = 2
    CH = 3

configs = {
    Model.RS : {
        "C"     : 100e-12   ,
        "Kz"    : 0.7e-6    ,
        "Er"    : -60e-3    ,
        "Et"    : -40e-3    ,
        "a"     : 0.03e3    ,
        "b"     : -2e-9     ,
        "c"     : -50e-3    ,
        "d"     : 100e-12   ,
        "vp"    : 35e-3     ,
    },
    Model.IB : {
        "C"     : 150e-12   ,
        "Kz"    : 1.2e-6    ,
        "Er"    : -75e-3    ,
        "Et"    : -45e-3    ,
        "a"     : 0.01e3    ,
        "b"     : +5e-9     ,
        "c"     : -56e-3    ,
        "d"     : 130e-12   ,
        "vp"    : 30e-3     ,
    },
    Model.CH : {
        "C"     : 50e-12    ,
        "Kz"    : 1.5e-6    ,
        "Er"    : -60e-3    ,
        "Et"    : -40e-3    ,
        "a"     : 0.03e3    ,
        "b"     : +1e-9     ,
        "c"     : -40e-3    ,
        "d"     : 150e-12   ,
        "vp"    : 25e-3     ,
    },
}

T = 500e-3
delta = 0.1e-3
M = int(T/delta)

def IZH_V(U, V, I, model):
    '''Returns the value of f given v(t), i(t)
    for equation involving voltage 
    '''
    _p = configs[model]
    return (_p["Kz"]*(V-_p["Er"])*(V-_p["Et"]) - U + I)/_p["C"]

def IZH_U(U, V, model):
    '''Returns the value of f given v(t), i(t)
    for equation involving U value function 
    '''
    _p = configs[model]
    return _p["a"]*(_p["b"]*(V-_p["Er"])-U)

def IZH_ssV(model):
    _p = configs[model]
    return _p["Et"] + _p["b"]/_p["Kz"]
def IZH_ssU(model):
    _p = configs[model]
    return _p["b"]*(IZH_ssV(model) - _p["Er"])

def initalize_models_and_current():
    # ctypes = np.array([400e-12]*3 + [500e-12]*3 + [600e-12]*3)
    # models = [Model.RS, Model.IB, Model.CH]*3
    ctypes = np.array([400e-12, 500e-12, 600e-12]*3)
    models = [Model.RS]*3 + [Model.IB]*3 + [Model.CH]*3
    models = list(zip(models, ctypes))

    cvals = np.tile(ctypes, (M, 1)).transpose()
    currents = { x: y for x, y in zip(models, cvals)}
    return models, currents

def initalize_U_V(models):
    N = len(models)
    U = np.zeros((N, M))
    V = np.zeros((N, M))
    initUs = [ IZH_ssU(model) for model in models ]
    initVs = [ IZH_ssV(model) for model in models ]
    U[:, 0] = initUs
    V[:, 0] = initVs
    return U, V

def runge_kutta_4_sim_and_reset(U, V, f, g, models):

    def f_and_g(*args):
        return f(*args), g(*args)

    for i in range(1, M):
        for mn, model in enumerate(models):
            
            k1, l1 = f_and_g(U[mn, i-1]              , V[mn, i-1]              , i-1   , model)
            k2, l2 = f_and_g(U[mn, i-1] + delta*k1/2 , V[mn, i-1] + delta*l1/2 , i-0.5 , model)
            k3, l3 = f_and_g(U[mn, i-1] + delta*k2/2 , V[mn, i-1] + delta*l2/2 , i-0.5 , model)
            k4, l4 = f_and_g(U[mn, i-1] + delta*k3   , V[mn, i-1] + delta*l3   , i     , model)

            U[mn, i] = U[mn, i-1] + delta*(k1+2*k2+2*k3+k4)/6
            V[mn, i] = V[mn, i-1] + delta*(l1+2*l2+2*l3+l4)/6

            reset = (V[mn, i] >= configs[model[0]]["vp"]).astype(int)
            rest_U_add = configs[model[0]]["d"]
            rest_V = configs[model[0]]["c"]

            U[mn, i] = rest_U_add*reset + U[mn, i]
            V[mn, i] = rest_V*reset + V[mn, i]*(1-reset)

    return U, V

def main():

    # for model in [Model.RS, Model.IB, Model.CH]:
    #     print(model)
    #     print("U", IZH_ssU(model))
    #     print("V", IZH_ssV(model))

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

    U, V = runge_kutta_4_sim_and_reset(
        U, V,
        f=lambda U, V, t, model: IZH_U(U, V, model[0]),
        g=lambda U, V, t, model: IZH_V(U, V, get_curr(currents[model], t), model[0]),
        models=models,
        )

    for mn, model in enumerate(models):
        if mn % 3 == 0:
            plt.figure()
        plt.plot(np.arange(M)*delta, V[mn], label="Current: %s"%(currents[model][0]))
        plt.legend(loc='upper right', shadow=True)
        if mn % 3 == 2:
            plt.savefig("Q2.%d.png"%(mn-2))

    # plt.show()

if __name__ == '__main__':
    main()
    # pass