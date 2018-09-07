#!/usr/bin/python3

import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt

from aefrs import Model, initalize_U_V, get_U_V

# np.random.seed(0)

T = 500e-3
delta = 0.1e-3
lda = 10.0

M = int(T/delta)

def get_spike_train():

    opt = np.exp(-lda*delta) * lda*delta
    op = np.random.uniform(size=(M))

    train = (op <= opt).astype(np.float32)
    indices = np.where(train)[0]

    return train, indices

def get_current_through_synapse(spikes):
    
    Io = 1e-12
    we = 500
    tau = 15e-3
    taus = tau/4

    current = np.zeros(M)

    si = 0
    nsp = spikes.shape[0]

    cspks = []
    for i in range(M):

        if si < nsp and spikes[si] == i-1:
            cspks.append(spikes[si])
            si += 1

        _cspks = np.array(cspks)
        exps = np.exp(-(i-_cspks)*delta/tau) - np.exp(-(i-_cspks)*delta/taus)
        current[i] = Io * we * np.sum( exps )

    return current

def main():

    train, spikes = get_spike_train()
    print("Stimulus:", spikes)
    current = get_current_through_synapse(spikes)

    model = Model.RS
    U, V = initalize_U_V(model, M, True)
    U, V, _x = get_U_V(U, V, model, current)

    print("Spikes: ", _x)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.xlabel("time (s)")

    ax1.set_ylim([-0.5*1e-10, 7.5*1e-10])
    ax1.set_title("Current through synapse (A)")
    ax1.plot(np.arange(M)*delta, current)

    ax2.set_ylim([-75*1e-3, 0])
    ax2.set_title("Posterior Neuron Response (V)")
    ax2.plot(np.arange(M)*delta, V)

    fig.savefig("Q1.png")

if __name__ == '__main__':
    main()