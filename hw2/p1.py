#!/usr/bin/python3

import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt

from aefrs import Model, initalize_U_V, get_U_V

np.random.seed(0)

T = 500e-3
delta = 0.1e-3
lda = 10.0

M = int(T/delta)

def get_spike_train():

    opt = lda / float(M)
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

    for i in range(M):
        cspks = spikes[ spikes < i ]
        current[i] = Io * we * \
            np.sum( np.exp(-(i-cspks)/tau) - np.exp(-(i-cspks)/taus) )

    return current

def main():

    train, spikes = get_spike_train()
    current = get_current_through_synapse(spikes)

    current[:] = 0

    model = (Model.RS, None)
    currents = {
        model : current
    }
    U, V = initalize_U_V([model[0]], M)
    U, V = get_U_V(U, V, [model], currents)
    U, V = U[0], V[0]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.xlabel("time (s)")

    ax1.set_title("Current through synapse (A)")
    ax1.plot(np.arange(M)*delta, current)

    ax2.set_title("Posterior Neuron Response (V)")
    ax2.plot(np.arange(M)*delta, V)

    fig.savefig("Q1.png")

if __name__ == '__main__':
    main()