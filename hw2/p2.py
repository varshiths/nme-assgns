#!/usr/bin/python3

import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt

from aefrs import Model, initalize_U_V, get_U_V

np.random.seed(0)

N = 100

w0 = 50 
sigma0 = 5 

w1 = 250 
sigma1 = 25 

T = 500e-3
delta = 0.1e-3
lda = 1.0

M = int(T/delta)

def get_spike_train():

    opt = np.exp(-lda*delta) * lda*delta
    op = np.random.uniform(size=(N, M))

    train = (op <= opt).astype(np.float32)

    indices = []
    for i in range(N):
        index = np.where(train[i])
        indices.append(index[0])

    return train, indices

def get_current_through_synapse(spikes, we):
    
    Io = 1e-12
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

def get_cumulative_current(spikes, ws):
    
    currents = np.zeros((N, M))
    for i in range(N):
        currents[i] = get_current_through_synapse(spikes[i], ws[i])

    return np.sum(currents, axis=0)

def plot_curr_and_resp(current, V, filen):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.xlabel("time (s)")

    ax1.set_ylim([-0.5*1e-10, 7.5*1e-10])
    ax1.set_title("Total Current Input (A)")
    ax1.plot(np.arange(M)*delta, current)

    ax2.set_ylim([-75*1e-3, 0])
    ax2.set_title("Posterior Neuron Response (V)")
    ax2.plot(np.arange(M)*delta, V)

    fig.savefig(filen)

def main():

    train, spikes = get_spike_train()
    # print("Stimulus:", spikes)
    ws0 = np.random.normal(loc=w0, scale=sigma0, size=(N))
    ws1 = np.random.normal(loc=w1, scale=sigma1, size=(N))
    current0 = get_cumulative_current(spikes, ws0)
    current1 = get_cumulative_current(spikes, ws1)

    model = Model.RS
    initU, initV = initalize_U_V(model, M, True)
    _, V0, _x0 = get_U_V(initU, initV, model, current0)
    _, V1, _x1 = get_U_V(initU, initV, model, current1)
    print("Spikes: %d, w_mean: %f, w_std: %f" % (len(_x0), w0, sigma0))
    print("Spikes: %d, w_mean: %f, w_std: %f" % (len(_x1), w1, sigma1))

    plot_curr_and_resp( current0, V0, "Q2.a.png" )
    plot_curr_and_resp( current1, V1, "Q2.b.png" )

if __name__ == '__main__':
    main()