#!/usr/bin/python3

import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt

from aefrs import Model, initalize_U_V, get_U_V

np.random.seed(2)

N = 100

w0 = 50 
sigma0 = 5 

WE_MAX=500.0
GAMMA=1.0

T = 500e-3
delta = 0.1e-3
lda = 1.0

Io = 1e-12
tau = 15e-3
taus = tau/4

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

def train_synapses(we, vr, spikes):

    tmax = np.argmax(vr)
    deltk = tmax - spikes

    mask = np.ones(N)
    for i in range(N):
        currtks = deltk[i][deltk[i] >= 0]
        if currtks.size == 0:
            mask[i] = 0
            deltk[i] = 0
        else:
            deltk[i] = np.min( currtks )
    deltk = deltk.astype(np.float32)

    delwe = we * GAMMA * ( np.exp(-deltk*delta/tau) - np.exp(-deltk*delta/taus) ) * mask
    return np.clip(we+delwe, None, WE_MAX), delwe[mask == 1], deltk[mask == 1]

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

def plot_learning_scatter(delwk, deltk, _iter, filen="Q3.png"):

    plt.xlabel("time (s)")
    plt.ylabel("change in synapse weight")

    plt.xlim(-0.01, 0.3)
    plt.ylim(-10, 200)

    plt.scatter(deltk*delta, delwk, s=4, alpha=0.5, label="Iteration %d"%_iter)

    plt.legend(loc='upper right')
    plt.savefig(filen)

def main():

    train, spikes = get_spike_train()
    # print("Stimulus:", spikes)

    wsi = np.random.normal(loc=w0, scale=sigma0, size=(N))
    model = Model.RS
    initU, initV = initalize_U_V(model, M, True)

    wsn = wsi
    currenti = get_cumulative_current(spikes, wsi)
    _, V, _aspikes = get_U_V(initU, initV, model, currenti)

    _iter = 0 
    while len(_aspikes) == 0:
        _iter+=1; print("Iteration:", _iter)
        wsn, delwk, deltk = train_synapses(wsn, V, spikes)
        plot_learning_scatter( delwk, deltk, _iter )
        currenti = get_cumulative_current(spikes, wsn)
        _, V, _aspikes = get_U_V(initU, initV, model, currenti)
    print("Required number of iterations:", _iter)
    print("Weights of synapses:", wsn)

if __name__ == '__main__':
    main()