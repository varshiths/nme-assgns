#!/usr/bin/python3

import numpy as np
import pdb as pdb

import matplotlib.pyplot as plt

from lif import initalize_V, get_voltage_ref_spikes

np.random.seed(3)

iI = 50e-9
w0 = 3000

Io = 1e-12
tau = 15e-3
taus = tau/4

T = 100e-3
delta = 0.1e-3
M = int(T/delta)

N = 5

def get_connectivity_matrices(N):
    ''' returns weights and delays in indices
    '''

    weights = np.zeros((N, N))
    delays = np.zeros((N, N))

    # b->a
    weights[1, 0] = w0
    delays[1, 0] = int(1e-3/delta)
    # b->e
    weights[1, 4] = w0
    delays[1, 4] = int(8e-3/delta)

    # c->a
    weights[2, 0] = w0
    delays[2, 0] = int(5e-3/delta)
    # c->e
    weights[2, 4] = w0
    delays[2, 4] = int(5e-3/delta)

    # d->a
    weights[3, 0] = w0
    delays[3, 0] = int(9e-3/delta)
    # d->e
    weights[3, 4] = w0
    delays[3, 4] = int(1e-3/delta)

    return weights, delays

def get_initial_external_currents(N):
    '''returns only currents for bcd, dcb
    '''
    dmilm = int(1e-3/delta)

    tcur1 = np.zeros((N, M))
    tcur2 = np.zeros((N, M))

    tcur1[1, 0*dmilm:(0+1)*dmilm ] = iI
    tcur1[2, 4*dmilm:(4+1)*dmilm ] = iI
    tcur1[3, 8*dmilm:(8+1)*dmilm ] = iI

    tcur2[3, 0*dmilm:(0+1)*dmilm ] = iI
    tcur2[2, 3*dmilm:(3+1)*dmilm ] = iI
    tcur2[1, 7*dmilm:(7+1)*dmilm ] = iI

    return tcur1, tcur2

def get_current_due_to_postspikes(weights, delays, spikes, spikers, t):

    current = np.zeros(weights.shape[0])
    for pn in range(weights.shape[0]):
        current[pn] = Io * sum([
                weights[spiker][pn] * \
                (np.exp(-(t-spike-delays[spiker][pn])*delta/tau) - np.exp(-(t-spike-delays[spiker][pn])*delta/taus)) * \
                int(t-spike-delays[spiker][pn] > 0)
            for spike, spiker in zip(spikes[pn], spikers[pn])
            ])

    # import pdb; pdb.set_trace()

    return current

def curr_due_to_prespikes(weights, delays, spikes, i):

    N = len(weights)
    ispikes = [[] for i in range(N)]
    ispikers = [[] for i in range(N)]

    for post in range(N):
        for pre in range(N):
            if weights[pre][post] > 0:
                ispikes[post].extend(spikes[pre])
                ispikers[post].extend([pre]*len(spikes[pre]))

    # import pdb; pdb.set_trace()

    current = get_current_due_to_postspikes(weights, delays, ispikes, ispikers, i)
    return current

def simulate_network(V, current, weights, delays):
    V = V.copy()
    current = current.copy()

    spikes = [ [] for i in range(N) ]
    refraction = np.zeros((N))

    for i in range(1, M):
        # update main current with current generated by spikes
        current[:, i] += curr_due_to_prespikes(weights, delays, spikes, i)

        # update voltage with correct value
        # with refractory period updates and spikes
        Vi, refraction, spikers = get_voltage_ref_spikes(
            V[:, i-1], 
            current[:, i-1:i+1], 
            refraction=refraction, 
            )

        V[:, i] = Vi
        for spiker in spikers:
            spikes[spiker].append(i)

    return V, current

def main():

    weights, delays = get_connectivity_matrices(N)

    current1, current2 = get_initial_external_currents(N)
    V = initalize_V(N, M)

    V1, current1 = simulate_network(V, current1, weights, delays)
    V2, current2 = simulate_network(V, current2, weights, delays)

    for i in range(N):
        plot_curr_and_resp(current1[i], V1[i], "P1.1.%d.png" % i)

    for i in range(N):
        plot_curr_and_resp(current2[i], V2[i], "P1.2.%d.png" % i)

def plot_curr_and_resp(current, V, filen):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.xlabel("time (s)")

    ax1.set_ylim([-0.5*1e-10, 60*1e-9])
    ax1.set_title("Total Current Input (A)")
    ax1.plot(np.arange(M)*delta, current)

    ax2.set_ylim([-75*1e-3, +75*1e-3])
    ax2.set_title("Posterior Neuron Response (V)")
    ax2.plot(np.arange(M)*delta, V)

    fig.savefig(filen)

if __name__ == '__main__':
    main()