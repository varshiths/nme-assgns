#!/usr/bin/python3

import numpy as np
import pdb as pdb
import os

import matplotlib.pyplot as plt

import timeit

from lif import initalize_V, get_voltage_ref_spikes

np.random.seed(3)
os.system("taskset -p 0xff %d" % os.getpid())

iI = 50e-9
w0 = 3000

Io = 1e-12
tau = 15e-3
taus = tau/4

T = 1000e-3
# T = 20e-3

delta = 0.1e-3
# delta = 1e-3

M = int(T/delta)

N = 500
# N = 15

def get_spike_train(N):

    lda = 100.0
    opt = np.exp(-lda*delta) * lda*delta
    op = np.random.uniform(size=(N, M))

    train = (op <= opt).astype(np.float32)
    indices = [ np.where(train[i])[0] for i in range(N) ]

    return train, indices

def get_current_through_synapse(spikes):
    
    Io = 1e-12
    ws = 3000
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
        current[i] = Io * ws * np.sum( exps )

    return current

def get_initial_poisson_currents(N):

    _,spikes = get_spike_train(25)

    current = np.zeros((N, M))
    for i in range(25):
    # for i in range(int(5)):
        current[i] = get_current_through_synapse(spikes[i])

    return current

def get_connectivity_matrices(N):
    ''' returns weights and delays in indices
    '''

    e8 = int(0.8*N)
    e2 = int(0.2*N)
    e1 = int(0.1*N)
    # e1 = int(0.4*N)

    weights = np.zeros((N, N))
    delays = np.zeros((N, N))

    for i in range(N):
        if i < e8:
            choice = np.random.choice(N, e1, replace=False)
        else:
            choice = np.random.choice(e8, e1, replace=False)
        weights[i, choice] = \
            w0*(choice<e8).astype(int) + \
            -w0*(choice>=e8).astype(int)

    delays[:e8, :] = np.random.randint(low=1, high=int(20e-3/delta), size=(e8, N))
    delays[e8:, :] = int(1e-3/delta)

    return weights, delays

def get_current_due_to_postspikes(weights, delays, ispikes, ispikers, t):

    current = np.zeros(weights.shape[0])

    # wadj = np.reshape(weights.transpose(), (N, N, 1))
    # dadj = np.reshape(delays.transpose(), (N, N, 1))

    # fspikes = (t - np.arange(M) - dadj) * ispikes
    # icurrs = wadj * ( np.exp( -fspikes*delta/tau ) - np.exp( -fspikes*delta/taus ) ) * (fspikes > 0).astype(int)
    # current = Io * np.sum( icurrs, axis=(1, 2) )

    adjusted = list(zip(ispikes[pn], ispikers[pn]))


    for pn in range(weights.shape[0]):
        current[pn] = Io * sum([
                weights[spiker][pn] * \
                (np.exp(-(t-spike-delays[spiker][pn])*delta/tau) - np.exp(-(t-spike-delays[spiker][pn])*delta/taus)) * \
                int(t-spike-delays[spiker][pn] > 0)
            for spike, spiker in adjusted
            ])

    # import pdb; pdb.set_trace()

    return current

def get_current_due_to_postspikes_no_delay(weights, ispikes, ispikers, t):

    current = np.zeros(weights.shape[0])

    # wadj = np.reshape(weights.transpose(), (N, N, 1))
    # dadj = np.reshape(delays.transpose(), (N, N, 1))

    # fspikes = (t - np.arange(M) - dadj) * ispikes
    # icurrs = wadj * ( np.exp( -fspikes*delta/tau ) - np.exp( -fspikes*delta/taus ) ) * (fspikes > 0).astype(int)
    # current = Io * np.sum( icurrs, axis=(1, 2) )

    # frame = int(tau/delta)

    for pn in range(weights.shape[0]):

        adjusted = list(zip(ispikes[pn], ispikers[pn]))[-20:]
        # adjusted = [ (spk, spkr) for spk, spkr in zip(ispikes[pn], ispikers[pn]) if spk < t and t-frame < spk]
        # print(len(adjusted), end=" ")

        current[pn] = Io * sum([
                weights[spiker][pn] * \
                (np.exp(-(t-spike)*delta/tau) - np.exp(-(t-spike)*delta/taus))
            for spike, spiker in adjusted
            ])

    # import pdb; pdb.set_trace()

    return current

def get_current_due_to_postspikes_vec(weights, delays, ispikes, t):

    current = np.zeros(weights.shape[0])

    wadj = np.reshape(weights.transpose(), (N, N, 1))
    dadj = np.reshape(delays.transpose(), (N, N, 1))

    fspikes = (t - np.arange(M) - dadj) * ispikes
    icurrs = wadj * ( np.exp( -fspikes*delta/tau ) - np.exp( -fspikes*delta/taus ) ) * (fspikes > 0).astype(int)
    current = Io * np.sum( icurrs, axis=(1, 2) )

    # for pn in range(weights.shape[0]):
    #     current[pn] = Io * sum([
    #             weights[spiker][pn] * \
    #             (np.exp(-(t-spike-delays[spiker][pn])*delta/tau) - np.exp(-(t-spike-delays[spiker][pn])*delta/taus)) * \
    #             int(t-spike-delays[spiker][pn] > 0)
    #         for spike, spiker in zip(ispikes[pn], ispikers[pn])
    #         ])

    # import pdb; pdb.set_trace()

    return current

def main():

    print("Input Current Calculation")
    current = get_initial_poisson_currents(N)

    print("Network Connections")
    weights, delays = get_connectivity_matrices(N)

    V = initalize_V(N, M)

    spikes = [ [] for i in range(N) ]

    ispikes = [ [] for i in range(N) ]
    ispikers = [ [] for i in range(N) ]

    # post, pre, time of spike
    # ispikes = np.zeros((N, N, M))

    refraction = np.zeros((N))
    tmask = (weights != 0).astype(int)

    # import pdb; pdb.set_trace()

    print("Simulation")
    for i in range(1, M):
        if i%50 == 0:
            print("Step:", i)
        # print("Step:", i)

        # starta = timeit.timeit()

        # update main current with current generated by spikes
        # current[:, i] += get_current_due_to_postspikes(weights, delays, ispikes, ispikers, i)
        current[:, i] += get_current_due_to_postspikes_no_delay(weights, ispikes, ispikers, i)
        # current[:, i] += get_current_due_to_postspikes_vec(weights, delays, ispikes, i)

        # enda = timeit.timeit()
        # print("curr", enda-starta)

        # startv = timeit.timeit()
        # update voltage with correct value
        # with refractory period updates and spikes
        Vi, refraction, spikers = get_voltage_ref_spikes(
            V[:, i-1], 
            current[:, i-1:i+1], 
            refraction=refraction, 
            )
        
        # endv = timeit.timeit()
        # print("volt", endv-startv)

        V[:, i] = Vi

        # import pdb; pdb.set_trace()

        # startr = timeit.timeit()
        # transmit spikes to post neurons
        # ispikes[:, :, i] = np.transpose(tmask*spikers)


        # print("spiker", np.where(spikers)[0])
        # print("spikes", spikes)
        for spiker in np.where(spikers)[0]:
            spikes[spiker].append(i)

        for post in range(N):
            rev_spikes = tmask[:, post]*spikers
            rev_spikers = np.where(rev_spikes)[0]
            # print(post, np.sum(rev_spikes))

            ispikes[post].extend([i+delays[spiker][post] for spiker in rev_spikers])
            ispikers[post].extend(rev_spikers)

        # endr = timeit.timeit()

        # if i > 420:
        #     import pdb; pdb.set_trace()

        # if i > 500:
        #     print("curr, volt, rev: %f %f %f" % (enda-starta, endv-startv, endr-startr))

        # import pdb; pdb.set_trace()

        # print(ispikes, ispikers)

            # # for pre in range(N):
            #     if weights[pre][post] > 0:
            #         ispikes[post].extend(spikes[pre])
            #         ispikers[post].extend([pre]*len(spikes[pre]))


    print(spikes)
    plt.eventplot(spikes)
    plt.savefig("P2.png")
    # for i in range(3):
    #     plot_curr_and_resp(current[i], V[i], "P2.%d.png" % i)

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