#!/usr/bin/python3

import numpy as np
import pdb as pdb
import pickle
import os

import matplotlib.pyplot as plt

import timeit

from lif import initalize_V, get_voltage_ref_spikes

np.random.seed(3)
os.system("taskset -p 0xff %d" % os.getpid())

# learning parameters
Aup = 0.01
Adown = -0.02
_tauL = 20e-3

iI = 50e-9

Io = 1e-12
tau = 15e-3
taus = tau/4

T = 1000e-3
# T = 100e-3

delta = 0.1e-3
# delta = 1e-3

tauL = _tauL // delta
M = int(T/delta)

alpha_tau = None
def pre_compute_alpha_tau(tau, taus):
    global alpha_tau
    if alpha_tau is None:
        ts = np.arange(M)
        alpha_tau = np.exp(-ts*delta/tau) - np.exp(-ts*delta/taus)
    
def fn_alpha_tau(t):
    return alpha_tau[int(t)]

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

def get_initial_poisson_currents(N):

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

    _, spikes = get_spike_train(25)

    current = np.zeros((N, M))
    for i in range(25):
    # for i in range(int(5)):
        current[i] = get_current_through_synapse(spikes[i])

    return current

def get_connectivity_matrices_P1(N):
    ''' returns weights and delays in indices
    '''

    weights = np.zeros((N, N))
    delays = np.zeros((N, N))

    w0 = 3000

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

def get_connectivity_matrices(N, we, wi):
    ''' returns weights and delays in indices
    '''

    e8 = int(0.8*N)
    e2 = int(0.2*N)
    e1 = int(0.1*N)

    weights = np.zeros((N, N))
    delays = np.zeros((N, N))

    for i in range(N):
        if i < e8:
            choice = np.random.choice(N, e1, replace=False)
            weights[i, choice] = we
        else:
            choice = np.random.choice(e8, e1, replace=False)
            weights[i, choice] = -wi

    delays[:e8, :] = np.random.randint(low=1, high=int(20e-3/delta), size=(e8, N))
    delays[e8:, :] = int(1e-3/delta)

    return weights, delays

def get_current_due_to_postspikes(weights, ispikes, ispikers, t):

    current = np.zeros(weights.shape[0])

    frame = int(tau/delta)

    for pn in range(weights.shape[0]):

        # adjusted = [ (spk, spkr) for spk, spkr in zip(ispikes[pn], ispikers[pn]) if spk < t ]
        adjusted = [ (spk, spkr) for spk, spkr in zip(ispikes[pn], ispikers[pn]) if spk < t and t-frame < spk]

        # adjusted = adjusted[-20:]

        current[pn] = Io * sum([
                weights[spiker][pn] * \
                fn_alpha_tau(t-spike)
            for spike, spiker in adjusted
            ])

    # import pdb; pdb.set_trace()

    return current

def simulate_network_and_learn(V, current, weights, delays):
    V = V.copy()
    current = current.copy()
    weights = weights.copy()

    N = weights.shape[0]
    e8 = int(0.8*N)

    weights_tr = np.zeros((M, N, N))
    weights_tr[0, :, :] = weights[:, :]

    spikes = [ [] for i in range(N) ]
    refraction = np.zeros((N))

    ispikes = [ [] for i in range(N) ]
    ispikers = [ [] for i in range(N) ]

    print("Simulation")
    for i in range(1, M):
        if i%50 == 0:
            print("Step:", i)

        current[:, i] += get_current_due_to_postspikes(weights, ispikes, ispikers, i)

        # update voltage with correct value
        # with refractory period updates and spikes
        Vi, refraction, spikers = get_voltage_ref_spikes(
            V[:, i-1], 
            current[:, i-1:i+1], 
            refraction=refraction, 
            )
        
        V[:, i] = Vi
        for spiker in np.where(spikers)[0]:
            spikes[spiker].append(i)

        # if np.sum(spikers)!=0:
        #     print("spikers:", spikers)

        # learning rule application only for excitatory neurons
        for spiker in np.where(spikers)[0]:
            # only first e8
            if spiker >= e8:
                continue

            # for upstream neurons
            r_ispikes = [j for j in ispikes[spiker] if j < i][::-1]
            r_ispikers = ispikers[spiker][:len(r_ispikes)][::-1]
            upstreamers = tmask[:, spiker]
            for upstreamer in np.where(upstreamers)[0]:
                try:
                    tlast = r_ispikes[r_ispikers.index(upstreamer)]
                    weights[upstreamer, spiker] += \
                        weights[upstreamer, spiker] * Aup * np.exp(-(i-tlast)/tauL)
                except ValueError as e:
                    pass

            # for downstream neurons
            downstreamers = tmask[spiker, :]
            for downstreamer in np.where(downstreamers)[0]:
                tlast = spikes[downstreamer][-1:]
                if len(tlast) != 0:
                    weights[spiker, downstreamer] += \
                        weights[spiker, downstreamer] * Adown * np.exp(-(i+delays[spiker, downstreamer]-tlast[0])/tauL)
                else:
                    pass
        # weight tracking
        weights_tr[i, :, :] = weights[:, :]


        # transmission of spikes forward
        tmask = (weights != 0).astype(int)
        for post in range(N):
            rev_spikes = tmask[:, post]*spikers
            rev_spikers = np.where(rev_spikes)[0]
            # print(post, np.sum(rev_spikes))

            # if np.sum(rev_spikers) != 0:
            #     print(post, rev_spikers)
            #     print("+delays:", [i+delays[spiker][post] for spiker in rev_spikers])

            ispikes[post].extend([i+delays[spiker][post] for spiker in rev_spikers])
            ispikers[post].extend(rev_spikers)

        # print("ispikes:", ispikes)
        # print("ispikers:", ispikers)

    return V, current, spikes, weights_tr

def main():

    N = 200
    w0 = 3000
    # N = 50

    print("Pre compute alpha_tau")
    pre_compute_alpha_tau(tau, taus)

    print("Input Current Calculation")
    # current, _ = get_initial_external_currents(N)
    current = get_initial_poisson_currents(N)

    print("Network Connections")
    iweights, delays = get_connectivity_matrices(N, w0, -w0)
    # iweights, delays = get_connectivity_matrices_P1(N)

    V = initalize_V(N, M)

    V, current, spikes, aweights = simulate_network_and_learn(V, current, iweights, delays)

    with open("P4.spikes.pickle", "wb") as f:
        pickle.dump([spikes, aweights], f)
    with open("P4.spikes.pickle", "rb") as f:
        spikes, aweights = pickle.load(f)

    plot_spikes(spikes, "P4.0.png")
    plot_nspikes_t(spikes, "P4.1.png", "P4.2.png")
    plot_weights_trace(aweights, "P4.3.png")
    # for i in range(N):
        # plot_curr_and_resp(current[i], V[i], "P2.ex.%d.png" % i)

def plot_spikes(spikes, filen):

    plt.figure()
    plt.eventplot(spikes)
    plt.savefig(filen)

def plot_nspikes_t(spikes, filen1, filen2):

    e8 = int(0.8*len(spikes))
    window = int(10e-3/delta)

    espikes = np.zeros((M))
    for rspikes in spikes[:e8]:
        espikes[rspikes] += 1.0
        
    ispikes = np.zeros((M))
    for rspikes in spikes[e8:]:
        ispikes[rspikes] += 1.0

    redata = np.convolve(espikes, np.ones(window), mode="valid")
    ridata = np.convolve(ispikes, np.ones(window), mode="valid")

    plt.figure()
    plt.plot(np.arange(redata.shape[0])*delta, redata)
    plt.savefig(filen1)

    plt.figure()
    plt.plot(np.arange(ridata.shape[0])*delta, ridata)
    plt.savefig(filen2)

def plot_weights_trace(weights, filen):

    data = np.mean(weights, axis=(1, 2))

    plt.figure()
    plt.plot(np.arange(data.shape[0])*delta, data)
    plt.savefig(filen)

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