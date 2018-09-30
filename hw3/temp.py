#!/usr/bin/python3

import numpy as np
import pdb as pdb
import os

import matplotlib.pyplot as plt

def main():

    spikes = [ 2, 3, 1, 4, 0, 3, 2 ]

    plt.plot(np.arange(len(spikes))*0.001, spikes)

    plt.show()

if __name__ == '__main__':
    main()