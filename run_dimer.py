'''
Christian Bunker
M^2QM at UF
October 2021

Steady state transport of a single electron through a one dimensional wire
Part of the wire is scattering region, where the electron spin degrees of
freedom can interact with impurity spin degrees of freedom

Impurity hamiltonians calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

import sys

#### run code

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
sztotal = 5/2; # specify total sz subspace
sourcei = int(sys.argv[1]);
param_dev = 0.5; # how much params can deviate from ab initio vals in grid sweep

# initialize source vector
source = np.zeros(3)
source[sourcei] = 1;

# tight binding params
tl = 1.0; # 2e hopping, in meV
JK1 = 0.4;
JK2 = JK1;

# Ab initio params, in meV:
Jx = 0.209;
Jy = Jx; 
Jz = 0.124;
DO = 0.674;
DT = 0.370;
An = 0.031;
An = 0; # for convenience
abinit_params = Jx, Jy, Jz, DO, DT, An, JK1, JK2; # package

#######################################################################################
#### get data for all entangled state pairs, across physical param space sweep

# sweep entangled pairs
for pair in wfm.utils.sweep_pairs(source):

    # sweep physical params
    for params in wfm.utils.sweep_param_space(abinit_params, param_dev):

        #### set up system

        # construct second quantized ham
        h1e, g2e = wfm.utils.h_dimer_2q(abinit_params); 

        # def particles and their single particle states
        species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
        states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz

        # construct h_SR (determinant basis) in a given total Sz subspace
        mystates, myis, mystrs = wfm.utils.subspace(sztotal); # subspace
        h_SR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = mystates);
        print("Test workflow in the SzT = 5/2 subspace");
        print(h_SR);
        h_SR = wfm.utils.entangle(h_SR, *pair);
        print(h_SR);

        #### iter over energy

        # sweep from bottom of band up
        Evals = np.linspace(-2*tl, -2*tl + 0.6*tl, 20);

        # package h, t block matrices
        hblocks = np.array([np.zeros_like(h_SR), h_SR, np.zeros_like(h_SR)]);
        tblocks = np.array([-tl*np.eye(*np.shape(h_SR)),-tl*np.eye(*np.shape(h_SR))]);

        # get coefs at each energy
        Tvals = [];
        for Ei in range(len(Evals) ):
            Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Evals[Ei], source));

        # plot Tvals vs E
        Tvals = np.array(Tvals);
        fig, ax = plt.subplots();
        xlab = "$E+2t_l $"
        for pi in range(len(myis)):
            ax.scatter(Evals + 2*tl,Tvals[:,pi], marker = 's',label = mystrs[pi]);

        # format and plot
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        ax.set_xlabel(xlab);
        ax.set_ylabel("$T$");
        #ax.set_title("source = "+mysource+", $J_{K} = $"+str(JK));
        plt.legend();
        plt.show();
        del fig, ax, hblocks, tblocks, Tvals;                                              

        #### iter over N

        # fix energy near bottom of band
        Energy = -2*tl + 0.5;
        ka = np.arccos(Energy/(-2*tl));

        # iter over N
        Nmax = 10;
        Nvals = np.linspace(0,Nmax,30,dtype = int);
        Tvals = [];
        for N in Nvals:

            # package as block hams 
            # number of blocks depends on N
            hblocks = [np.zeros_like(h_SR)]
            tblocks = [-tl*np.eye(*np.shape(h_SR)) ];
            for Ni in range(N):
                hblocks.append(np.copy(h_SR));
                tblocks.append(-tl*np.eye(*np.shape(h_SR)) );
            hblocks.append(np.zeros_like(h_SR) );
            hblocks = np.array(hblocks);
            tblocks = np.array(tblocks);

            # coefs
            Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

        # plot Tvals vs E
        Tvals = np.array(Tvals);
        fig, ax = plt.subplots();
        xlab = "$N$"
        for pi in range(len(myis)):
            ax.plot(Nvals,Tvals[:,pi], label = mystrs[pi]);

        # format and plot
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        ax.set_xlabel(xlab);
        ax.set_ylabel("$T$");
        #ax.set_title("source = "+mysource+", $J_{K} = $"+str(JK));
        plt.legend();
        plt.show();




