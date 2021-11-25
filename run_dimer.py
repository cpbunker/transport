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
import itertools

import sys
import time

#### setup

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 3;
sourcei = int(sys.argv[1]);
param_dev = 0.4; # % params can deviate from ab initio vals in grid sweep
numEvals = 9; # energy pts, -2 < E < -1

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]]

# initialize source vector
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(len(dets));
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
print("\nSource:\n"+source_str);

# tight binding params
tl = 1.0; # 2e hopping, in meV
JK1 = 0.3;
JK2 = JK1;

# Ab initio params, in meV:
Jx = 0.209;
Jy = Jx; 
Jz = 0.124;
DO = 0.674;
DT = 0.370;
An = 0.031;
abinit_params = Jx, Jz, DO, DT, An, JK1; # package

#### get data for all entangled state pairs, across physical param space sweep

# save all features
# source, *pair, *params, peak vs N
features = [];

# sweep over entangled pairs
start = time.time()
for pair in wfm.utils.sweep_pairs(dets, sourcei):

    if(verbose):
        print("\nEntangled pair:");
        pair_strs = [];
        for pi in pair:
            pair_str = "|";
            for si in dets[pi]: pair_str += state_strs[si];
            pair_str += ">";
            print(pair_str);
            pair_strs.append(pair_str);

    # sweep over energy
    for Energy in [-1.0]: # keep in quadratic regime

        # sweep over physical params
        for params in wfm.utils.sweep_param_space(abinit_params, param_dev, numEvals):

            # truncate to 3d param space for now, remove later!
            Jx, Jz, DO, DT, An, JK1 = tuple(params);
            params = Jx, Jx, Jz, DO, DT, An, JK1, JK1;

            # construct second quantized ham
            h1e, g2e = wfm.utils.h_dimer_2q(params); 

            # construct h_SR (determinant basis) in a given total Sz subspace
            mystates, myis, mystrs = wfm.utils.subspace(3/2); # subspace
            h_SR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets);
            h_SR = wfm.utils.entangle(h_SR, *pair);
            if(verbose > 4): print("\n- Entangled hamiltonian\n", h_SR);

            # iter over N
            Nmax = 10;
            Nvals = np.linspace(0,Nmax,30,dtype = int);
            Tvals = [];
            ka = np.arccos(Energy/(-2*tl));
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

            # save features
            Tvals = np.array(Tvals);
            row = [Energy, *params]; # 9 actual physical features
            for si in dets[sourcei]: row.append(float(state_strs[si][:-1])); # source Sz's
            for pi in pair:
                for si in dets[pi]: row.append(float(state_strs[si][:-1])); # pair Sz's

            # save target
            peakplus = np.max(Tvals[:,pair[0]]);
            peakminus = np.max(Tvals[:,pair[1]]);
            row.append(peakplus);
            row.append(peakminus);
            row = np.array(row);
            features.append(np.copy(row));
            if(verbose):
                print("\n - Energy = ", Energy);
                print("    ",row[1:9]); # physical features
                print("    ",row[9:18]); # bookkeeping features
                print("    ",row[18:]); # targets

            # plot Tvals vs N
            if(verbose > 3):
                fig, ax = plt.subplots();
                ax.plot(Nvals,Tvals[:,sourcei], label = source_str);
                ax.plot(Nvals,Tvals[:,pair[0]], label = str(pair_strs[0])+"+"+str(pair_strs[1]));
                ax.plot(Nvals,Tvals[:,pair[0]], label = str(pair_strs[0])+"--"+str(pair_strs[1]));
                ax.minorticks_on();
                ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
                ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
                ax.set_xlabel("$N$");
                ax.set_ylabel("$T$");
                ax.set_title("Incident energy = "+str(Energy));
                plt.legend();
                plt.show();

#### save data
fname = "dat/"+str(source_str[1:-1]);
print("Saving data to "+fname);
assert( features != []);
np.save(fname, np.array(features) );
stop = time.time();
print("Elapsed time = ", (stop - start)/60, " minutes.");




