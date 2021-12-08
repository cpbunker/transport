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
#np.set_printoptions(precision = 4, suppress = True);
plt.style.use("seaborn-dark-palette");
verbose = 5;
sourcei = int(sys.argv[1]);

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets32 = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]]; # total spin 3/2 subspace
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# initialize source vector
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(len(dets));
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
print("\nSource:\n"+source_str);

# tight binding params
tl = 0.01; # in Hartree, s.t. a = a0 = Bohr radius = 0.529 Angstrom

# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209/Ha2meV; # convert to hartree
Jz = 0.124/Ha2meV;
DO = 0.674/Ha2meV;
DT = 0.370/Ha2meV;
An = 0.031/Ha2meV;

#### get data for all entangled state pairs, across physical param space sweep

# save all features
# source, *pair, *params, peak vs N
features = [];

# sweep over entangled pairs
start = time.time()
for pair in [(1,4)]: #wfm.utils.sweep_pairs(dets, sourcei)[:1]:

    if(verbose):
        print("\nEntangled pair:");
        pair_strs = [];
        for pi in pair:
            pair_str = "|";
            for si in dets[pi]: pair_str += state_strs[si];
            pair_str += ">";
            print(pair_str);
            pair_strs.append(pair_str);

    # sweep over incident wavevector
    for ka in [0.1]:
        Energy = -2*tl*np.cos(ka);

        # sweep over JK
        JKreson = (4/5)*(DO - (3/4)*Jx + (3/4)*Jz);
        for JK in [JKreson]: # np.linspace(JKreson*(1-0.25), JKreson*(1+0.25),7):

            # define all physical params
            DT = DO;
            params = Jx, Jx, Jz, DO, DT, An, JK, JK;
            
            # construct second quantized ham
            h1e, g2e = wfm.utils.h_dimer_2q(params); 

            # construct h_SR (determinant basis)
            h_SR = fci_mod.single_to_det(h1e, g2e, species, states);
            h_SR = wfm.utils.entangle(h_SR, *pair);
            if(verbose > 4):
                print("\n - Energy = ", Energy + 2*tl);
                print(" - Barrier = ", abs(np.array(params)));
                h_SR_sub = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                h_SR_sub = wfm.utils.entangle(h_SR_sub, 0, 1);
                print("\n- Entangled hamiltonian\n", h_SR_sub);

            # iter over N
            Nmax = 50;
            Nvals = np.linspace(0,Nmax,min(Nmax+1,30),dtype = int);
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

            # plot Tvals vs N
            if(verbose > 3):

                # first plot is just source and entangled pair
                fig, axes = plt.subplots(2, sharex = True);
                axes[0].set_title("$E$ = "+str(int(100*Energy/tl)/100)+"$t_l, ka$ = "+str(int(100*ka)/100)+" $t_l$ = "+str(tl));
                axes[0].set_xlim(0,int(1.3*Nmax));
                axes[0].set_ylim(0,1.05);
                axes[0].scatter(Nvals,Tvals[:,sourcei], marker = 's', label = "$|d>$");
                axes[0].scatter(Nvals,Tvals[:,pair[0]], marker = 's', label = "$|+>$");
                axes[0].scatter(Nvals,Tvals[:,pair[1]], marker = 's', label = "$|->$");
                axes[0].legend(loc = 'upper right');
                
                # second plot is contamination
                contamination = np.zeros_like(Tvals[:,0]);
                for contami in range(len(dets)):
                    if((contami not in pair) and (dets[contami][0] != dets[sourcei][0])):
                        contamination += Tvals[:, contami];
                contamination = contamination/(contamination+Tvals[:,pair[0]]); 
                axes[1].scatter(Nvals, contamination, marker = 's', color = "grey");
                
                # format
                for ax in axes:
                    ax.minorticks_on();
                    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
                    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
                axes[-1].set_xlabel("$N$");
                axes[0].set_ylabel("$T$");
                axes[1].set_ylabel("Contamination");
                plt.show();

#### save data
assert( features != []);
fname = str(source_str[1:-1]);
print("Saving data to "+fname);
np.save(fname, np.array(features) );
stop = time.time();
print("Elapsed time = ", (stop - start)/60, " minutes.");




