'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 3:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
import itertools

import sys
import time

#### top level
#np.set_printoptions(precision = 4, suppress = True);
plt.style.use("seaborn-dark-palette");
verbose = 5;
kalims = (0, np.pi/4);

#### setup

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets32 = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]]; # total spin 3/2 subspace
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# tight binding params
tl = 1; # lead hopping, in Hartree

# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209/Ha2meV; # convert to hartree
Jz = 0.124/Ha2meV;
DO = 0.674/Ha2meV;
DT = 0.370/Ha2meV;
An = 0.031/Ha2meV;

# initialize source vector
sourcei = 16; # |down, 3/2, 3/2 >
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(len(dets));
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
if(verbose): print("\nSource:\n"+source_str);
if(verbose): print(" - Checking that source is an eigenstate when JK's = 0");
h1e_JK0, g2e_JK0 = wfm.utils.h_dimer_2q((Jx, Jx, Jz, DO, DT, An, 0, 0)); 
hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets52);
hSR_JK0 = wfm.utils.entangle(hSR_JK0, 0, 1);
# do incident potential energy later

# initialize pair
pair = (1,4); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

# sweep over JK
start = time.time()
JKreson = (4/5)*(DO - (3/4)*Jx + (3/4)*Jz);
print(JKreson);

for JK in [5*DO]: # np.linspace(JKreson*(1-0.25), JKreson*(1+0.25),7):

    # physics of scattering region -> array of [H at octo, H at tetra]
    hblocks, tblocks = [], []; # lists to hold on site and hopping blocks in the SR

    # kondo interaction terms
    JKO, JKT = JK, JK
    params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
    OmegaR = np.power((np.sqrt(6)/4)*(JKO + JKT),2) + np.power(3*Jx/4-3*Jz/4-(1/2)*(DO+DT)+(5/8)*(JKO + JKT),2);
    OmegaR = np.sqrt(OmegaR)*tl;
    print(OmegaR)
    print("\nRabi period = ", 2*np.pi/OmegaR/2);
    
    # construct second quantized ham
    h1e, g2e = wfm.utils.h_dimer_2q(params); 

    # construct h_SR (determinant basis)
    hSR = fci_mod.single_to_det(h1e, g2e, species, states);
    hSR = wfm.utils.entangle(hSR, *pair);
    if(verbose > 4):
        hSR_sub = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
        print("\nUnentangled hamiltonian\n", hSR_sub);
        hSR_sub = wfm.utils.entangle(hSR_sub, 0, 1);
        print("\nEntangled hamiltonian\n", hSR_sub);

    # interaction as gaussian centered at N0 with char length a
    NSR = 9; # total num sites in SR
    N0 = 5; # site where interaction is peaked
    hblocks = [np.zeros_like(hSR)]; # mu_LL = 0
    tblocks = [-tl*np.eye(np.shape(hSR)[0]) ]; # LL to SR hopping
    for N in range(1,NSR+1):
        pref = np.exp(-np.power(N-N0,2)); # gaussian of char length a
        hblocks.append(pref*np.copy(hSR));
        tblocks.append(-tl*np.eye(np.shape(hSR)[0]) ); # inter SR hopping
    hblocks.append(np.zeros_like(hSR) );
    hblocks = np.array(hblocks);
    tblocks = np.array(tblocks);

    # iter over energy
    #Energy = -1.9999*tl;
    Elims = 0.001,0.01; # in meV
    Evals = np.linspace(*Elims, 10);
    Tvals = [];
    for Energy in Evals:
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, (Energy/Ha2meV)-2*tl, source));
    Tvals = np.array(Tvals);
    
    # first plot is just source and entangled pair
    fig, axes = plt.subplots(2, sharex = True);
    axes[0].set_title( "$t_l$ = "+str(tl)+" Ha, $J_K$ = "+str(int(100*Ha2meV*JK)/100)+" meV");
    axes[0].scatter(Evals,Tvals[:,sourcei], marker = 's', label = "$|i>$");
    axes[0].scatter(Evals,Tvals[:,pair[0]], marker = 's', label = "$|+>$");
    axes[0].scatter(Evals,Tvals[:,pair[1]], marker = 's', label = "$|->$");
    axes[0].legend(loc = 'upper right');
    
    # second plot is contamination
    contamination = np.zeros_like(Tvals[:,0]);
    for contami in range(len(dets)):
        if((contami not in pair) and (dets[contami][0] != dets[sourcei][0])):
            contamination += Tvals[:, contami];
    contamination = contamination/(contamination+Tvals[:,pair[0]]); 
    axes[1].scatter(Evals, contamination, marker = 's', color = "grey");
    
    # format
    for ax in axes:
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    axes[1].set_xlabel("$E$ (meV)");
    axes[0].set_ylabel("$T$");
    axes[1].set_ylabel("Contamination");
    plt.show();

#### save data
fname = "dat/dimer/"+str(source_str[1:-1]);
print("Saving data to "+fname);
np.save(fname, np.append(Tvals, Evals) );
stop = time.time();
print("Elapsed time = ", (stop - start)/60, " minutes.");




