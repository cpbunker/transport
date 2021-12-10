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
tl = 0.005; # lead hopping, in Hartree
th = 0.004; # SR hybridization
td = 0.003;  # hopping between imps
epsO = -0.5; # octahedral Co onsite energy
epsT = -1.0; # tetrahedral Co onsite energy

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
h1e, g2e = wfm.utils.h_dimer_2q((Jx, Jx, Jz, DO, DT, An, 0, 0)); 
_ = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest=[dets[sourcei]]);
assert False;

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
for JK in [JKreson]: # np.linspace(JKreson*(1-0.25), JKreson*(1+0.25),7):

    # physics of scattering region -> array of [H at octo, H at tetra]
    hblocks, tblocks = [], []; # lists to hold on site and hopping blocks in the SR
    for Coi in range(2):

        # define all physical params
        JKO, JKT = 0, 0;
        if Coi == 0: JKO = JK; # J S dot sigma is onsite only
        else: JKT = JK;
        params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
        params = 1, 0, 0, 0, 0, 0, 0, 0
        
        # construct second quantized ham
        h1e, g2e = wfm.utils.h_dimer_2q(params); 

        # construct h_SR (determinant basis)
        h_SR = fci_mod.single_to_det(h1e, g2e, species, states);
        h_SR = wfm.utils.entangle(h_SR, *pair);
        if(verbose > 4):
            h_SR_sub = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = [dets[sourcei]]);
            print("\nUnentangled hamiltonian\n", h_SR_sub);
            h_SR_sub = wfm.utils.entangle(h_SR_sub, 0, 1);
            print("\nEntangled hamiltonian\n", h_SR_sub);

        # hopping between sites
        V_SR = td*np.eye(np.shape(h_SR)[0])
        
        # add to blocks list
        hblocks.append(np.copy(h_SR));
        if(Coi > 0): tblocks.append(np.copy(V_SR));

    # get data
    assert False
    kavals, Tvals = wfm.Data(source, hblocks, tblocks, th, tl, kalims )

    # plot Tvals vs N
    if(verbose > 3):

        # first plot is just source and entangled pair
        fig, axes = plt.subplots(2, sharex = True);
        axes[0].set_title(" $t_l$ = "+str(tl));
        axes[0].scatter(kavals/np.pi,Tvals[:,sourcei], marker = 's', label = "$|i>$");
        axes[0].scatter(kavals/np.pi,Tvals[:,pair[0]], marker = 's', label = "$|+>$");
        axes[0].scatter(kavals/np.pi,Tvals[:,pair[1]], marker = 's', label = "$|->$");
        axes[0].legend(loc = 'upper right');
        
        # second plot is contamination
        contamination = np.zeros_like(Tvals[:,0]);
        for contami in range(len(dets)):
            if((contami not in pair) and (dets[contami][0] != dets[sourcei][0])):
                contamination += Tvals[:, contami];
        contamination = contamination/(contamination+Tvals[:,pair[0]]); 
        axes[1].scatter(kavals/np.pi, contamination, marker = 's', color = "grey");
        
        # format
        for ax in axes:
            ax.minorticks_on();
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
            ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        axes[0].set_xlabel("$ka/\pi$");
        axes[0].set_ylabel("$T$");
        axes[1].set_ylabel("Contamination");
        plt.show();

#### save data
fname = "dat/dimer/"+str(source_str[1:-1]);
print("Saving data to "+fname);
np.save(fname, np.append(Tvals, kavals) );
stop = time.time();
print("Elapsed time = ", (stop - start)/60, " minutes.");




