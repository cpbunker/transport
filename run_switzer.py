'''
Christian Bunker
M^2QM at UF
October 2021

Steady state transport of a single electron through a one dimensional wire
Part of the wire is scattering region, where the electron spin degrees of
freedom can interact with impurity spin degrees of freedom
In this case, impurities follow eric's paper
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

import itertools

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# tight binding params
tl = 1.0; # hopping >> other params

# Eric's params
D = -0.06
JH = -0.005;
JK1 = -0.04;
JK2 = -0.04;
JK = (JK1 + JK2)/2;
DeltaK = JK1 - JK2;

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4],[5,6,7]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.0_","0.0_","-1.0_","1.0_","0.0_","-1.0_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets32 = [[0,2,6],[0,3,5],[1,2,5]]; # total spin 3/2 subspace

# source
sourcei = 9; #| down, 1, 1 > = [1 2 5]
source = np.zeros(len(dets));
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
print("\nSource:\n"+source_str);

# entangled pair
pair = (1,3); #|up, 1, 0> = [0 2 6] and |up,0,1> = [0,3,5]
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

# fix energy near bottom of band
Energy = -2*tl + 0.5;
ka = np.arccos(Energy/(-2*tl));

# 2nd qu'd ham
h1e, g2e = ops.h_switzer(D, JH, JK1, JK2);

# convert to many body form
h_SR = fci_mod.single_to_det(h1e,g2e, species, states);

# entangle the me up states into eric's me, s12, m12> = up, 2, 1> state
h_SR = wfm.utils.entangle(h_SR, *pair);
if(verbose > 4):
    h_SR_32 = fci_mod.single_to_det(h1e,g2e, species, states, dets_interest=dets32);
    h_SR_32 = wfm.utils.entangle(h_SR_32, 0, 1);
    print("\n- Entangled hamiltonian\n", h_SR_32);

# iter over N
Nmax = 80
Nvals = np.linspace(1,Nmax,min(Nmax,20),dtype = int);
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
    Tvals.append(wfm.kernel(hblocks, tblocks, tl, Energy, source));

# plot
Tvals = np.array(Tvals);
fig, ax = plt.subplots();
ax.scatter(Nvals, Tvals[:,sourcei], marker = 's', label = "$|d>$");
ax.scatter(Nvals, Tvals[:,pair[0]], marker = 's', label = "$|+>$");
ax.scatter(Nvals, Tvals[:,pair[1]], marker = 's', label = "$|->$");

# format
#ax.set_title("Transmission at resonance, $J_K = 2D/3$");
ax.set_ylabel("$T$");
ax.set_xlabel("$N$");
ax.set_ylim(0.0,1.05);
plt.legend();
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
plt.show();






