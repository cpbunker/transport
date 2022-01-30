'''
'''

from transport import fci_mod, wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# dft params
# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209/Ha2meV; # convert to hartree
Jz = 0.124/Ha2meV;
DO = 0.674/Ha2meV;
DT = 0.370/Ha2meV;
An = 0;

# iter over JK
JKvals = DO*np.array(range(1,2));
for JK in JKvals:

    # 3 state ham
    h1e, g2e = wfm.utils.h_dimer_2q((Jx, Jx, Jz, DO, DT, An, 0, 0));
    hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest=dets52);
    hSR = wfm.utils.entangle(hSR, 0, 1);

    # diagonalize
    leadEs, Udiag = np.linalg.eigh(hSR);
    for coli in range(len(leadEs)): print(np.real(Udiag.T[coli]), Ha2meV*leadEs[coli]);
    hSR_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR, Udiag));
    print("diag hamiltonian\n",Ha2meV*np.real(hSR_diag)); # Udiag -> lead eigenstate basis
       

