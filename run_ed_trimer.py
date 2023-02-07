'''
Christian Bunker
M^2QM at UF
January 2023

Toy model of molecule with itinerant electrons
solved in time-dependent DMRG (approximate many body QM) method in transport/tddmrg
'''

from dmrg_utils import *

from transport import fci_mod
#from transport.fci_mod import ops_dmrg

import numpy as np
import matplotlib.pyplot as plt

#from pyblock3.algebra.mpe import MPE

# top level
verbose = 3;
np.set_printoptions(precision = 4, suppress = True);

##################################################################################
#### toy model of molecule

# phys params, must be floats
tm = 1.0; # hopping within molecule
chiral_breaking = -1e-4;
Bmol = -2*(0.001); # B field * gfactor / Bohr magneton
Belec = 1*Bmol; # B field * gfactor / Bohr magneton
JH = 0.0;
JK = 0.5; 

# electrons
n_mols = 3; # number of magnetic molecules
s_mols = 1/2; # spin of the mols
n_elecs = 1;
n_loc_dof = int((2**n_elecs)*((2*s_mols+1)**n_mols));
n_elecs = (n_elecs,0); # spin blind

#### hamiltonian
n_sys_orbs = n_mols; # Don't change! # molecular orbs in the sys
n_fer_orbs = n_loc_dof*n_sys_orbs; # total fermionic orbs

#### hamiltonian
hilbert_size = n_loc_dof**n_fer_orbs;
bdims = 500*n_fer_orbs**2*np.array([1.0,1.2,1.4,1.6,1.8]);
bdims = list(bdims.astype(int));
noises = [2e-1,1e-1,2e-2,1e-2,1e-5];
h_arr = get_h1e(n_mols,s_mols,n_sys_orbs,tm, Bmol, Belec,JH,JK,chiral_breaking,verbose=verbose);
if(verbose): print("1. Hamiltonian\n-h1e = \n");
if(verbose>1): print_H_alpha(h_arr);
              
#### exact soln
if(verbose): print("4. Exact solution");
h_arr_2d = fci_mod.mat_4d_to_2d(h_arr);
eigvals, eigvecs = np.linalg.eigh(h_arr_2d);
E0, psi0 = eigvals[0], eigvecs[:,0].T
if(verbose):
    print("-exact gd state energy:",E0);
    print("-exact gd state:\n",psi0);

# chirality
chiral_arr = get_chiral_op(n_mols, s_mols,n_sys_orbs);
chiral_arr = fci_mod.mat_4d_to_2d(chiral_arr);
chiral_exp = np.dot(np.conj(psi0),np.dot(chiral_arr,psi0));
print("-<\chi> = ",chiral_exp);

# site occupancy
occ_vals = np.zeros((n_sys_orbs),dtype=complex);
for a in range(n_sys_orbs):
    occ_arr = get_occ(n_loc_dof, n_sys_orbs, a);
    occ_vals[a] = np.dot(np.conj(psi0),np.dot(fci_mod.mat_4d_to_2d(occ_arr),psi0));
    print("-<occ["+str(a)+"]>",occ_vals[a]);

# site electron spin
sigz_vals = np.zeros((n_sys_orbs),dtype=complex);
for a in range(n_sys_orbs):
    sigz_arr = get_sigz(n_loc_dof, n_sys_orbs, a);
    sigz_vals[a] = np.dot(np.conj(psi0),np.dot(fci_mod.mat_4d_to_2d(sigz_arr),psi0));
    print("-<sigma_z["+str(a)+"]>",sigz_vals[a]);

# spin-spin correlation
SaSb_vals = np.zeros((n_mols,n_mols),dtype=complex);
for a in range(n_mols):
    for b in range(n_mols):
        if(a < b or True):
            SaSb_arr = get_SaSb(n_mols,s_mols,n_sys_orbs,a,b);
            if(verbose > 3): print(SaSb_arr[0,0][::2,::2]);
            SaSb_exp = np.dot(np.conj(psi0),np.dot(fci_mod.mat_4d_to_2d(SaSb_arr),psi0));
            if(verbose): print("-< S_"+str(a)+" S_"+str(b)+"> = ",SaSb_exp);
            SaSb_vals[a,b] = SaSb_exp;
print("- \sum_ab <S_a S_b> = ",np.sum(np.ravel(SaSb_vals)));

SaSb_sum = np.zeros_like(SaSb_arr);
for a in range(n_mols):
    for b in range(n_mols):
        if(a < b or True):
            SaSb_sum += get_SaSb(n_mols,s_mols,n_sys_orbs,a,b);
SaSb_sum_exp = np.dot(np.conj(psi0),np.dot(fci_mod.mat_4d_to_2d(SaSb_sum),psi0));
print("- <\sum_ab S_a S_b> = ",SaSb_sum_exp);

# spin-fermion correlation
SaSigb_vals = np.zeros((n_mols,n_sys_orbs),dtype=complex);
for a in range(n_mols):
    for b in range(n_sys_orbs):
        if(a <= b or True):
            SaSigb_arr = get_SaSigb(n_mols,s_mols,n_sys_orbs,a,b);
            if(verbose > 3): print(SaSigb_arr[a,b][::2,::2]);
            SaSigb_arr = fci_mod.mat_4d_to_2d(SaSigb_arr);
            SaSigb_exp = np.dot(np.conj(psi0),np.dot(SaSigb_arr,psi0));
            if(verbose): print("-< S_"+str(a)+" \sigma_"+str(b)+"> = ",SaSigb_exp);
            SaSigb_vals[a,b] = SaSigb_exp;
print("-F = ",np.sum(np.ravel(SaSigb_vals)));

# total spin
Stot2_exp = 0.25; # elec spin magnitude
# add in individual spin magnitudes
for a in range(n_mols):
    Stot2_exp += SaSb_vals[a,a];
# add in unique spin-spin correlations
for a in range(n_mols):
    for b in range(n_mols):
        if(a < b):
            Stot2_exp += 2*SaSb_vals[a,b];
# add in spin-fermion correlations
for a in range(n_mols):
    for b in range(n_sys_orbs):
        Stot2_exp += 2*SaSigb_vals[a,b];
print("- <S_tot^2> = ", Stot2_exp);


