'''
Christian Bunker
M^2QM at UF
January 2023

Toy model of molecule with itinerant electrons
solved with time-dependent DMRG (approximate many body QM) method in transport/tddmrg
'''

from transport import tddmrg
from transport.tddmrg import utils

from transport import fci_mod
from transport.fci_mod import ops_dmrg

import numpy as np
import matplotlib.pyplot as plt

# top level
solve_dmrg = True; # whether to do DMRG solution
verbose = 1;
np.set_printoptions(precision = 4, suppress = True);

# set up molecule geometry
from pyscf import gto, scf, fci
coords = [["O", (0,0,0)],["H",(0,0,1.92)]];
basis = "sto-3g"; # minimal
X_mol = gto.M(atom=coords, basis=basis, unit="bohr", spin = 1); # mol geometry
if(verbose):
    print("Basis = ");
    for label in X_mol.ao_labels(): print(label);
X_scf = scf.RHF(X_mol).run();
print("\n\n\n");

h, g = fci_mod.scf_to_arr(X_mol, X_scf)


##################################################################################
#### toy model of molecule

# phys params, must be floats
tm = 1.0; # hopping within molecule
chiral_breaking = -1e-4;
Bmol = -2*(0.001); # B field * gfactor / Bohr magneton
Belec = 1*Bmol; # B field * gfactor / Bohr magneton
JH = -0.2;
JK = 0.0; 

# electrons
n_mols = 3; # number of magnetic molecules
s_mols = 1/2; # spin of the mols
n_elecs = 1; # this is what Kumar uses, also the only one I can currently implement
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
h_arr = utils.get_h1e(n_mols,s_mols,n_sys_orbs,tm, Bmol, Belec,JH,JK,chiral_breaking,verbose=verbose);
if(verbose): print("1. Hamiltonian\n-h1e = \n");
if(verbose>1): utils.print_H_alpha(h_arr);

#### DMRG solution
if(solve_dmrg):
    if(verbose): print("2. DMRG solution");
    from pyblock3.algebra.mpe import MPE
    if(sum(n_elecs) != 1): raise NotImplementedError;

    # MPS ansatz
    h_arr = fci_mod.mat_4d_to_2d(h_arr);
    g_arr = np.zeros((len(h_arr),len(h_arr),len(h_arr),len(h_arr)),dtype=h_arr.dtype);
    h_obj, h_mpo, psi_init = fci_mod.arr_to_mpo(h_arr, g_arr, n_elecs, bdims[0]);
    if verbose: print("- built H as compressed MPO: ", h_mpo.show_bond_dims() );
    E_init = ops_dmrg.compute_obs(h_mpo, psi_init);

    # MPS ground state
    dmrg_mpe = MPE(psi_init, h_mpo, psi_init);
    # MPE.dmrg method controls bdims, noises, n_sweeps, max iterations, tolerance
    # noise typically needs to be high
    dmrg_obj = dmrg_mpe.dmrg(bdims=bdims,noises=noises,iprint=-5);

    #### MPS state -> observables
    mps_occs = np.zeros((n_fer_orbs//2,));
    mps_szs = np.zeros((n_fer_orbs//2,));
    for orbi in range(0,n_fer_orbs,2):
        occ_mpo = h_obj.build_mpo(ops_dmrg.occ(np.array([orbi,orbi+1]), n_fer_orbs));
        sz_mpo  = h_obj.build_mpo(ops_dmrg.Sz( np.array([orbi,orbi+1]), n_fer_orbs));
        mps_occs[orbi//2] = ops_dmrg.compute_obs(occ_mpo,dmrg_mpe.ket);
        mps_szs [orbi//2] = ops_dmrg.compute_obs(sz_mpo, dmrg_mpe.ket);
            
    if verbose:
        print("3. Observables");
        print("- guessed gd energy = ", E_init);
        print("- variational gd energy = ", dmrg_obj.energies[-1]);
        print("-MPS site occ:",mps_occs);
        print("-MPS site Sz:",mps_szs);

    # spin-spin correlation
    SaSb_vals = np.zeros((n_mols,n_mols),dtype=complex);
    for a in range(n_mols):
        for b in range(n_mols):
            if(a != b):
                SaSb_arr = utils.get_SaSb(n_mols,s_mols,n_sys_orbs,a,b);
                if(verbose > 3): utils.print_H_alpha(SaSb_arr);
                SaSb_arr = fci_mod.mat_4d_to_2d(SaSb_arr);
                SaSb_exp = np.dot(mps_occs,np.dot(SaSb_arr,mps_occs));
                if(verbose): print("-< S_"+str(a)+" S_"+str(b)+"> = ",SaSb_exp);
                SaSb_vals[a,b] = SaSb_exp;
                
#### exact soln
if(verbose): print("4. Exact solution");
h_arr_2d = fci_mod.mat_4d_to_2d(h_arr);
eigvals, eigvecs = np.linalg.eigh(h_arr_2d);
E0, psi0 = eigvals[0], eigvecs[:,0].T
if(verbose):
    print("-exact gd state energy:",E0);
    print("-exact gd state:\n",psi0);

# chirality
chiral_arr = utils.get_chiral_op(n_mols, s_mols,n_sys_orbs);
chiral_arr = fci_mod.mat_4d_to_2d(chiral_arr);
chiral_exp = np.dot(np.conj(psi0),np.dot(chiral_arr,psi0));
print("-<\chi> = ",chiral_exp);

# site occupancy
occ_vals = np.zeros((n_sys_orbs),dtype=complex);
for a in range(n_sys_orbs):
    occ_arr = utils.get_occ(n_loc_dof, n_sys_orbs, a);
    occ_vals[a] = np.dot(np.conj(psi0),np.dot(fci_mod.mat_4d_to_2d(occ_arr),psi0));
    print("-<occ["+str(a)+"]>",occ_vals[a]);

# site electron spin
sigz_vals = np.zeros((n_sys_orbs),dtype=complex);
for a in range(n_sys_orbs):
    sigz_arr = utils.get_sigz(n_loc_dof, n_sys_orbs, a);
    sigz_vals[a] = np.dot(np.conj(psi0),np.dot(fci_mod.mat_4d_to_2d(sigz_arr),psi0));
    print("-<sigma_z["+str(a)+"]>",sigz_vals[a]);

# spin-spin correlation
SaSb_vals = np.zeros((n_mols,n_mols),dtype=complex);
for a in range(n_mols):
    for b in range(n_mols):
        if(a < b):
            SaSb_arr = utils.get_SaSb(n_mols,s_mols,n_sys_orbs,a,b);
            if(verbose > 3): print(SaSb_arr[0,0][::2,::2]);
            SaSb_arr = fci_mod.mat_4d_to_2d(SaSb_arr);
            SaSb_exp = np.dot(np.conj(psi0),np.dot(SaSb_arr,psi0));
            if(verbose): print("-< S_"+str(a)+" S_"+str(b)+"> = ",SaSb_exp);
            SaSb_vals[a,b] = SaSb_exp;
print("-D = ",np.sum(np.ravel(SaSb_vals)));

# spin-fermion correlation
SaSigb_vals = np.zeros((n_mols,n_sys_orbs),dtype=complex);
for a in range(n_mols):
    for b in range(n_sys_orbs):
        if(a <= b or True):
            SaSigb_arr = utils.get_SaSigb(n_mols,s_mols,n_sys_orbs,a,b);
            if(verbose > 3): print(SaSigb_arr[a,b][::2,::2]);
            SaSigb_arr = fci_mod.mat_4d_to_2d(SaSigb_arr);
            SaSigb_exp = np.dot(np.conj(psi0),np.dot(SaSigb_arr,psi0));
            if(verbose): print("-< S_"+str(a)+" \sigma_"+str(b)+"> = ",SaSigb_exp);
            SaSigb_vals[a,b] = SaSigb_exp;
print("-F = ",np.sum(np.ravel(SaSigb_vals)));


