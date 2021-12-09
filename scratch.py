'''
'''

import numpy as np
from pyscf import gto, scf, fci

#### system inputs

# hamiltonian params must be floats
epsilon1 = -2.0; # on site energy, site 1
epsilon2 = -1.0; # site 2
t = 0.0 # hopping
U = 5.0 # hubbard repulsion strength
if(verbose):
    print("\nInputs:\nepsilon1 = ",epsilon1,"\nepsilon2 = ",epsilon2,"\nt = ",t,"\nU = ",U);

#### solution

# ASU = put all electrons as up and make p,q... spin orbitals
nelecs = (1,0); # put in all spins as spin up
norbs = 2;      # spin orbs ie 1alpha, 1beta, 2alpha, 2beta -> 0,1,2,3
nroots = 6;

# implement h1e and h2e
# doing it this way forces floats which is a good fail safe
h1e = np.zeros((norbs,norbs));
g2e = np.zeros((norbs,norbs,norbs,norbs));

# on site energy
h1e[0,0] = epsilon1;
h1e[1,1] = epsilon2;

# 2 particle terms: hubbard
g2e[0,0,1,1] = U;
g2e[1,1,0,0] = U;  # interchange particle labels

# scf obj
Pa = np.zeros(norbs)
Pa[::2] = 1.0
Pa = np.diag(Pa)
print(Pa);
mol = gto.M(); # geometry is meaningless
mol.incore_anyway = True
mol.nelectron = sum(nelecs)
mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
scf_inst = scf.UHF(mol)
scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
scf_inst._eri = g2e # put h2e into scf solver
if( nelecs == (1,0) ):
    scf_inst.kernel(); # no dm
else:
    scf_inst.kernel(dm0=(Pa, Pa));

# fci on scf obj
cisolver = fci.direct_uhf.FCISolver(mol);

# slater determinant coefficients
mo_a = scf_inst.mo_coeff[0]
mo_b = scf_inst.mo_coeff[1]

# since we are in UHF formalism, need to split all hams by alpha, beta
# but since everything is spin blind, all beta matrices are zeros
h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
h1e_b = np.zeros_like(h1e_b);
h2e_aa = ao2mo.incore.general(scf_inst._eri, (mo_a,)*4, compact=False)
h2e_aa = h2e_aa.reshape(norbs,norbs,norbs,norbs)
h2e_ab = np.zeros_like(h2e_aa)
h2e_bb = np.zeros_like(h2e_aa)
h1e_tup = (h1e_a, h1e_b)
h2e_tup = (h2e_aa, h2e_ab, h2e_bb)

# run kernel to get exact energy
E_fci, v_fci = cisolver.kernel(h1e_tup, h2e_tup, norbs, nelecs, nroots = nroots)

E_formatter = "{0:6.6f}"
if(verbose):
    
    print("\n2. All spin up formalism: nelecs = ",nelecs, " nroots = ",nroots);
    print(" - E = ", E_asu);
    print(cisolver.mo_coeff)
    if False:
        # SCF
        molo, scfo = fci_mod.arr_to_scf(h1e_asu, g2e_asu, norbs, nelecs);
        E_scf, v_scf = fci_mod.scf_FCI(molo, scfo, nroots = nroots, verbose = verbose);
                                       
        # fnd Sx, Sz for each state
        Sxop, Szop = ops.Sx([0,1,2,3],norbs), ops.Sz([0,1,2,3], norbs);
        Sxeris, Szeris = ops.ERIs(Sxop, np.zeros((norbs,norbs,norbs,norbs)),scfo.mo_coeff), ops.ERIs(Szop, np.zeros((norbs,norbs,norbs,norbs)),scfo.mo_coeff);
        sferis = ops.ERIs(np.zeros((norbs, norbs)), ops.spinflip([0,1,2,3],norbs), scfo.mo_coeff);

        # info for each eigvec
        for vi in range(len(v_scf)):
            v = v_scf[vi]
            ciobj = ops.CIObject(v, norbs, nelecs);
            Eform = E_formatter.format(E_asu[vi]);
            Sxval = ops.compute_energy( *ciobj.compute_rdm12(), Sxeris);
            Szval = ops.compute_energy( *ciobj.compute_rdm12(), Szeris);
            Concur = abs(ops.compute_energy( *ciobj.compute_rdm12(), sferis ));
            print("- E = ",Eform, ", <Sx> = ",Sxval," <Sz> = ",Szval, " Concur = ",Concur );
            if(verbose > 2):
                print(v);
            
            
######################################################################
#### use dmrg to solve hubbard
assert(False);

import os
import pickle
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ

# top level inputs
bond_dim_i = 200;

if(verbose): print("\n3. DMRG (All spin up): nelecs = ",nelecs, " nroots = ",nroots);

# store hamiltonian matrices in fcidump
# syntax: point group, num MOs, total num elecs (int), 2S = na - nb, h1e, g2e
# I use ASU formalism so MOs are spin orbs
hdump = fcidump.FCIDUMP(pg = 'd2h', n_sites = norbs, n_elec = sum(nelecs), twos = nelecs[0] - nelecs[1], h1e = h1e_asu, g2e = g2e_asu)
if verbose: print("- Created fcidump");

# get hamiltonian from fcidump
# now instead of np arrays it is a pyblock3 Hamiltonian class
h = hamiltonian.Hamiltonian(hdump, True);
h_mpo = h.build_qc_mpo(); # hamiltonian as matrix product operator (DMRG lingo)
#mpo, error = h_mpo.compress(flat = True, left=True, cutoff=1E-9, norm_cutoff=1E-9)
if verbose: print("- Built H as MPO");

# initial ansatz and energy
psi_mps = h.build_mps(bond_dim_i); # multiplies as np array
if verbose: 
    print('MPO = ', h_mpo.show_bond_dims())
    print('MPS = ', psi_mps.show_bond_dims())
psi_sq = np.dot(psi_mps.conj(), psi_mps);
E_psi = np.dot(psi_mps.conj(), h_mpo @ psi_mps)/psi_sq; # initial exp value of energy
print("- Initial gd energy = ", E_psi);

# ground-state DMRG
# runs thru an MPE (matrix product expectation) class built from mpo, mps
MPE_obj = MPE(psi_mps, h_mpo, psi_mps);

# solve system by doing dmrg sweeps
# MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
bonddims = [bond_dim_i,bond_dim_i+100,bond_dim_i+200]; # increase
noises = [1e-4,1e-5,0]; # slowly turn off. limits num sweeps if shorter than bdims, but not vice versa
# can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
dmrg_obj = MPE_obj.dmrg(bdims=bonddims, noises=noises, iprint = 1);
E_dmrg = dmrg_obj.energies;
print("- Final gd energy = ", E_dmrg[-1]);
print("- Final energies = ", E_dmrg);

        

