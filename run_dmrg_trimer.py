'''
Christian Bunker
M^2QM at UF
January 2023

Toy model of molecule with itinerant electrons
solved in time-dependent DMRG (approximate many body QM) method in transport/tddmrg
'''

from transport import fci_mod
from transport.fci_mod import ops_dmrg

import numpy as np
import matplotlib.pyplot as plt
import itertools

from pyblock3.algebra.mpe import MPE

# top level
verbose = 3;
np.set_printoptions(precision = 4, suppress = True);


##################################################################################
#### toy model of molecule

# phys params, must be floats
tm = 1.0; # hopping within molecule
gfactor = 2; # electron g factor
B_by_mu = 0.2; # B field / Bohr magneton

# electrons
m_mols = 1; # number of magnetic molecules
s_mols = 1/2; # spin of the mols
n_elecs = 1;
n_loc_dof = int((2**n_elecs)*((2*s_mols+1)**m_mols));
n_elecs = (n_elecs,0); # spin blind

#### hamiltonian
n_sys_orbs = 3; # = n_mols # molecular orbs in the sys
n_fer_orbs = n_loc_dof*n_sys_orbs; # total fermionic orbs

def get_hg(mytm, myB_mm, myB_elec):
    '''
    make the 1body and 2body parts of the 2nd qu'd ham
    '''
    assert n_elecs == (1,0);
    assert n_loc_dof % 2 == 0;
    h1e = np.zeros((n_fer_orbs, n_fer_orbs));
    g2e = np.zeros((n_fer_orbs,n_fer_orbs,n_fer_orbs,n_fer_orbs));

    # spin-independent hopping between n.n. sys orbs
    for sysi in range(0,n_loc_dof*n_sys_orbs-n_loc_dof,n_loc_dof):
        # iter over local dofs (up, down, etc)
        for loci in range(n_loc_dof):
            h1e[sysi+loci,sysi+loci+n_loc_dof] += -mytm; 
            h1e[sysi+loci+n_loc_dof,sysi+loci] += -mytm;
    if(m_mols > 2): # last to first hopping
        for loci in range(n_loc_dof): 
            h1e[n_loc_dof*n_sys_orbs-n_loc_dof+loci,loci] += -mytm; 
            h1e[loci,n_loc_dof*n_sys_orbs-n_loc_dof+loci] += -mytm;

    # Zeeman terms
    for sysi in range(0,n_fer_orbs,n_loc_dof):
        # have to iter over local dofs paticle-by-particle
        # first iter over all (2s+1)^M MM states
        mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
        mol_states = np.array([x for x in itertools.product(*(m_mols*(mol_projections,)))]);
        for mol_statei in range(len(mol_states)):
            # now iter over electron spin 
            for sigma in range(2):
                loci = 2*mol_statei+sigma;
                h1e[sysi+loci,sysi+loci] += myB_elec*(1/2-sigma) + myB_mm*sum(mol_states[mol_statei]);
                print(loci,1/2-sigma,mol_states[mol_statei],h1e[sysi+loci,sysi+loci])

    return h1e, g2e;


#### hamiltonian
hilbert_size = n_loc_dof**n_fer_orbs;
bdims = 5*n_fer_orbs**2*np.array([1.0,1.2,1.4]);
bdims = list(bdims.astype(int));
harr, garr = get_hg(tm, gfactor*B_by_mu, -gfactor*B_by_mu);
if(verbose): print("1. Hamiltonian\n-h1e = \n",harr);

#### DMRG solution
if(verbose): print("2. DMRG solution");

# MPS ansatz
h_obj, h_mpo, psi_init = fci_mod.arr_to_mpo(harr, garr, n_elecs, bdims[0]);
if verbose: print("- built H as compressed MPO: ", h_mpo.show_bond_dims() );
E_init = ops_dmrg.compute_obs(h_mpo, psi_init);
if verbose: print("- guessed gd energy = ", E_init);

# MPS ground state
dmrg_mpe = MPE(psi_init, h_mpo, psi_init);
# MPE.dmrg method controls bdims,noises, n_sweeps,conv tol (tol),verbose (iprint)
# noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
dmrg_obj = dmrg_mpe.dmrg(bdims=bdims, tol = 1e-8, iprint=-5);
if verbose: print("- variational gd energy = ", dmrg_obj.energies[-1]);

# MPS state -> observables
mps_occs = np.zeros((n_fer_orbs//2,));
mps_szs = np.zeros((n_fer_orbs//2,));
for orbi in range(0,n_fer_orbs,2):
    occ_mpo = h_obj.build_mpo(ops_dmrg.occ(np.array([orbi,orbi+1]), n_fer_orbs));
    sz_mpo  = h_obj.build_mpo(ops_dmrg.Sz( np.array([orbi,orbi+1]), n_fer_orbs));
    mps_occs[orbi//2] = ops_dmrg.compute_obs(occ_mpo,dmrg_mpe.ket);
    mps_szs [orbi//2] = ops_dmrg.compute_obs(sz_mpo, dmrg_mpe.ket);

print(mps_occs);
print(mps_szs);


