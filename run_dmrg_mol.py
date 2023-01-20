'''
Christian Bunker
M^2QM at UF
January 2023

Toy model of molecule with itinerant electrons
solved in time-dependent DMRG (approximate many body QM) method in transport/tddmrg
'''

from transport import tddmrg

import numpy as np
import matplotlib.pyplot as plt

import sys

# top level
verbose = 3;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

##################################################################################
#### toy model of molecule

# phys params, must be floats
tm = 1.0; # hopping within molecule
th = 0.8; # hopping on/off molecule
Vg = -4*tm; # gate voltage of substrate

# time info
timestep = 0.02
timefinal = 1.0;

# electrons
n_loc_dof = 2;
nelecs = 1;
nelecs = (nelecs,0); # spin blind

#### hamiltonian
n_sys_orbs = 2; # molecular orbs in the sys
n_sub_orbs = 1; # molecular orbs in the substrate
n_fer_orbs = n_loc_dof*n_sys_orbs+n_loc_dof*n_sub_orbs; # total fermionic orbs

def get_hg(mytm, myth, myVg):
    '''
    make the 1body and 2body parts of the 2nd qu'd ham
    '''
    h1e = np.zeros((n_fer_orbs, n_fer_orbs));
    g2e = np.zeros((n_fer_orbs,n_fer_orbs,n_fer_orbs,n_fer_orbs));

    # spin-independent hopping between sys orbs
    for sysi in range(0,n_loc_dof*n_sys_orbs-n_loc_dof,n_loc_dof):
        # iter over local dofs (up, down, etc)
        for loci in range(n_loc_dof):
            h1e[sysi+loci,sysi+loci+n_loc_dof] = -mytm; 
            h1e[sysi+loci+n_loc_dof,sysi+loci] = -mytm;

    # spin-independent hopping between sys and substrate
    for sysi in range(0,n_loc_dof*n_sys_orbs,n_loc_dof):
        for subi in range(n_loc_dof*n_sys_orbs,n_fer_orbs,n_loc_dof):
            # iter over local dofs (up, down, etc)
            for loci in range(n_loc_dof):
                h1e[sysi+loci,subi+loci] = -myth; # down hopping
                h1e[subi+loci,sysi+loci] = -myth;

    # substrate on-site
    for subi in range(n_loc_dof*n_sys_orbs,n_fer_orbs,n_loc_dof):
        # iter over local dofs (up, down, etc)
        for loci in range(n_loc_dof):
            h1e[subi+loci,subi+loci] = myVg;

    return h1e, g2e;


#### time evol
hilbert_size = n_loc_dof**n_fer_orbs;
bdims_init = n_fer_orbs**2;
bdims = bdims_init*np.array([1,1.2,1.4]);
bdims = list(bdims.astype(int));
harr, garr = get_hg(tm, 0, Vg); # initial state = no sub-sys hopping
harr_neq, _ = get_hg(tm, th, Vg); # add in sub-sys hopping
obs = tddmrg.kernel(harr, garr, harr_neq, nelecs, bdims,timefinal, timestep, verbose = verbose);

# plot
plt.plot(np.real(obs[:,0]), np.real(obs[:,2]));
plt.plot(np.real(obs[:,0]), np.real(obs[:,4]));
plt.plot(np.real(obs[:,0]), np.real(obs[:,6]));
plt.show();

