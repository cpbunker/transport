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
th = 0.5; # hopping on/off molecule
Vg = -10*tm; # gate voltage of substrate

# time info
timestep = 0.1
timefinal = 1.0;

# electrons
nelecs = 1;
nelecs = (nelecs,0); # spin blind

#### hamiltonian
n_sys_orbs = 2; # molecular orbs in the sys
n_sub_orbs = 1; # molecular orbs in the substrate
n_fer_orbs = 2*n_sys_orbs+2*n_sub_orbs; # total fermionic orbs

def get_hg(mytm, myth, myVg)-> (np.ndarray, np.ndarray):
    '''
    make the 1body and 2body parts of the 2nd qu'd ham
    '''
    h1e = np.zeros((n_fer_orbs, n_fer_orbs));
    g2e = np.zeros((n_fer_orbs,n_fer_orbs,n_fer_orbs,n_fer_orbs));

    # spin-independent hopping between sys orbs
    for sysi in range(0,2*n_sys_orbs-2,2):
        h1e[sysi,sysi+2] = -mytm; # up hopping
        h1e[sysi+2,sysi] = -mytm;
        h1e[sysi+1,sysi+1+2] = -mytm; # down hopping
        h1e[sysi+1+2,sysi+1] = -mytm;

    # spin-independent hopping between sys and substrate
    for sysi in range(0,2*n_sys_orbs,2):
        for subi in range(2*n_sys_orbs,n_fer_orbs,2):
            h1e[sysi,subi] = -myth; # up hopping
            h1e[subi,sysi] = -myth;
            h1e[sysi+1,subi+1] = -myth; # down hopping
            h1e[subi+1,sysi+1] = -myth;

    # substrate on-site
    for subi in range(2*n_sys_orbs,n_fer_orbs,2):
        h1e[subi,subi] = myVg;
        h1e[subi+1,subi+1] = myVg;



#### time evol
hilbert_size = 2**n_fer_orbs;
bdims_init = n_fer_orbs**2;
bdims = bdims_init*np.array([1,1.2,1.4,1.6]);
bdims = bdims.astype(int);
obs = tddmrg.kernel(h1e, g2e, nelecs, bdims,timefinal, timestep, verbose = verbose);

# plot
timei, Ei = 0,1;
plt.plot(obs[:,timei], obs[:,Ei]);
plt.show();

