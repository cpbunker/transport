'''
Christian Bunker
M^2QM at UF
July 2021

td_dmrg.py

Use Huanchen Zhai's DMRG code (pyblock3) to do time dependence in SIAM

Combine with fci code later
'''

import ops_dmrg

import numpy as np
from pyblock3 import hamiltonian, fcidump
from pyblock3.algebra.mpe import MPE

##########################################################################################################
#### compute observables

def compute_obs(op,mps):
    '''
    Compute expectation value of observable repped by given operator from MPS wf
    op must be an MPO
    '''

    return np.dot(mps.conj(), op @ mps)/np.dot(mps.conj(),mps);

    
##########################################################################################################
#### time propagation

def kernel(mpo, h_obj, mps, tf, dt, i_dot, bdims, verbose = 0):
    '''
    Drive time prop for dmrg
    Use real time time dependent dmrg method outlined here:
    https://pyblock3.readthedocs.io/en/latest/Documentation/rttddmrg.html

    Args:
    - mpo, a matrix product operator form of the hamiltonian
    - h_obj, a pyblock3.hamiltonian.Hamiltonian form of the hailtonian
    '''

    # check inputs
    assert(isinstance(bdims, list));

    # unpack
    norbs = mps.n_sites
    N = int(tf/dt+1e-6); # num steps
    n_generic_obs = 7; # 7 are time, E, 4 J's, concurrence
    sites = np.array(range(norbs)).reshape(int(norbs/2),2); # all indices, sep'd by site
    mpe_obj = MPE(mps, mpo, mps); # init mpe obj
    # return vals
    observables = np.zeros((N+1, n_generic_obs+4*len(sites) ), dtype = complex ); # generic plus occ, Sx, Sy, Sz per site

    # mpos for observables
    obs_mpos = [];
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jup(i_dot, norbs)[0] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jup(i_dot, norbs)[1] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jdown(i_dot, norbs)[0] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jdown(i_dot, norbs)[1] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.spinflip(i_dot, norbs) ) );
    for site in sites: # site specific observables
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.occ(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sx(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sy(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sz(site, norbs) ) );

    # loop over time
    for i in range(N+1):

        if(verbose>2): print("    time: ", i*dt);

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        energies = mpe_obj.tddmrg(bdims,-np.complex(0,dt), n_sweeps = 1, iprint=0, cutoff = 0).energies
        mpst = mpe_obj.ket; # update wf

        # compute observables
        observables[i,0] = i*dt; # time
        observables[i,1] = energies[-1]; # energy
        for mi in range(len(obs_mpos)): # iter over mpos
            print(mi);
            observables[i,mi+2] = compute_obs(obs_mpos[mi], mpst);
        
        # before any time stepping, get initial state
        if(i==0):
            # get site specific observables at t=0 in array where rows are sites
            initobs = np.real(np.reshape(observables[i,n_generic_obs:],(len(sites), 4) ) );

    # return time and tuple of observables as functions of time, 1d arrays
    return initobs, observables;


##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
