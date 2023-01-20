'''
Christian Bunker
M^2QM at UF
July 2021

td_dmrg.py

Use Huanchen Zhai's DMRG code (pyblock3) to do time dependence in SIAM

Combine with fci code later
'''
from transport import fci_mod
from transport.fci_mod import ops_dmrg

import numpy as np

try:
    from pyblock3 import hamiltonian, fcidump
    from pyblock3.algebra.mpe import MPE
except:
    pass;

import time

    
##########################################################################################################
#### time propagation

#def kernel(mpo, h_obj, mps, tf, dt, bdims, verbose = 0):
def kernel(h1e, g2e, nelecs, bdims):
    '''
    Drive time prop for dmrg
    Use real time time dependent dmrg method outlined here:
    https://pyblock3.readthedocs.io/en/latest/Documentation/rttddmrg.html

    Args:
    -mpo, a matrix product operator form of the hamiltonian
    -h_obj, a pyblock3.hamiltonian.Hamiltonian form of the hamiltonian
    -mps, a matrix product state
    -tf, float, the time to end the time evolution at
    -dt, float, the time step of the time evolution
    -bdims, list of ints, bond dimension of the DMRG solver
    '''
    if(not isinstance(h1e, np.ndarray)): raise TypeError;
    if(not isinstance(g2e, np.ndarray)): raise TypeError;
    if(not isinstance(nelecs, tuple)): raise TypeError;
    if(not isinstance(bdims, np.ndarray)): raise TypeError;

    # convert to matrix product form
    mpe_obj = fci_mod.arr_to_mpe(h1e, g2e, nelecs, bdims[0]);


    # unpack
    norbs = mps.n_sites
    nsteps = int(tf/dt+1e-6); # num steps
    sites = np.array(range(norbs)).reshape(int(norbs/2),2); # index as sites[spatial index, spin index]
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

    # return tuple of observables at t=0 and observables as arrays vs time
    return initobs, observables;


##########################################################################################################
#### utils

def compute_obs(op,mps):
    '''
    Compute expectation value of observable repped by given operator from MPS wf
    op must be an MPO
    '''

    return np.dot(mps.conj(), op @ mps)/np.dot(mps.conj(),mps);



