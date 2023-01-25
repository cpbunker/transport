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

from pyblock3 import hamiltonian, fcidump
from pyblock3.algebra.mpe import MPE

import time

    
##########################################################################################################
#### time propagation

#def kernel(mpo, h_obj, mps, tf, dt, bdims, verbose = 0):
def kernel(h1e, g2e, h1e_neq, nelecs, bdims, tf, dt, verbose = 0) -> np.ndarray:
    '''
    Drive time prop for dmrg
    Use real time time dependent dmrg method outlined here:
    https://pyblock3.readthedocs.io/en/latest/Documentation/rttddmrg.html

    Args:
    -h1e, ndarray, 1body 2nd qu'd ham
    -g2e, ndarray, 2body 2nd qu'd ham
    -h1e_neq, ndarray, 1body ham that drives time evol, e.g. turns on hopping
    -nelecs, tuple, number of up, down electrons
    -bdims, ndarray, bond dimension of the DMRG solver
    -tf, float, the time to end the time evolution at
    -dt, float, the time step of the time evolution
    '''
    if(not isinstance(h1e, np.ndarray)): raise TypeError;
    if(not isinstance(g2e, np.ndarray)): raise TypeError;
    if(not isinstance(nelecs, tuple)): raise TypeError;
    if(not isinstance(bdims, list)): raise TypeError;
    if(not bdims[0] <= bdims[-1]): raise ValueError; # bdims must have increasing behavior 

    # unpack
    if(verbose): print("1. Hamiltonian\n-h1e = \n",h1e);
    norbs = len(h1e); # n fermion orbs
    nsteps = 1+int(tf/dt+1e-6); # n time steps
    sites = np.array(range(norbs)).reshape(norbs//2,2); #sites[i] gives a list of fermion orb indices spanning spin space
    
    # convert everything to matrix product form
    if(verbose): print("2. DMRG solution");
    h_obj, h_mpo, psi_init = fci_mod.arr_to_mpo(h1e, g2e, nelecs, bdims[0]);
    if verbose: print("- built H as compressed MPO: ", h_mpo.show_bond_dims() );
    E_init = ops_dmrg.compute_obs(h_mpo, psi_init);
    if verbose: print("- guessed gd energy = ", E_init);

    # solve ham with DMRG
    dmrg_mpe = MPE(psi_init, h_mpo, psi_init);
    # MPE.dmrg method controls bdims,noises, n_sweeps,conv tol (tol),verbose (iprint)
    # noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
    dmrg_obj = dmrg_mpe.dmrg(bdims=bdims, tol = 1e-8, iprint=0);
    if verbose: print("- variational gd energy = ", dmrg_obj.energies[-1]);

    # return vals
    obs_gen = 2; # time, E
    obs_per_site = 2; # occ, Sz
    observables = np.empty((nsteps,obs_gen+obs_per_site*len(sites)), dtype=complex); 

    # mpos for observables
    obs_mpos = [];
    for site in sites: # site specific observables
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.occ(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sz(site, norbs) ) );

    # time evol
    _, h_mpo_neq, _ = fci_mod.arr_to_mpo(h1e_neq, g2e, nelecs, bdims[0]);
    dmrg_mpe_neq = MPE(psi_init, h_mpo_neq, psi_init); # must be built with initial state!
    if(verbose): print("3. Time evolution\n-h1e_neq = \n",h1e_neq);
    for ti in range(nsteps):
        if(verbose>2): print("-time: ", ti*dt);

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        E_t = dmrg_mpe_neq.tddmrg(bdims,-np.complex(0,dt), n_sweeps = 1, iprint=0, cutoff = 0).energies
        psi_t = dmrg_mpe_neq.ket; # update wf

        # compute observables
        observables[ti,0] = ti*dt; # time
        observables[ti,1] = E_t[-1]; # energy
        for mi in range(len(obs_mpos)): # iter over mpos
            observables[ti,obs_gen+mi] = ops_dmrg.compute_obs(obs_mpos[mi], psi_t);
        

    # site specific observables at t=0 in array where rows are sites
    initobs = np.real(np.reshape(observables[0,obs_gen:],(len(sites), obs_per_site)));
    print("-init observables:\n",initobs);
    
    # return observables as arrays vs time
    return observables;





