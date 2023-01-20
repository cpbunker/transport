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

def kernel(mpo, h_obj, mps, tf, dt, i_dot, bdims, verbose = 0):
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
    -i_dot, int, the site index of the impurity
    =bdims, list of ints, bond dimension of the DMRG solver
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

def coefs(mps):
    '''
    Get a coefficient at each site
    '''

    return;

def arr_to_mpe(h1e, g2e, nelecs, bdim_i, cutoff = 1e-15):
    '''
    Convert physics contained in an FCIDUMP object or file
    to a MatrixProduct Expectation (MPE) for doing DMRG

    Args:
    fd, a pyblock3.fcidump.FCIDUMP object, or filename of such an object
    bdim_i, int, initial bond dimension of the MPE

    Returns:
    MPE object
    '''

    # unpack
    norbs = np.shape(h1e)[0];

    # convert arrays to fcidump
    fd = fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]);

    # convert fcidump to hamiltonian obj
    h_obj = hamiltonian.Hamiltonian(fd,flat=True);

    # from hamiltonian obj, build Matrix Product Operator
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff = cutoff);
    h_mps = h_obj.build_mps(bdim_i);

    # MPE
    return MPE(h_mps, h_mpo, h_mps);


##########################################################################################################
#### wrappers

def Data(source, leadsites, h1e, g2e, tf, dt, bond_dims, noises, fname = "dmrg_data.npy", verbose = 0):
    '''
    Wrapper for taking a system setup (geometry spec'd by leadsites, physics by
    h1e, g2e, and electronic config by source) and going through the entire
    tddmrg process.

    Args:
    source, list, spin orbs to fill with an electron initially
    leadsites, tuple of how many sites in left, right lead
    h1e, 2d arr, one body interactions
    g2e, 4d arr, two body interactions
    tf, float, time to stop propagation
    dt, float, time step for propagation
    bond_dims, list, increasing bond dimensions for DMRG energy minimization
    noises, list, decreasing noises for DMRG energy minimization
    '''

    # check inputs
    assert(np.shape(h1e) == np.shape(g2e)[:2]);
    assert( bond_dims[0] <= bond_dims[-1]); # checks bdims has increasing behavior and is list
    assert( noises[0] >= noises[-1] ); # checks noises has decreasing behavior and is list
    
    # set up
    hstring = time.asctime(); # for printing
    nelecs = (len(source), 0);
    norbs = np.shape(h1e)[0]; # num spin orbs
    imp_i = [2*leadsites[0],norbs - 2*leadsites[1]-1];

    # prep initial state
    hinit = -np.ones_like(h1e, dtype = float);
    for i in source:
        hinit[i,i] += -1e6;

    # initial mps = ground state of init state ham
    hinit_MPE = arr_to_mpe(hinit, np.zeros((norbs,)*4), nelecs, bond_dims[0]);
    psi_init = hinit_MPE.ket;

    # convert physics from array to MPE
    if(verbose): hstring += "\n1. DMRG solution";
    h_obj = hamiltonian.Hamiltonian(fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]),flat=True);
    h_mpo = h_obj.build_complex_qc_mpo(max_bond_dim=-5);
    h_mpo, _ = h_mpo.compress(cutoff=1e-15); # compressing saves memory
    if verbose: hstring += "\n- Built H as compressed MPO: "+str( h_mpo.show_bond_dims())
    # initial ansatz for wf, in matrix product state (MPS) form
    h_mps = h_obj.build_mps(bond_dims[0]);
    E_dmrg0 = compute_obs(h_mpo, h_mps);
    if verbose: hstring += "\n- Guess gd energy = "+str(E_dmrg0);

    # solve using ground-state DMRG which runs thru MPE class
    h_MPE = MPE(h_mps, h_mpo, h_mps);

    # solve system by doing dmrg sweeps
    # MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
    # can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
    # noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
    dmrg_obj = h_MPE.dmrg(bdims=bond_dims, noises = noises, tol = 1e-8, iprint=0);
    psi_mps = h_MPE.ket; # actual wf
    E_dmrg = compute_obs(h_mpo,psi_mps);
    if verbose: hstring += "\n- Actual gd energy = "+str(E_dmrg);
    
    # time propagate the init state
    # td dmrg uses highest bond dim
    if(verbose): hstring += "\n2. Time propagation";
    init, observables = kernel(h_mpo, h_obj, psi_init, tf, dt, imp_i, [bond_dims[-1]], verbose = verbose);

    # write results to external file
    hstring += "\ntf = "+str(tf)+"\ndt = "+str(dt)+"\nbdims = "+str(bond_dims)+"\nnoises = "+str(noises);
    hstring += "\n"+str(h1e);
    np.savetxt(fname[:-4]+".txt", init, header = hstring); # saves info to txt
    np.save(fname, observables);
    if(verbose): print("3. Saved data to "+fname);
    
    return; # end custom data data dmrg


