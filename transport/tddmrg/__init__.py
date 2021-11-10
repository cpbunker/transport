'''
Christian Bunker
M^2QM at UF
July 2021

td_dmrg.py

Use Huanchen Zhai's DMRG code (pyblock3) to do time dependence in SIAM

Combine with fci code later
'''

from transport import ops

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
    obs_mpos.append(h_obj.build_mpo(ops.dmrg.Jup(i_dot, norbs)[0] ) );
    obs_mpos.append(h_obj.build_mpo(ops.dmrg.Jup(i_dot, norbs)[1] ) );
    obs_mpos.append(h_obj.build_mpo(ops.dmrg.Jdown(i_dot, norbs)[0] ) );
    obs_mpos.append(h_obj.build_mpo(ops.dmrg.Jdown(i_dot, norbs)[1] ) );
    obs_mpos.append(h_obj.build_mpo(ops.dmrg.spinflip(i_dot, norbs) ) );
    for site in sites: # site specific observables
        obs_mpos.append( h_obj.build_mpo(ops.dmrg.occ(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops.dmrg.Sx(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops.dmrg.Sy(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops.dmrg.Sz(site, norbs) ) );

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
#### utils

def compute_obs(op,mps):
    '''
    Compute expectation value of observable repped by given operator from MPS wf
    op must be an MPO
    '''

    return np.dot(mps.conj(), op @ mps)/np.dot(mps.conj(),mps);


##########################################################################################################
#### wrappers

def Data(occs, nleads, h1e, g2e, tf, dt, bond_dims, noises, fname = "dmrg_custom.npy", verbose = 0):
    '''
    '''

    # check inputs
    assert( bond_dims[0] <= bond_dims[-1]); # checks bdims has increasing behavior and is list
    assert( noises[0] >= noises[-1] ); # checks noises has decreasing behavior and is list
    
    # set up
    hstring = time.asctime(); # for printing
    nelecs = (len(occs), 0);
    norbs = np.shape(h1e)[0]; # num spin orbs
    imp_i = [2*nleads[0],norbs - 2*nleads[1]];

    # initial state
    hinit = np.zeros_like(h1e);
    for i in occs:
        hinit[i,i] = -1e6;

    # ground state of init state ham
    # follows process explained in depth for h1e below
    hinit_obj = hamiltonian.Hamiltonian(fcidump.FCIDUMP(h1e=hinit,g2e=np.zeros_like(g2e),pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]), flat = True);
    hinit_mpo, _ = hinit_obj.build_qc_mpo().compress(cutoff=1e-15);
    hinit_mps = hinit_obj.build_mps(bond_dims[0]);
    hinit_MPE = MPE(hinit_mps, hinit_mpo, hinit_mps);
    hinit_MPE.dmrg(bdims=bond_dims, noises = noises, tol = 1e-8, iprint=0);
    psi_init = hinit_MPE.ket;

    # convert physics from array to MPE   
    h_obj = hamiltonian.Hamiltonian(fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]),flat=True);
    h_mpo = h_obj.build_qc_mpo().compress(cutoff=1E-15); # compressing saves memory
    if verbose: hstring += "\n- Built H as compressed MPO: "+str( h_mpo.show_bond_dims())

    # initial ansatz for wf, in matrix product state (MPS) form
    h_mps = h_obj.build_mps(bond_dims[0]);
    E_dmrg0 = compute_obs(h_mpo, h_mps);
    if verbose: hstring += "\n- Initial gd energy = "+str(E_dmrg0);

    # solve using ground-state DMRG which runs thru MPE class
    if(verbose): hstring += "\n2. DMRG solution";
    h_MPE = MPE(h_mps, h_mpo, h_mps);

    # solve system by doing dmrg sweeps
    # MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
    # can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
    # noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
    dmrg_obj = h_MPE.dmrg(bdims=bond_dims, noises = noises, tol = 1e-8, iprint=0);
    E_dmrg = dmrg_obj.energies;
    psi_mps = h_MPE.ket; # actual wf
    if verbose: hstring += "\n- Final gd energy = "+str(E_dmrg[-1]);
    if verbose: hstring += "\n- Final gd energy = "+str(compute_obs(h_mpo,psi_mps));
    
    # time propagate the init state
    # td dmrg uses highest bond dim
    if(verbose): hstring += "\n3. Time propagation";
    init, observables = kernel(h_mpo, h_obj, psi_init, tf, dt, imp_i, [bond_dims[-1]], verbose = verbose);

    # write results to external file
    hstring += "\ntf = "+str(timestop)+"\ndt = "+str(deltat)+"\nbdims = "+str(bond_dims)+"\nnoises = "+str(noises);
    hstring += "\n"+str(h1e);
    np.savetxt(fname[:-4]+".txt", init, header = hstring); # saves info to txt
    np.save(fname, observables);
    if(verbose): print("4. Saved data to "+fname);
    
    return; # end custom data data dmrg


