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

def kernel(mpo, h_obj, mps, tf, dt, i_dot, thyb, bdims, verbose = 0):
    '''
    Drive time prop for dmrg
    Use real time time dependent dmrg method outlined here:
    https://pyblock3.readthedocs.io/en/latest/Documentation/rttddmrg.html

    Args:
    - mpo, a matrix product operator form of the hamiltonian
    - h_obj, a pyblock3.hamiltonian.Hamiltonian form of the hailtonian
    '''

    assert(isinstance(bdims, list));

    N = int(tf/dt+1e-6); # num steps
    i_all = np.arange(0,h_obj.n_sites, 1, dtype = int);
    i_left = i_all[:i_dot[0] ];
    i_right = i_all[i_dot[-1]+1:];
    
    # return vals
    timevals = np.zeros(N+1);
    energyvals = np.zeros(N+1);
    currentvals = np.zeros((2, N+1));
    occvals = np.zeros( (3,N+1), dtype = complex );
    Szvals = np.zeros( (3,N+1), dtype = complex );

    # create observable MPOs outside loop to save time
    norbs = mps.n_sites
    current_mpo_up = h_obj.build_mpo(ops_dmrg.Jup(i_dot, norbs) );
    current_mpo_down = h_obj.build_mpo(ops_dmrg.Jdown(i_dot, norbs) );
    occ_mpo_L = h_obj.build_mpo(ops_dmrg.occ(i_left, norbs) );
    occ_mpo_D = h_obj.build_mpo(ops_dmrg.occ(i_dot, norbs) );
    occ_mpo_R = h_obj.build_mpo(ops_dmrg.occ(i_right, norbs) );
    Sz_mpo_L = h_obj.build_mpo(ops_dmrg.Sz(i_left, norbs) );
    Sz_mpo_D = h_obj.build_mpo(ops_dmrg.Sz(i_dot, norbs) );
    Sz_mpo_R = h_obj.build_mpo(ops_dmrg.Sz(i_right, norbs) );

    # init mpe
    mpe_obj = MPE(mps, mpo, mps);

    # loop over time
    for i in range(N+1):

        # before any time stepping, get initial state
        if(i==0):
            occ_init, Sz_init = np.zeros(len(i_all), dtype = complex), np.zeros(len(i_all), dtype = complex);
            for sitej in i_all: # iter over sites
                
                if(sitej % 2 == 0): # spin up sites
                    occ_init[sitej] = compute_obs(h_obj.build_mpo(ops_dmrg.occ([sitej,sitej+1], norbs) ), mps);
                    Sz_init[sitej] = compute_obs(h_obj.build_mpo(ops_dmrg.Sz([sitej,sitej+1], norbs) ), mps);
                else: # spin down sites
                    occ_init[sitej] = 0.0
                    Sz_init[sitej] = 0.0;

            initstatestr = "\nInitial state:"
            initstatestr += "\n    occ = "+str(np.real(occ_init));
            initstatestr += "\n    Sz = "+str(np.real(Sz_init));

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        energies = mpe_obj.tddmrg(bdims,-np.complex(0,dt), n_sweeps = 1, iprint=0, cutoff = 0).energies
        mpst = mpe_obj.ket; # update wf

        # compute observables
        timevals[i] = i*dt;
        energyvals[i] = energies[-1];
        currentvals[0][i] = -np.imag(compute_obs(current_mpo_up, mpst) );
        currentvals[1][i] = -np.imag(compute_obs(current_mpo_down, mpst) );
        occvals[0][i] = compute_obs(occ_mpo_L, mpst);
        occvals[1][i] = compute_obs(occ_mpo_D, mpst);
        occvals[2][i] = compute_obs(occ_mpo_R, mpst);
        Szvals[0][i] = compute_obs(Sz_mpo_L, mpst);
        Szvals[1][i] = compute_obs(Sz_mpo_D, mpst);
        Szvals[2][i] = compute_obs(Sz_mpo_R, mpst);
        
        # update stdout        
        if(verbose>2): print("    time: ", i*dt);

    # return time and tuple of observables as functions of time, 1d arrays
    observables = [timevals, energyvals, thyb*currentvals[0], thyb*currentvals[1], occvals[0], occvals[1], occvals[2], Szvals[0], Szvals[1], Szvals[2]];
    return initstatestr, np.array(observables);


##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
