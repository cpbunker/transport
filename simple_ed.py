'''
Christian Bunker
M^2QM at UF
November 2022

Scattering from two spin-s MSQs
Want to make a SWAP gate
solved in time-independent QM using wfm method in transport/wfm
'''

import numpy as np
import matplotlib.pyplot as plt

import sys

#####################################################################
#### util functions

def solver(ham):
    '''
    '''
    eigvals, eigvecs = np.linalg.eigh(ham);
    eigvecs = eigvecs.T;
    return eigvals, eigvecs;

def propagator(init, dt, eigvals, eigvecs):
    '''
    '''

    # in eigenbasis
    init_eig = np.zeros_like(init,dtype=complex);
    for veci in range(len(init)):
        init_eig[veci] = np.dot( np.conj(eigvecs[veci]), init);
        
    # propagate
    prop = np.exp(eigvals*complex(0,-dt)); # propagator operator in eigenbasis
    final_eig = prop*init_eig;

    # back to original basis
    origvecs = np.eye(len(init));
    final = np.zeros_like(init,dtype=complex);
    for veci in range(len(init)):
        final[veci] = np.dot( np.conj(origvecs[veci]), final_eig);

    return final;

def snapshot(state,nsites,time):
    '''
    '''

    fig, ax = plt.subplots(2,sharex=True);
    occs = np.zeros((nsites,),dtype=float);
    szs = np.zeros((nsites,),dtype=float);
    for sitei in range(nsites):
        occ_ret = np.dot( np.conj(state), np.matmul(occ_op_kwarg[sitei],state));
        assert(np.imag(occ_ret) < 1e-10);
        occs[sitei] = np.real(occ_ret);
        sz_ret = np.dot( np.conj(state), np.matmul(sz_op_kwarg[sitei],state));
        assert(np.imag(sz_ret) < 1e-10);
        szs[sitei] = np.real(sz_ret);

    ax[0].scatter(np.arange(nsites),occs,color="darkblue",marker='o',linestyle="solid");
    ax[0].set_ylim(0,1.0);
    ax[0].set_ylabel("$\langle n \\rangle $");
    ax[1].scatter(np.arange(nsites),szs,color="darkblue",marker='o',linestyle="solid");
    ax[1].set_ylim(-0.5,0.5);
    ax[1].set_ylabel("$\langle s_z \\rangle $");
    ax[-1].set_xlabel("Site");
    ax[0].set_title("Time = {:.2f}".format(time));
    plt.tight_layout();
    plt.show();

def main(ham, nsites, init_state, time_snap, time_N):
    '''
    '''

    eigvals, eigvecs = solver(ham);

    # time prop with observables
    state = np.copy(init_state);
    snapshot(state,nsites,0.0);
    for time_stepi in range(1,time_N+1):
        state = propagator(state, time_snap, eigvals, eigvecs);
        snapshot(state,nsites,time_stepi*time_snap);

    return;
        
    

#### top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
case = sys.argv[1];

# physical parameters
L = 5; # num sites
tl = 1.0;
Jsd = 0.5;

# hamiltonian
H = np.zeros((L,L),dtype=float);
for i in range(L-1):
    H[i,i+1] += -tl;
    H[i+1,i] += -tl;

# occ op
occ_op_kwarg = np.array([
    [[1,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0]],
    [[0,0,0,0,0],
     [0,1,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0]],
    [[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,1,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0]],
    [[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,1,0],
     [0,0,0,0,0]],
    [[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,1]],
    ]);

sz_op_kwarg = np.copy(occ_op_kwarg) # wrong but I am lazy and doing spinless

# run
psii = np.array([1.0,0,0,0,0]);
print(">>> H =\n",H);
print(">>> psii = ",psii);
main(H,L,psii,0.5,4)















