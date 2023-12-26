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
    prop = np.exp(complex(0,-dt*eigvals)); # propagator operator in eigenbasis
    final_eig = prop*init_eig;

    # back to original basis
    origvecs = np.eye(len(init));
    final = np.zeros_like(init,dtype=complex);
    for veci in range(len(init)):
        final[veci] = np.dot( np.conj(origvecs[veci]), final_eig);

    return final;

def snapshot(state):
    '''
    '''

    pass;

def main(ham, init_state, time_snap, time_N):
    '''
    '''

    eigvals, eigvecs = solver(ham);

    # time prop with observables
    state = np.copy(init_state);
    for time_stepi in range(time_N):
        state = propagator(state, time_snap, eigvals, eigvecs);
        snapshot(state);

    return;
        
    

#### top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
case = sys.argv[1];

# physical parameters
tl = 1.0;
Jsd = 0.5;

# hamiltonian
H = np.array([[
    ]]);


















