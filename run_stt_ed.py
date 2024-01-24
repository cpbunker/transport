'''
Christian Bunker
M^2QM at UF
November 2022

Scattering from two spin-s MSQs
Want to make a SWAP gate
solved in time-independent QM using wfm method in transport/wfm
'''

from transport import tdfci
from transport.tdfci import utils

import numpy as np
import matplotlib.pyplot as plt

import sys

#####################################################################
#### util functions

def snapshot(state,the_H,nsites,time, the_sites,plot=False):
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

    # printout
    print("Time = {:.2f}".format(time));
    print("Total energy (ED) = {:.6f}".format(np.dot(np.conj(state),np.matmul(the_H, state))));
    for sitei in range(len(the_sites)):
        print("Site {:.0f} <n> (ED) = {:.6f}".format(the_sites[sitei],occs[sitei])); 
        print("Site {:.0f} <Sz> (ED) = {:.6f}".format(the_sites[sitei],szs[sitei]));
    print("Summed occ (ED) = {:.6f}".format(np.sum(occs)));
    print("Summed sz (ED) = {:.6f}".format(np.sum(szs)));
    
    # plot and format data
    if(plot):
        ax[0].plot(np.arange(nsites),occs,color="darkblue",marker='o');
        ax[0].set_ylim(0,1.0);
        ax[0].set_ylabel("$\langle n \\rangle $");
        ax[1].plot(np.arange(nsites),szs,color="darkblue",marker='o');
        ax[1].set_ylim(-0.5,0.5);
        ax[1].set_ylabel("$\langle s_z \\rangle $");
        ax[-1].set_xlabel("Site");
        ax[0].set_title("Time = {:.2f}".format(time));
        plt.tight_layout();
        plt.show();

#####################################################################
#### wrapper code

def main(ham, nsites, init_state, time_snap, time_N, the_sites):
    '''
    wraps all the time evolution and printout steps
    '''

    eigvals, eigvecs = tdfci.solver(ham);

    # time prop with observables
    state = np.copy(init_state);
    snapshot(state,ham,nsites,0.0, the_sites);
    for time_stepi in range(1,time_N+1):
        state = tdfci.propagator(state, time_snap, eigvals, eigvecs);
        snapshot(state,ham,nsites,time_stepi*time_snap, the_sites);

    return;
        
#####################################################################
#### exec code

# top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
case = sys.argv[1];

if(case == "Ne1"): # 1 electron, 1D TB problem

    # physical parameters
    L = 20; # num sites
    Nconf = 2; # initial sites
    tl = 1.0;

    # hamiltonian
    H = np.zeros((L,L),dtype=float);
    for i in range(L-1):
        H[i,i+1] += -tl;
        H[i+1,i] += -tl;

    # occ op
    occ_op_kwarg = [];
    for sitei in range(L):
        nj = np.zeros_like(H);
        nj[sitei,sitei] = 1.0;
        occ_op_kwarg.append(nj);
    occ_op_kwarg = np.array(occ_op_kwarg);

    # sz op
    sz_op_kwarg = 0.5*np.copy(occ_op_kwarg) # wrong but I am lazy and doing spinless

    # run
    psii = np.zeros((L,),dtype=float);
    for sitei in range(Nconf):
        psii[sitei] = 1;
    psii = psii/np.sqrt(np.dot(np.conj(psii),psii)); # normalize
    print(">>> H =\n",H);
    print(">>> psii = ",psii);
    my_sites = np.arange(L); # sites to print out
    my_interval = 1.0; # how often to plot/printout
    my_updates = 12; # number of time interval repetitions
    main(H,L,psii,my_interval,my_updates,my_sites);

elif(case=="Heisenberg_sd"): # 2 impurity SWAP

    # physical parameters
    L = 2; # num sites
    Jzx = 1.5708;

    # hamiltonian
    H = (Jzx/4)*np.array([[1,0,0,0],
                          [0,-1,2,0],
                          [0,2,-1,0],
                          [0,0,0,1]]);

    # occ op
    occ_op_kwarg = np.array([np.eye(len(H)),np.eye(len(H))]);

    # sz op
    sz_op_kwarg = np.array([np.diagflat([0.5,0.5,-0.5,-0.5]),np.diagflat([0.5,-0.5,0.5,-0.5])]);

    # run
    psii = np.array([0,1.0,0,0]);
    print(">>> H =\n",H);
    print(">>> psii = ",psii);
    my_sites = [0,1]; # sites to print out
    my_interval = 0.4; # how often to plot/printout
    my_updates = 10; # number of time interval repetitions
    main(H,L,psii,my_interval,my_updates,my_sites);

elif(case=="Ne1_NFM2"): # 2 impurity SWAP

    # physical parameters
    L = 10; # num sites
    tl = 1.0;
    Jsd = 0.5;
    jds = [4,5];
    n_loc_dof = 8; # combined non-spatial dofs

    # j fixed blocks
    trian_H = -tl*np.eye(n_loc_dof);
    shape_0 = 0.0*np.eye(n_loc_dof);
    squar_H = 0.0*np.eye(n_loc_dof);
    star_j1 = (Jsd/4)*np.array([[1,0, 0, 0, 0, 0,0,0],
                        [0,1, 0, 0, 0, 0,0,0],
                        [0,0,-1, 0, 2, 0,0,0],
                        [0,0, 0,-1, 0, 2,0,0],
                        [0,0, 2, 0,-1, 0,0,0],
                        [0,0, 0, 2, 0,-1,0,0],
                        [0,0, 0, 0, 0, 0,1,0],
                        [0,0, 0, 0, 0, 0,0,1]]);
    star_j2 = (Jsd/4)*np.array([[1, 0, 0, 0, 0, 0, 0,0],
                        [0,-1, 0, 0, 2, 0, 0,0],
                        [0, 0, 1, 0, 0, 0, 0,0],
                        [0, 0, 0,-1, 0, 0, 2,0],
                        [0, 2, 0, 0,-1, 0, 0,0],
                        [0, 0, 0, 0, 0, 1, 0,0],
                        [0, 0, 0, 2, 0, 0,-1,0],
                        [0, 0, 0, 0, 0, 0, 0,1]]);

    # Hamiltonian (4d)
    H_4d = np.zeros((L,L,n_loc_dof,n_loc_dof),dtype=float);
    for sitei in range(L):
        # off diagonal
        if(sitei < L-1):
            H_4d[sitei,sitei+1] = np.copy(trian_H);
            H_4d[sitei+1,sitei] = np.copy(trian_H);
        # diagonal
        if(sitei==jds[0]):
            H_4d[sitei,sitei] += np.copy(star_j1);
        if(sitei==jds[1]):
            H_4d[sitei,sitei] += np.copy(star_j2);
        if(sitei not in jds): # ie overall else
            H_4d[sitei,sitei] += np.copy(squar_H);

    # impurity spin operators (4d)
    star_z1 = (1/2)*np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0,-1, 0, 0, 0, 0, 0],
                        [0, 0, 0,-1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0,-1, 0],
                        [0, 0, 0, 0, 0, 0, 0,-1]]);
    Sz1_4d = np.zeros_like(H_4d);
    for sitei in range(L):
        Sz1_4d[sitei,sitei] = np.copy(star_z1);
    star_z2 = (1/2)*np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0,-1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0,-1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0,-1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0,-1]]);
    Sz2_4d = np.zeros_like(H_4d);
    for sitei in range(L):
        Sz2_4d[sitei,sitei] = np.copy(star_z2);

    # site occ operators, listed
    occ_op_kwarg = [];
    for sitei in range(L):
        nj_4d = np.zeros_like(H_4d);
        nj_4d[sitei,sitei] = np.eye(n_loc_dof);
        occ_op_kwarg.append(utils.mat_4d_to_2d(nj_4d));
    occ_op_kwarg = np.array(occ_op_kwarg);
    
    # Sd spin operators, listed
    sz_op_kwarg = [];
    for sitei in range(L):
        if(sitei==jds[0]):
            sz_op_kwarg.append(utils.mat_4d_to_2d(Sz1_4d));
        elif(sitei==jds[1]): # this elif will NOT execute if jds[0]==jds[1] eg Eric
            sz_op_kwarg.append(utils.mat_4d_to_2d(Sz2_4d));
        else:
            sz_op_kwarg.append(utils.mat_4d_to_2d(np.zeros_like(Sz1_4d)));
    
    # run
    psii = np.zeros((L*n_loc_dof,),dtype=float);
    psii[3] = 0.495835 # j=0, Sze=up, Sz1=Sz2=down
    psii[3+n_loc_dof] = 0.504030  # j=1, Sze=up, Sz1=Sz2=down
    psii[3+2*n_loc_dof] = 0.000135 # j=2, Sze=up, Sz1=Sz2=down
    psii = psii/np.sqrt(np.dot(np.conj(psii),psii)); # normalize
    print(">>> H =\n",H_4d);
    print(">>> psii = ",psii);
    my_sites = np.arange(0,len(sz_op_kwarg));
    my_interval = 3.0; # how often to plot/printout
    my_updates = 2; # number of time interval repetitions
    main(utils.mat_4d_to_2d(H_4d),len(sz_op_kwarg),psii,my_interval,my_updates,my_sites);
