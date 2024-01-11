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

if(case == "Ne1"): # 5 site TB problem

    # physical parameters
    L = 5; # num sites
    tl = 1.0;

    # hamiltonian
    H = np.zeros((L,L),dtype=float);
    for i in range(L-1):
        H[i,i+1] += -tl;
        H[i+1,i] += -tl;

    # occ op
    occ_op_kwarg = np.array([np.diagflat([1,0,0,0,0]),
                             np.diagflat([0,1,0,0,0]),
                             np.diagflat([0,0,1,0,0]),
                             np.diagflat([0,0,0,1,0]),
                             np.diagflat([0,0,0,0,1])]);

    sz_op_kwarg = 0.5*np.copy(occ_op_kwarg) # wrong but I am lazy and doing spinless

    # run
    psii = np.array([1.0,0,0,0,0]);
    print(">>> H =\n",H);
    print(">>> psii = ",psii);
    my_sites = [0,1]; # sites to print out
    my_interval = 1.0; # how often to plot/printout
    my_updates = 2; # number of time interval repetitions
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
    H_4d =    np.array([[squar_H,trian_H,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [trian_H,squar_H,trian_H,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,trian_H,squar_H,trian_H,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,trian_H,squar_H,trian_H,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,trian_H,star_j1,trian_H,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,trian_H,star_j2,trian_H,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,trian_H,squar_H,trian_H,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,trian_H,squar_H,trian_H,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,trian_H,squar_H,trian_H],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,trian_H,squar_H]]);

    # impurity spin operators (4d)
    star_z1 = (1/2)*np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0,-1, 0, 0, 0, 0, 0],
                        [0, 0, 0,-1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0,-1, 0],
                        [0, 0, 0, 0, 0, 0, 0,-1]]);
    Sz1_4d =  np.array([[star_z1,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,star_z1,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,star_z1,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,star_z1,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,star_z1,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,star_z1,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z1,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z1,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z1,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z1]]);
    star_z2 = (1/2)*np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0,-1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0,-1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0,-1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0,-1]]);
    Sz2_4d =  np.array([[star_z2,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,star_z2,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,star_z2,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,star_z2,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,star_z2,shape_0,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,star_z2,shape_0,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z2,shape_0,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z2,shape_0,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z2,shape_0],
                        [shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,shape_0,star_z2]]);

    # site occ operators (4d)
    nj0_4d = np.zeros_like(Sz1_4d);
    nj0_4d[0,0] = np.eye(n_loc_dof);
    nj1_4d = np.zeros_like(Sz1_4d);
    nj1_4d[1,1] = np.eye(n_loc_dof);
    nj7_4d = np.zeros_like(Sz1_4d);
    nj7_4d[7,7] = np.eye(n_loc_dof);
    nj8_4d = np.zeros_like(Sz1_4d);
    nj8_4d[8,8] = np.eye(n_loc_dof);
    
    # convert all to 2d
    H = utils.mat_4d_to_2d(H_4d);
    # we need sites 0,1,4,5,7,8
    zero_filler = utils.mat_4d_to_2d(np.zeros_like(Sz1_4d));
    sz_op_kwarg = np.array([zero_filler,zero_filler,utils.mat_4d_to_2d(Sz1_4d),utils.mat_4d_to_2d(Sz2_4d),zero_filler,zero_filler]);
    occ_op_kwarg = np.array([utils.mat_4d_to_2d(nj0_4d),utils.mat_4d_to_2d(nj1_4d),zero_filler,zero_filler,utils.mat_4d_to_2d(nj7_4d),utils.mat_4d_to_2d(nj8_4d)]);

    # treat j/d as same
    occ_op_kwarg = [];
    sz_op_kwarg = [];
    for jdindex in range(L):
        nj_4d = np.zeros_like(Sz1_4d);
        nj_4d[jdindex,jdindex] = np.eye(n_loc_dof);
        occ_op_kwarg.append(utils.mat_4d_to_2d(nj_4d));
        szd_4d = np.zeros_like(Sz1_4d);
        if(jdindex == 4):
            szd_4d = np.copy(Sz1_4d);
        elif(jdindex == 5):
            szd_4d = np.copy(Sz2_4d);
        sz_op_kwarg.append(utils.mat_4d_to_2d(szd_4d));
    

    # run
    psii = np.zeros((L*n_loc_dof,),dtype=float);
    psii[3] = 1.0 #4.95864938e-01 # j=0, Sze=up, Sz1=Sz2=down
    psii[3+n_loc_dof] = 1.0 #5.03992770e-01  # j=1, Sze=up, Sz1=Sz2=down
    psii = psii/np.sqrt(np.dot(np.conj(psii),psii));
    print(">>> H =\n",H);
    print(">>> psii = ",psii);
    my_sites = np.arange(0,len(sz_op_kwarg));
    my_interval = 3.0; # how often to plot/printout
    my_updates = 2; # number of time interval repetitions
    main(H,len(sz_op_kwarg),psii,my_interval,my_updates,my_sites);
