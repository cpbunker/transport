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

def snapshot(state,the_H,nsites,time, imp_sites):
    '''
    Args:
    state
    the_H
    nsites
    time
    imp_sites, list, indices of impurities (so we can retrieve <Sdz>)
    '''

    # observables
    occs = np.zeros((nsites,),dtype=float);
    Sdzs = np.zeros((nsites,),dtype=float);
    for sitei in range(nsites):
        occ_ret = np.dot( np.conj(state), np.matmul(occ_op_kwarg[sitei],state));
        assert(np.imag(occ_ret) < 1e-10);
        occs[sitei] = np.real(occ_ret);
        Sdz_ret = np.dot( np.conj(state), np.matmul(Sdz_op_kwarg[sitei],state));
        assert(np.imag(Sdz_ret) < 1e-10);
        Sdzs[sitei] = np.real(Sdz_ret);
    

    # printout
    print("Time = {:.2f}".format(time));
    print("Total energy (ED) = {:.6f}".format(np.dot(np.conj(state),np.matmul(the_H, state))));
    for sitei in range(nsites):
        print("Site {:.0f} <n>  (ED) = {:.6f}".format(sitei,occs[sitei])); 
        #print("Imp  {:.0f} <Sz> (ED) = {:.6f}".format(sitei,Sdzs[sitei]));
    print("Summed occ (ED) = {:.6f}".format(np.sum(occs)));
    print("Summed Sdz (ED) = {:.6f}".format(np.sum(Sdzs)));
    
    # use occs to get n_R
    #
    
    # return relevant data
    Sdz_return = [];
    for impi in imp_sites:
        Sdz_return.append(Sdzs[impi]);
    return (Sdz_return, occs)

#####################################################################
#### wrapper code

def main(the_H, nsites, init_state, time_snap, time_N, imp_sites, savedir):
    '''
    wraps all the time evolution and printout steps

    Args:
    the_H, np array, Hamiltonian in determinant basis
    nsites, int, number of fermionic sites in system (imp sites are supersited)
    init_state, np array, time 0 many-body state, to be propagated forward
    time_snap, smallest timestep of the fci propagation operation, aka `dt`
    time_N, int, total number of dt steps to take forward in time
    imp_sites, list, indices of impurities (so we can retrieve <Sdz>)
    savedir, string, where to save observable vs time arrays to
    '''
    assert(len(the_H) == len(init_state))
    for impi in imp_sites: assert(impi in np.arange(0,nsites));
    
    # observables vs time
    times = np.zeros((time_N+1,),dtype=float);
    Sdz1_vals = np.zeros_like(times);
    Sdz2_vals = np.zeros_like(times);
    nL_vals = np.zeros_like(times);

    # solve the system
    eigvals, eigvecs = tdfci.solver(the_H);

    # time prop with observables
    state = np.copy(init_state);
    for time_stepi in range(0,time_N+1):
    
        # propagate
        if(time_stepi != 0):
            state = tdfci.propagator(state, time_snap, eigvals, eigvecs);
        
        # observables
        obs_ret = snapshot(state,the_H,nsites,time_stepi*time_snap, imp_sites);
        times[time_stepi] = time_stepi*time_snap;
        Sdz1_vals[time_stepi] = obs_ret[0][0];
        Sdz2_vals[time_stepi] = obs_ret[0][1];
        nL_vals[time_stepi] = np.sum(obs_ret[1][:imp_sites[0]])
        
        # save observables vs time
        Sdz_xjs = 1*imp_sites; 
        assert(len(Sdz_xjs)==2)
        Sdz_yjs = [Sdz1_vals[time_stepi], Sdz2_vals[time_stepi]]
        np.save(savedir+"Sdz_xjs_time{:.2f}.npy".format(times[time_stepi]), Sdz_xjs);
        np.save(savedir+"Sdz_yjs_time{:.2f}.npy".format(times[time_stepi]), Sdz_yjs);
        occ_xjs = np.arange(nsites);
        occ_yjs = obs_ret[1];
        np.save(savedir+"occ_xjs_time{:.2f}.npy".format(times[time_stepi]), occ_xjs);
        np.save(savedir+"occ_yjs_time{:.2f}.npy".format(times[time_stepi]), occ_yjs);   
        
        
    # plot observables versus time
    observable_vals = [Sdz1_vals, Sdz2_vals, nL_vals];
    observable_labs = ["$S_1^z$", "$S_2^z$","$n_L$"];
    fig, axes = plt.subplots(len(observable_vals), sharex=True);
    for axi in range(len(axes)):
        axes[axi].plot(times, observable_vals[axi]);
        axes[axi].set_ylabel(observable_labs[axi]);
        for line in [-0.5]: axes[axi].axhline(line, color="gray", linestyle="dashed");
    
    axes[-1].set_xlabel("Time");
    #axes[-1].set_ylim(-0.5,0.0);
    plt.show();
    return;
        
#####################################################################
#### exec code

# top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
case = sys.argv[1];

if(case==""):
    raise Exception("user did not input a case");

elif(case=="Ne1_NFM2"): # 2 impurities

    # physical parameters
    L = 22 #14; # num fermionic sites -- imp sites are supersited
    tl = 1.0;
    Jsd = 1.0;
    my_jds = [10,11]#[2,3]; # supersited sites
    n_loc_dof = 8; # combined non-spatial dofs
    
    
    ###################################
    #
    # set the time=0 wavefunction
    #
    # NB initial spin state:
    # Sze=up, Sz1=Sz2=down
    select_spin_state = 3;
    #
    #
    psii = np.zeros((L*n_loc_dof,),dtype=float);
    # all the js in left lead, w/ appropriate spin state, are Ne occupied
    # however, we have to occupy manually in order to agree w/ td-DMRG ground state
    if(False): #L==22):
        psii[select_spin_state+0*n_loc_dof] = np.sqrt(0.0144) # j=0,
        psii[select_spin_state+1*n_loc_dof] = np.sqrt(0.0530) # j=1,
        psii[select_spin_state+2*n_loc_dof] = np.sqrt(0.1036) # j=2,
        psii[select_spin_state+3*n_loc_dof] = np.sqrt(0.1501) # j=3, 
        psii[select_spin_state+4*n_loc_dof] = np.sqrt(0.1779) # j=4,
        psii[select_spin_state+5*n_loc_dof] = np.sqrt(0.1780) # j=5,
        psii[select_spin_state+6*n_loc_dof] = np.sqrt(0.1506) # j=6,
        psii[select_spin_state+7*n_loc_dof] = np.sqrt(0.1042) # j=7,
        psii[select_spin_state+8*n_loc_dof] = np.sqrt(0.0536) # j=8,
        psii[select_spin_state+9*n_loc_dof] = np.sqrt(0.0147) # j=9, # end of 10-site left lead
    elif(True): #L==14):
        psii[select_spin_state+0*n_loc_dof] = 1.0000 # j=0,
        psii[select_spin_state+1*n_loc_dof] = 1.0000 # j=1, # end of 2-site left lead
    else:
        raise Exception("L = {:.0f} invalid".format(L));
        
    psii = psii/np.sqrt(np.dot(np.conj(psii),psii)); # normalize
    print(">>> psii =\n",psii.reshape(len(psii)//n_loc_dof, n_loc_dof));

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
        if(sitei==my_jds[0]):
            H_4d[sitei,sitei] += np.copy(star_j1);
        if(sitei==my_jds[1]):
            H_4d[sitei,sitei] += np.copy(star_j2);
        if(sitei not in my_jds): # ie overall else
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
        
    #print(">>> H =\n",H_4d);


    ###############################
    ####
    #### here we define global operators !!

    # site occ operators, listed
    occ_op_kwarg = [];
    for sitei in range(L):
        nj_4d = np.zeros_like(H_4d);
        nj_4d[sitei,sitei] = np.eye(n_loc_dof);
        occ_op_kwarg.append(utils.mat_4d_to_2d(nj_4d));
    occ_op_kwarg = np.array(occ_op_kwarg);
    
    # spin operators, listed
    # if a fermionic site, the spin operator in this list is 0
    # if a impurity supersite, the spin operator is impurity spin(even tho there is also fermionic spin on the site)
    Sdz_op_kwarg = [];
    for sitei in range(L):
        if(sitei==my_jds[0]):
            Sdz_op_kwarg.append(utils.mat_4d_to_2d(Sz1_4d));
        elif(sitei==my_jds[1]): # this elif will NOT execute if jds[0]==jds[1] eg Eric
            Sdz_op_kwarg.append(utils.mat_4d_to_2d(Sz2_4d));
        else:
            assert(len(my_jds)==2);
            Sdz_op_kwarg.append(utils.mat_4d_to_2d(np.zeros_like(Sz1_4d)));

    ###############################
    ####
    #### finished defining global operators !!
    
    # run
    my_interval = 1.0;   # dt of exact propagator (dont think there is time step error)
                         # ALSO how often to plot/printout
    my_updates = 20; # number of time interval repetitions
    my_dir = sys.argv[2]; # where to save exact diag results to
    main(utils.mat_4d_to_2d(H_4d),len(Sdz_op_kwarg),psii,my_interval,my_updates, my_jds, my_dir);
    
   

elif(case == "Ne1"): # 1 electron, 1D TB problem

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

 
