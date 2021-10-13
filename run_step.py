'''
Christian Bunker
M^2QM at UF
October 2021

Scattering off a step potential with self energy
'''

import siam_current
import ops_dmrg
import td_dmrg

from pyblock3 import fcidump, hamiltonian
from pyblock3.algebra.mpe import MPE

import numpy as np
import plot
import matplotlib.pyplot as plt

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nelecs = (1,0);
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# step potential tight binding
tl = 1.0;
th = 1.0;
Vb = 0.2
B = 10;
Emin, Emax = -2.3, -1.5
Evals = np.linspace(Emin, Emax, 9);

#time info
dt = 0.01;
tf = 2.0;

# dmrg
bdims = [200,250,300];
noises = [1e-4,1e-5,0.0];

states = [];
if get_data: # must actually compute data
    
    # iter over energies
    for Eval in Evals:
        
        # custom ham, ASU formalism
        h1e = np.array([ [-B/2,0,0,0,0,0],
                         [0,+B/2,0,0,0,0],
                         [0,0,Vb,0,0,0],
                         [0,0,0,Vb,0,0],
                         [0,0,0,0,Vb,0],
                         [0,0,0,0,0,Vb] ], dtype = complex);  
        g2e = np.zeros((np.shape(h1e)[0],np.shape(h1e)[0],np.shape(h1e)[0],np.shape(h1e)[0]));
        
        h1e_neq = np.array([ [0,0,-th,0,0,0],
                             [0,0,0,-th,0,0],
                             [-th,0,Vb,0,-th,0],
                             [0,-th,0,Vb,0,-th],
                             [0,0,-th,0,Vb,0],
                             [0,0,0,-th,0,Vb] ], dtype = complex);  

        # self energies
        lamL = (Eval - 0)/(-2*tl);
        lamR = (Eval - Vb)/(-2*tl);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1);
        SigmaL = -tl/LambdaLminus;
        SigmaR = -tl*LambdaRplus;
        if True:
            h1e[0,0] += SigmaL;
            h1e[1,1] += SigmaL;
            h1e[-2,-2] += SigmaR;
            h1e[-1,-1] += SigmaR;
            h1e_neq[0,0] += SigmaL;
            h1e_neq[1,1] += SigmaL;
            h1e_neq[-2,-2] += SigmaR;
            h1e_neq[-1,-1] += SigmaR;
        print("Full noneq. hamiltonian:\n",h1e_neq);

        # td dmrg and save data
        # returns dmrg gd state
        siam_current.CustomDataDmrg(h1e, g2e, h1e_neq, nelecs, tf, dt, bdims, noises, fname = "step_E"+str(Eval)[:4]+".npy", verbose = verbose);

        # gd state observables
        if False:

            # get gd state
            norbs = np.shape(h1e)[0];
            hdump_neq = fcidump.FCIDUMP(h1e=h1e_neq,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); 
            h_obj_neq = hamiltonian.Hamiltonian(hdump_neq, flat=True);
            h_mpo_neq = h_obj_neq.build_qc_mpo(); # got mpo
            h_mpo_neq, _ = h_mpo_neq.compress(cutoff=1E-15); # compression saves memory

            # neq eigenstate
            psi_neq = h_obj_neq.build_mps(bdims[0]);
            MPE_obj_neq = MPE(psi_neq, h_mpo_neq, psi_neq);
            dmrg_obj_neq = MPE_obj_neq.dmrg(bdims = bdims, noises = noises, tol = 1e-8, iprint = 0); # solves neq system
            neq_eigenstate = MPE_obj_neq.ket;

            # make observable operators
            obs_ops = [];
            for sitei in range(int(norbs/2)):
                site = [2*sitei, 2*sitei+1];
                obs_ops.append( h_obj_neq.build_mpo(ops_dmrg.occ(site, norbs) ) );

            # get observables: energy and site occs
            obs = [Eval, dmrg_obj_neq.energies[-1]];
            for op in obs_ops:
                obs.append(td_dmrg.compute_obs(op, neq_eigenstate) );
            states.append(obs);

    # plot
    if False:
        states = np.array(states);
        fig, ax = plt.subplots();
        for sitei in range(int(norbs/2)):
            ax.scatter(Evals, states[:,2+sitei], marker = 's', label = str(sitei));
        plt.show();
                
    
else:

    # plot results
    datafs = sys.argv[2:]
    splots = ['Jup','Jdown','occ','Sz']; # which subplots to plot
    mysites = ['LL','SR','RL'];

    # get conductances
    conds = [];
    cond_window = 10;
    for f in datafs:

        # load
        observables = np.load(f);
        observables = observables.T;
        t, E, JupL, JupR = observables[0], observables[1], -np.imag(observables[2]), -np.imag(observables[3])
        Jup = (JupL + JupR)/2;
        Jup = Jup[:int(len(Jup)/2)]; # truncate
        maxi = np.argmax(Jup);
        conds.append( np.average( Jup[maxi]) );

    plt.scatter(Evals+2*tl, conds, marker = 's');
    plt.xlabel("$E+2t_l$");
    plt.ylabel("$J_{max}$");
    plt.show();

    


    








