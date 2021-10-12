'''
Christian Bunker
M^2QM at UF
October 2021

Scattering off a step potential with self energy
'''

import siam_current

import numpy as np
import scipy

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
bdims = [100,200,300];
noises = [1e-4,1e-5,0.0];

if get_data: # must actually compute data
    
    # iter over energies
    for Eval in []:
        
        # custom ham, ASU formalism
        h1e = np.array([ [-B/2,0,0,0,0,0],
                         [0,+B/2,0,0,0,0],
                         [0,0,Vb,0,0,0],
                         [0,0,0,Vb,0,0],
                         [0,0,0,0,0,0],
                         [0,0,0,0,0,0] ], dtype = complex);  
        g2e = np.zeros((np.shape(h1e)[0],np.shape(h1e)[0],np.shape(h1e)[0],np.shape(h1e)[0]));
        
        h1e_neq = np.array([ [0,0,-th,0,0,0],
                             [0,0,0,-th,0,0],
                             [-th,0,Vb,0,-th,0],
                             [0,-th,0,Vb,0,-th],
                             [0,0,-th,0,0,0],
                             [0,0,0,-th,0,0] ], dtype = complex);  

        # self energies
        lamL = (Eval - 0)/(-2*tl);
        lamR = (Eval - 0)/(-2*tl);
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
        
        siam_current.CustomDataDmrg(h1e, g2e, h1e_neq, nelecs, tf, dt, bdims, noises, fname = "step_E"+str(Eval)[:4]+".npy", verbose = verbose);   
    
else:

    import plot
    import matplotlib.pyplot as plt

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

    


    








