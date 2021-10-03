'''
Christian Bunker
M^2QM at UF
September 2021

Access electron transport regime for a 1 quantum dot model
eg Menezes' paper.

Try to replicate with eff J S1 dot S2 and with full treatment
'''

import wfm

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

##################################################################################
#### make contact with menezes

# top level
#plt.style.use('seaborn-dark-palette');
colors = seaborn.color_palette("dark");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5

if False: # plot at different eff J

    # siam inputs
    tl = 1.0;
    Vg = 0.1;
    U = 25.0;
    
    fig, axes = plt.subplots();
    axes = [axes];
    for J in [0.1,0.4]:

        #J S1 dot S2 
        Jmat = (J/4)*np.array([[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]]);

        # menezes just has single delta potential interaction
        h_menez = np.array([np.zeros_like(Jmat), Jmat, np.zeros_like(Jmat) ]);

        # construct hopping
        Tmat = -tl*np.eye(*np.shape(Jmat));
        tl_arr = np.array([ np.copy(Tmat), np.copy(Tmat) ]);
        if verbose: print(h_menez,"\n", tl_arr);

        # define source, ie what is incident at left bdy, in this case an up electron
        sourcei = 1; # up e, down imp

        if False: # test at max verbosity
            myT = wfm.Tcoef(h_menez, tl_arr, -1.99, sourcei, verbose = 5);
            if verbose: print("******",myT);

        # sweep over range of energies
        # def range
        Emin, Emax = -2,-2+0.2
        N = 10;
        Evals = np.linspace(Emin, Emax, N, dtype = complex);
        Tupvals = np.zeros_like(Evals);
        Tdownvals = np.zeros_like(Evals);

        # sweep thru E
        for Ei in range(len(Evals) ):
            dummyT1, Tupvals[Ei], Tdownvals[Ei], dummyT2 = wfm.Tcoef(h_menez, tl_arr, Evals[Ei], sourcei);
            assert(dummyT1 == 0 and dummyT2 == 0);

        # plot
        s2 = axes[0].scatter(Evals+2*tl, Tdownvals, marker = 's', label = "$T_{down},\, J = $"+str(J));

        # menezes prediction in the continuous case
        # all the definitions, vectorized funcs of E
        newEvals = np.linspace(0.0,0.2,100);
        kappa = np.lib.scimath.sqrt(newEvals);
        jprime = J/(4*kappa);
        l1, = axes[0].plot(newEvals, J*J/(16*newEvals));

    # format
    axes[0].set_title("Up electron scattering from a down spin impurity");
    axes[0].set_ylabel("$T$");
    axes[0].set_xlabel("$E + 2t_l$");
    axes[0].set_ylim(0.0,1.05);
    plt.legend();
    for ax in axes:
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();


if True: # use real ham instead of eff

    # siam inputs
    tl = 1.0;
    Vg = 20.0;
    U = 100.0;
    Jeff = -2*tl*tl/Vg

    # diag ham = blocks for LL, SR, RL
    h_L1 = np.array([[0,0,0,0,0,0],
                     [0,-Vg,0,0,0,0],
                     [0,0,-Vg,0,0,0],
                     [0,0,0,-Vg,0,0],
                     [0,0,0,0,-Vg,0],
                     [0,0,0,0,0,-Vg]]);
    h_SR = np.array([[U-2*Vg,0,tl,-tl,0,0],
                     [0,    0,-tl,tl,0,0],
                     [tl,-tl,-Vg,0,0,0],
                     [-tl,tl, 0,-Vg,0,0],
                     [0,0,0,0,-Vg,0],
                     [0,0,0,0,0,-Vg]]);
    h_R1 = np.array([[0,0,0,0,0,0],
                     [0,-Vg,0,0,0,0],
                     [0,0,-Vg,0,0,0],
                     [0,0,0,-Vg,0,0],
                     [0,0,0,0,-Vg,0],
                     [0,0,0,0,0,-Vg]]);


    h_L1 += Vg*np.eye(*np.shape(h_L1));
    h_SR += Vg*np.eye(*np.shape(h_SR));
    h_R1 += Vg*np.eye(*np.shape(h_R1));
    h_menez = np.array([h_L1, h_SR, h_R1]);

    # construct hopping
    tl_arr = np.array([-tl*np.eye(*np.shape(h_SR)),-tl*np.eye(*np.shape(h_SR))]);
    if verbose: print(h_menez,"\n", tl_arr);

    # define source, ie what is incident at left bdy, in this case an up electron
    sourcei = 2;

    if False: # test at max verbosity
        myT = wfm.Tcoef(h_menez, tl_arr, -1.99, sourcei, verbose = 5);
        if verbose: print("******",myT);

    # sweep over range of energies
    # def range
    Emin, Emax = -1.99999, -2.0 + 0.2
    N = 20;
    Evals = np.linspace(Emin, Emax, N, dtype = complex);
    Tvals = [];
    for Ei in range(len(Evals) ):
        Tvals.append(wfm.Tcoef(h_menez, tl_arr, Evals[Ei], sourcei));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.scatter(Evals + 2*tl,Tvals[:,sourcei+1], marker = 's',label = "$T_{down}$");

    # menezes prediction in the continuous case
    # all the definitions, vectorized funcs of E
    kappa = np.lib.scimath.sqrt(Evals);
    jprime = Jeff/(4*kappa);
    l1, = ax.plot(Evals+2*tl, Jeff*Jeff/(16*(Evals+2*tl)));

    # format
    ax.set_ylim(0.0,1.05);
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    ax.set_xlabel("$E+2t_l $");
    ax.set_ylabel("$T$");
    ax.set_title("Up electron scattering from down impurity");
    ax.legend(title = "$t_{l} = $"+str(tl)+"\n$V_g = $"+str(Vg)+"\n$U = $"+str(U)+"\n$J_{eff} = $"+str(Jeff));
    plt.show();

    








