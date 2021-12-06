'''
Christian Bunker
M^2QM at UF
October 2021

Steady state transport of a single electron through a one dimensional wire
Part of the wire is scattering region, where the electron spin degrees of
freedom can interact with impurity spin degrees of freedom
In this case, impurities follow eric's paper
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

import sys

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
option = sys.argv[1];

# define params according to Eric's paper
tl = 1.0; # hopping >> other params
D = -0.06
JH = -0.005;

# eff params at resonance
JK = 2*D/3;
DeltaK = 0;

#### different ways of doing the scattering region

if option == "direct": # directly write down ham --> rabi flopping at resonance

    # define source
    source = np.zeros(3);
    source[2] = 1; # me, m1, m2 = down, 1, 1 state
    
    # fix energy near bottom of band
    Energy = -2*tl + 0.5;
    ka = np.arccos(Energy/(-2*tl));
    #print("ka = ", ka);
    #print("vg = ", 2*tl*np.sin(ka));

    # iter over N
    Nmax = 80
    Nvals = np.linspace(1,Nmax,min(Nmax, 20),dtype = int);
    Tvals = [];
    for N in Nvals:

        # eff exchange in SR
        h_ex = (1/4)*np.array([[0,DeltaK,4*JK],   # block diag m=3/2 subspace
                         [DeltaK,-8*JH,-2*DeltaK],
                         [4*JK,-2*DeltaK,-6*JK+4*D] ]);

        # package as block hams 
        # number of blocks depends on N
        hblocks = [np.zeros_like(h_ex)]
        tblocks = [-tl*np.eye(*np.shape(h_ex)) ];
        for Ni in range(N):
            hblocks.append(np.copy(h_ex));
            tblocks.append(-tl*np.eye(*np.shape(h_ex)) );
        hblocks.append(np.zeros_like(h_ex) );
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);
        if (N==2): print(hblocks, "\n", tblocks);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    if False:
        ax.scatter(Nvals, Tvals[:,0], marker = 's', label = '|1/2, 1/2, 2, 1>');
        ax.scatter(Nvals, Tvals[:,1], marker = 's', label = '|1/2, 1/2, 1, 1>');
        ax.scatter(Nvals, Tvals[:,2], marker = 's', label = '|1/2,-1/2, 2, 2>');
    else:
        ax.plot(Nvals, Tvals[:,0], label = '|1/2, 1/2> |2, 1>');
        ax.plot(Nvals, Tvals[:,1], label = '|1/2, 1/2> |1, 1>');
        ax.plot(Nvals, Tvals[:,2], label = '|1/2,-1/2> |2, 2>');

    # format
    #ax.set_title("Transmission at resonance, $J_K = 2D/3$");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$N$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();


elif option == "2q": # second quantized form of eric's model

    # define source
    source = np.zeros(3);
    source[2] = 1;  # me, m1, m2 = down, 1, 1 state

    # fix energy near bottom of band
    Energy = -2*tl + 0.5;
    ka = np.arccos(Energy/(-2*tl));

    # iter over N
    Nmax = 80
    Nvals = np.linspace(1,Nmax,min(Nmax,20),dtype = int);
    Tvals = [];
    for N in Nvals:

        # 2nd qu'd ham
        h1e, g2e = ops.h_switzer(D, JH, JK, JK);

        # check
        #t1 = [(2,3),(3,2)];
        #c1 = [1/np.sqrt(2),1/np.sqrt(2)];
        #t1 = [(1,0),(0,1)];
        #c1 = [1/np.sqrt(2),1/np.sqrt(2)];

        # convert to many body form
        parts = np.array([1,1,1]); # one particle each
        states = [[0,1],[2,3,4],[5,6,7]]; # e up, down, spin 1 mz, spin 2 mz
        interests = [[0,2,6],[0,3,5],[1,2,5]]; # pick me, m1, m2 = up, 1, 0>, up, 0, 1>, down, 1, 1 states
        h_SR = fci_mod.single_to_det(h1e,g2e, parts, states, dets_interest = interests);
        #print("D = ",D,", JH = ",JH,", JK1 = ", JK, ", JK2 = ",JK," pred = ", 2*D+JH-0.5*JK-0.5*JK);
        #print(h_SR);

        # entangle the me up states into eric's me, s12, m12> = up, 2, 1> state
        h_SR = wfm.utils.entangle(h_SR, 0, 1);

        # package as block hams 
        # number of blocks depends on N
        hblocks = [np.zeros_like(h_SR)]
        tblocks = [-tl*np.eye(*np.shape(h_SR)) ];
        for Ni in range(N):
            hblocks.append(np.copy(h_SR));
            tblocks.append(-tl*np.eye(*np.shape(h_SR)) );
        hblocks.append(np.zeros_like(h_SR) );
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);
        if (N==2): print(hblocks, "\n", tblocks);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.plot(Nvals, Tvals[:,0], label = '|1/2, 1/2> |2, 1>');
    ax.plot(Nvals, Tvals[:,1], label = '|1/2, 1/2> |1, 1>');
    ax.plot(Nvals, Tvals[:,2], label = '|1/2,-1/2> |2, 2>');

    # format
    #ax.set_title("Transmission at resonance, $J_K = 2D/3$");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$N$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();
        


elif option == "JK": # iter over JK at fixed energy, then over N, looking for resonance

    # eff params at resonance
    # JK tbd
    DeltaK = 0;
    
    # modulate
    JKvals = np.linspace(-0.1,0.01,12);
    maxvals = [];
    for JK in JKvals:

        print(">>>> JK = ",JK);

        # iter over N, just looking for max 
        Nmax = 180;
        Nvals = np.linspace(1,Nmax,Nmax,dtype = int);
        Nvals = Nvals[Nvals % 2 == 0 ];
        Tvals = [];
        for N in Nvals:

            # eff exchange in SR
            h_ex = (1/4)*np.array([[0,DeltaK,4*JK,              0,0,0,0,0],   # block diag m=3/2
                             [DeltaK,-8*JH,-2*DeltaK,     0,0,0,0,0],
                             [4*JK,-2*DeltaK,-6*JK+4*D,   0,0,0,0,0],
                             [0,0,0,         0,0,0,0,0],   # block diag m=1/2
                             [0,0,0,         0,0,0,0,0],
                             [0,0,0,         0,0,0,0,0],
                             [0,0,0,         0,0,0,0,0],
                             [0,0,0,         0,0,0,0,0]]);

            # package as block hams 
            # number of blocks depends on N
            hblocks = [np.zeros_like(h_ex)]
            tblocks = [-tl*np.eye(*np.shape(h_ex)) ];
            for Ni in range(N):
                hblocks.append(np.copy(h_ex));
                tblocks.append(-tl*np.eye(*np.shape(h_ex)) );
            hblocks.append(np.zeros_like(h_ex) );
            hblocks = np.array(hblocks);
            tblocks = np.array(tblocks);

            # get transmission coefs
            Energy = -2*tl + 0.5;
            Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

        # get max of |up, 2, 1>
        Tvals = np.array(Tvals);
        maxvals.append(np.max(Tvals[:,0]));

        # plot vs N at this JK
        Tvals = np.array(Tvals);
        fig, ax = plt.subplots();
        ax.scatter(Nvals, Tvals[:,0], marker = 's', label = '|1/2, 1/2> |2, 1>');
        ax.scatter(Nvals, Tvals[:,1], marker = 's', label = '|1/2, 1/2> |1, 1>');
        ax.scatter(Nvals, Tvals[:,2], marker = 's', label = '|1/2,-1/2> |2, 2>');

        # format
        #ax.set_title("");
        ax.set_ylabel("$T$");
        ax.set_xlabel("$N$");
        ax.set_ylim(0.0,1.05);
        plt.legend();
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        plt.show();

    # plot max vals vs JK
    fig, ax = plt.subplots()
    ax.scatter(JKvals, maxvals, marker = 's');
    ax.axvline(D*2/3, color = "black", linestyle = "dashed");
    ax.set_ylabel("max($T_{flip}$)");
    ax.set_xlabel("$J_K$");
    ax.set_ylim(0.0,1.05);
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();
    

    








