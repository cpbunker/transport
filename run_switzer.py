'''
Christian Bunker
M^2QM at UF
October 2021

Transmit an itinerant electron through eric's model
'''

import wfm

import numpy as np
import matplotlib.pyplot as plt
import sys

##################################################################################
#### make contact with menezes

# top level
#plt.style.use('seaborn-dark-palette');
#colors = seaborn.color_palette("dark");
colors = ['tab:blue','tab:red','tab:green','tab:blue'];
np.set_printoptions(precision = 4, suppress = True);
verbose = 5

# define source
sourcei = 0; # corresponds to |up, 2, 1 > in the device basis

# define params according to Eric's paper
tl = 1.0; # hopping
D = 0.2;
JH = 0.3;
JK2 = 0.1;
JK3 = 0.1;

if True: # iter over tl

    # eff params at resonance
    JK = D*2/3;
    DeltaK = 0;

    tlvals = np.linspace(0.5,9.5,20);
    Tvals = [];
    for mytl in tlvals:

        # fix energy near bottom of band
        Energy = (-2)*mytl + 0.1;

        # eff exchange in SR
        h_ex = np.array([[0,DeltaK,4*JK,              0,0,0,0,0],   # block diag m=3/2
                         [DeltaK,-8*JH,-2*DeltaK,     0,0,0,0,0],
                         [4*JK,-2*DeltaK,-6*JK+4*D,   0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0],   # block diag m=1/2
                         [0,0,0,         0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0]]);

        # package as block hams
        hblocks = np.array([np.zeros_like(h_ex), h_ex, np.zeros_like(h_ex)]);
        tblocks = np.array([-mytl*np.eye(*np.shape(h_ex)), -mytl*np.eye(*np.shape(h_ex))]);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, Energy, sourcei));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.scatter(tlvals, Tvals[:,0], marker = 's', label = '|up, 2, 1>');
    ax.scatter(tlvals, Tvals[:,1], marker = 's', label = '|up, 1, 1>');
    ax.scatter(tlvals, Tvals[:,2], marker = 's', label = '|down, 2, 1>');

    # format
    ax.set_title("");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$E + 2*t_l$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();

if False: # iter over E

    # eff params
    JK = (JK2 + JK3)/2;
    DeltaK = JK2 - JK3;

    # eff exchange in SR
    h_ex = np.array([[0,DeltaK,4*JK,              0,0,0,0,0],   # block diag m=3/2
                     [DeltaK,-8*JH,-2*DeltaK,     0,0,0,0,0],
                     [4*JK,-2*DeltaK,-6*JK+4*D,   0,0,0,0,0],
                     [0,0,0,         0,0,0,0,0],   # block diag m=1/2
                     [0,0,0,         0,0,0,0,0],
                     [0,0,0,         0,0,0,0,0],
                     [0,0,0,         0,0,0,0,0],
                     [0,0,0,         0,0,0,0,0]]);

    # package as block hams
    hblocks = np.array([np.zeros_like(h_ex), h_ex, np.zeros_like(h_ex)]);
    tblocks = np.array([-tl*np.eye(*np.shape(h_ex)), -tl*np.eye(*np.shape(h_ex))]);

    # iter over E
    Emin, Emax = -1.99999, -2.0 + 0.9
    N = 40;
    Evals = np.linspace(Emin, Emax, N, dtype = complex);
    Tvals = [];
    for Ei in range(len(Evals) ):
        Tvals.append(wfm.Tcoef(hblocks, tblocks, Evals[Ei], sourcei));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.scatter(Evals+2*tl, Tvals[:,0], marker = 's', label = '|up, 2, 1>');
    ax.scatter(Evals+2*tl, Tvals[:,1], marker = 's', label = '|up, 1, 1>');
    ax.scatter(Evals+2*tl, Tvals[:,2], marker = 's', label = '|down, 2, 1>');

    # format
    ax.set_title("");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$E + 2*t_l$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();

if False: # iter over J

    # fix energy near bottom of band
    Energy = -2*tl + 0.1;
    
    # iter over JK looking for resonance
    jkvals = np.linspace(0.0,2.0,20);
    Tvals = []; # get transmission each time
    for jkval in jkvals:

        # eff params
        JK2, JK3 = jkval, jkval;
        JK = (JK2 + JK3)/2;
        DeltaK = JK2 - JK3;

        # eff exchange in SR
        h_ex = np.array([[0,DeltaK,4*JK,              0,0,0,0,0],   # block diag m=3/2
                         [DeltaK,-8*JH,-2*DeltaK,     0,0,0,0,0],
                         [4*JK,-2*DeltaK,-6*JK+4*D,   0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0],   # block diag m=1/2
                         [0,0,0,         0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0],
                         [0,0,0,         0,0,0,0,0]]);

        # package as block hams
        hblocks = np.array([np.zeros_like(h_ex), h_ex, np.zeros_like(h_ex)]);
        tblocks = np.array([-tl*np.eye(*np.shape(h_ex)), -tl*np.eye(*np.shape(h_ex))]);

        # get transmission coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, Energy, sourcei));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.scatter(jkvals, Tvals[:,0], marker = 's', label = '|up, 2, 1>');
    ax.scatter(jkvals, Tvals[:,1], marker = 's', label = '|up, 1, 1>');
    ax.scatter(jkvals, Tvals[:,2], marker = 's', label = '|down, 2, 1>');

    # format
    ax.set_title("");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$J_K$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();
    

    








