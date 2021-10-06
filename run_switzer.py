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
source = np.zeros(8); # corresponds to |up, 2, 1 > in the device basis
source[0] = 1;

# define params according to Eric's paper
tl = 1.0; # hopping
D = -0.06;
JH = -0.005;
JK2 = -0.04;
JK3 = -0.04;

if True: 

    # eff params at resonance
    JK = D*2/3;
    DeltaK = 0;

    # iter over N
    Nmax = 30
    Nvals = np.linspace(1,Nmax,Nmax,dtype = int);
    Tvals = [];
    for N in Nvals:

        # fix energy near bottom of band
        Energy = -2*tl + 0.5;

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
        ax.scatter(Nvals, Tvals[:,0], marker = 's', label = '|up, 2, 1>');
        ax.scatter(Nvals, Tvals[:,1], marker = 's', label = '|up, 1, 1>');
        ax.scatter(Nvals, Tvals[:,2], marker = 's', label = '|down, 2, 1>');
    else:
        ax.plot(Nvals, Tvals[:,0], label = '|up, 2, 1>');
        ax.plot(Nvals, Tvals[:,1], label = '|up, 1, 1>');
        ax.plot(Nvals, Tvals[:,2], label = '|down, 2, 1>');

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

if False: # iter over th, ie hopping in/out of SR

    # eff params at resonance
    JK = D*2/3;
    DeltaK = 0;
    
    # iter over JK looking for resonance
    thvals = np.linspace(0.1,0.9,40);
    Tvals = []; # get transmission each time
    for th in thvals:

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
        tblocks = np.array([-th*np.eye(*np.shape(h_ex)), -th*np.eye(*np.shape(h_ex))]);

        # get transmission coefs
        Energy = -2*tl + 2; # fix
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.scatter(thvals, Tvals[:,0], marker = 's', label = '|up, 2, 1>');
    ax.scatter(thvals, Tvals[:,1], marker = 's', label = '|up, 1, 1>');
    ax.scatter(thvals, Tvals[:,2], marker = 's', label = '|down, 2, 1>');

    # format
    ax.set_title("Transmission at resonance, $J_K = 2D/3$");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$t_h$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();

if False: 

    # eff params at resonance
    JK = D*2/3;
    DeltaK = 0;
    
    # modulate
    Vgvals = np.linspace(0.0,0.5,40);
    Tvals = []; # get transmission each time
    for Vg in Vgvals:

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
        hblocks = np.array([np.zeros_like(h_ex), h_ex-np.eye(*np.shape(h_ex))*Vg/2, np.zeros_like(h_ex)-np.eye(*np.shape(h_ex))*Vg]);
        tblocks = np.array([-tl*np.eye(*np.shape(h_ex)), -tl*np.eye(*np.shape(h_ex))]);

        # get transmission coefs
        Energy = -2*tl + 0.5; # fix
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    ax.scatter(Vgvals, Tvals[:,0], marker = 's', label = '|up, 2, 1>');
    ax.scatter(Vgvals, Tvals[:,1], marker = 's', label = '|up, 1, 1>');
    ax.scatter(Vgvals, Tvals[:,2], marker = 's', label = '|down, 2, 1>');

    # format
    ax.set_title("Transmission at resonance, $J_K = 2D/3$");
    ax.set_ylabel("$T$");
    ax.set_xlabel("$V_g$");
    ax.set_ylim(0.0,1.05);
    plt.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();
    

    








