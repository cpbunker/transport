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
source = np.zeros(8); 
source[2] = 1; # |1,2, -1/2> |2,2> as in Eric's Eq 10

# define params according to Eric's paper
tl = 1.0; # hopping
D = -0.06;
JH = -0.005;
JK2 = -0.04;
JK3 = -0.04;

if False: # vary N at resonance

    # eff params at resonance
    JK = D*2/3;
    DeltaK = 0;

    # iter over N
    Nmax = 120
    Nvals = np.linspace(1,Nmax,Nmax,dtype = int);
    Tvals = [];
    for N in Nvals:

        # fix energy near bottom of band
        Energy = -2*tl + 0.5;
        ka = np.arccos(Energy/(-2*tl)) 
        print("ka = ", ka);
        print("vg = ", 2*tl*np.sin(ka));

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


if True: # iter over JK at fixed energy, then over N, looking at peak size

    # eff params at resonance
    # JK tbd
    DeltaK = 0;
    
    # modulate
    JKvals = np.linspace(-0.1,0.01,12);
    maxvals = [];
    for JK in JKvals:

        print(">>>> JK = ",JK);

        # iter over N, just looking for max 
        Nmax = 60;
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
    

    








