'''
Christian Bunker
M^2QM at UF
September 2021

Electron scattering off of two localized spin impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
'''

import wfm

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

# top level
plt.style.use('seaborn-dark-palette');
colors = seaborn.color_palette("dark");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5
get_data = int(sys.argv[1]);

# dispersion relation for tight binding
def E_disp(k,a,t):
    # vectorized conversion from k to E(k), measured from bottom of band
    return -2*t*np.cos(k*a);

def k_disp(E,a,t):
    return np.arccos(E/(-2*t))/a;

def construct_h_cicc(J, t, i1, i2, Nsites):
    '''
    construct hams
    formalism works by
    1) having 3 by 3 block's each block is differant site for itinerant e
          H_LL T    0
          T    H_SR T
          0    T    H_RL        T is hopping between leads and scattering region
    2) all other dof's encoded into blocks

    Args:
    - J, float, eff heisenberg coupling
    - t, float, hopping btwn sites
    = i1, int, site of 1st imp
    - i2, int, site of 2nd imp
    - Nsites, int, total num sites in SR
    '''
    
    # heisenberg interaction matrices
    Se_dot_S1 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,1,0,0,0,0,0,0],
                        [0,0,-1,0,2,0,0,0],
                        [0,0,0,-1,0,2,0,0],
                        [0,0,2,0,-1,0,0,0],
                        [0,0,0,2,0,-1,0,0],
                        [0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,1] ]);

    Se_dot_S2 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-1,0,0,2,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,-1,0,0,2,0],
                        [0,2,0,0,-1,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,2,0,0,-1,0],
                        [0,0,0,0,0,0,0,1] ]);

    # insert these local interactions
    h_cicc =[];
    for sitei in range(Nsites): # iter over all sites
        if(sitei == i1):
            h_cicc.append(Se_dot_S1);
        elif(sitei == i2):
            h_cicc.append(Se_dot_S2);
        else:
            h_cicc.append(np.zeros_like(Se_dot_S1) );
    h_cicc = np.array(h_cicc);

    # hopping connects like spin orientations only, ie is identity
    tl_arr = []
    for sitei in range(Nsites-1):
        tl_arr.append(t*np.eye(*np.shape(Se_dot_S1)) );
    tl_arr = np.array(tl_arr);

    return h_cicc, tl_arr;


##################################################################################
#### test production of cicc hamiltonian

if False:
    
    # siam inputs
    Nsites = 2; # total num SR sites
    i1, i2 = 1, 2; # locations of impurities
    tl = 1.0;
    Vg = 10;
    Jeff = 2*tl*tl/Vg; # eff heisenberg # double check

    # souce specifies which basis vector is boundary condition
    sourcei = 3; # incident up, imps + down, down

    # make diag and off diag block matrices
    h_cicc, tl_arr = construct_h_cicc(Jeff, tl, i1, i2, Nsites);

    # test at max verbosity
    testE = 0.001;
    testT = wfm.Tcoef(h_cicc, tl_arr, testE, sourcei, verbose = 5);
    print("\nT(E = "+str(testE)+") = "+str(testT) );
    print(40*"#","\n");


##################################################################################
#### data and plots comparing T vs E, T vs k, T vs kx0
    
if False:

    # siam inputs
    Nsites = 3; # total num SR sites
    i1, i2 = 1, Nsites; # locations of impurities
    tl = 1.0;
    Vg = 10;
    alat = 1.0; # lattice spacing
    Jeff = 2*tl*tl/Vg; # eff heisenberg # double check

    # souce specifies which basis vector is boundary condition
    sourcei = 3; # incident up, imps + down, down

    # make diag and off diag block matrices
    h_cicc, tl_arr = construct_h_cicc(Jeff, tl, i1, i2, Nsites);

    # E, k mesh
    alat = 1.0; # should always cancel for E and kx0
    kmin, kmax = 0.0, np.pi/(10*alat); # should also be near bottom of band
    Npts = 20;
    kvals = np.linspace(kmin, kmax, Npts, dtype = complex); # k mesh
    kx0vals = kvals*(i2-i1)*alat; # k*x0 mesh
    Evals = E_disp(kvals, alat, tl); # E mesh
    Tvals = []

    # get T(E) data
    for Ei in range(len(Evals) ):
        Tvals.append(list(wfm.Tcoef(h_cicc, tl_arr, Evals[Ei], sourcei)) );
    Tvals = np.array(Tvals);

    # check data
    if True:
        # total Sz should be conserved
        Sztot_by_sourcei = np.array([1.5,0.5,0.5,-0.5,0.5,-0.5,-0.5,-1.5]);
        Sztot = Sztot_by_sourcei[sourcei]; # total Sz spec'd by source
        for si in range(np.shape(h_cicc[0])[0]): # iter over all transmitted total Sz states
            if( Sztot_by_sourcei[si] != Sztot): # ie a transmitted state that doesn't conserve Sz
                for Ei in range(len(Tvals)):
                    assert(abs(Tvals[Ei,si]) <= 1e-8 ); # should be zero

    # plot total T at each E, k, kx0
    fig, axes = plt.subplots(3);
    Ttotals = np.sum(Tvals, axis = 1);
    axes[0].scatter(Evals+2*tl, Ttotals, marker = 's');
    axes[1].scatter(kvals, Ttotals, marker = 's');
    axes[2].scatter(kx0vals, Ttotals, marker = 's');
    axes[0].plot(Evals+2*tl, (Evals+2*tl)/Jeff );
    axes[0].plot(Evals+2*tl, Jeff*alat/np.pi *np.sqrt(2/((Evals+2*tl)*2*alat*alat*tl) ) );

    # format and show
    axes[0].set_xlabel("$E + 2t_l$");
    axes[1].set_xlabel("$k$");
    axes[2].set_xlabel("$kx_{0}$");
    axes[0].set_title("Incident up electron");
    axes[0].set_ylabel("$T$");
    for ax in axes:
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();

##################################################################################
#### data and plots for fixed E, k, but varying x0
    
if get_data:

    # siam inputs
    tl = 1.0;
    Vg = 5;
    Jeff = 2*tl*tl/Vg/2; # eff heisenberg

    # cicc inputs
    alat = 1.0; # should always cancel for E and kx0
    m = 1/(2*alat*alat*tl);
    rhoJ_int = 10.0; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJ_int*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = k_disp(E_rho-2*tl, alat, tl); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - E_disp(k_rho,alat, tl) ) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
    print("k*a = ",k_rho*alat);
    print("rho*J = ",np.sqrt(2/(2*alat*alat*tl) *(1/E_rho) )*Jeff/np.pi );
    E_rho = E_rho - 2*tl; # measure from mu

    # choose boundary condition
    sourcei = 1; # incident up, imps + down, down
    spinstate = "aab"
    
    # mesh of x0s (= N0s * alat)
    kx0max = 1.1*np.pi;
    N0max = int(kx0max/(k_rho*alat));
    if verbose: print("N0max = ",N0max);
    N0vals = np.linspace(1, N0max, N0max, dtype = int); # always integer
    kx0vals = k_rho*alat*N0vals;

    # iter over all the differen impurity spacings, get transmission
    Tvals = []
    for N0i in range(len(N0vals)):

        # construct hams
        N0 = N0vals[N0i];
        i1, i2 = 1, 1+N0;
        Nsites = i2+2; # 1 lead site on each side
        print("i1, i2, Nsites = ",i1, i2, Nsites)
        hmats, tmats = construct_h_cicc(Jeff, tl, i1, i2, Nsites);

        # get T from this setup
        Tvals.append(list(wfm.Tcoef(hmats, tmats, E_rho , sourcei)) );

    # package into one array
    Tvals = np.array(Tvals);
    info = np.zeros_like(kx0vals);
    info[0], info[1], info[2], info[3], info[4] = tl, Jeff, rhoJ_int, E_rho, k_rho; # save info we need
    data = [info, N0vals, kx0vals];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "dat/cicc/eff/"+spinstate+"/"
    fname +="gf_E"+str(E_rho)[:6]+"_k"+str(k_rho)[:4]+"_"+str(rhoJ_int)[:4]+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);

else: # plot

    # plot each file given at command line
    fig, axes = plt.subplots();
    axes = [axes];
    datafs = sys.argv[2:];
    for fi in range(len(datafs)):

        # unpack
        data = np.load(datafs[fi]);
        tl, Jeff, rhoJ_int, E_rho, k_rho = data[0,0], data[0,1], data[0,2], data[0,3], data[0,4],
        N0vals, kx0vals = data[1], data[2];
        Tvals = data[3:];

        # convert T
        Ttotals = np.sum(Tvals, axis = 0);

        # plot
        axes[0].scatter(kx0vals, Ttotals, marker = 's', label = "$\\rho J = $"+str(np.real(rhoJ_int)));

    # format and show
    axes[0].axvline(np.pi, color = "black", linestyle = "dashed");
    axes[0].axvline(2*np.pi, color = "black", linestyle = "dashed");
    axes[0].set_ylim(0.0,1.05);
    axes[0].set_xlabel("$kx_{0}$");
    axes[0].set_title("Up electron scattering from up, down impurities");
    axes[0].set_ylabel("$T$");
    for ax in axes:
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.legend();
    plt.show();
        
