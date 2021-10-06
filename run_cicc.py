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


##################################################################################
#### data and plots for fixed E, k, but varying x0
#### effective ham, choose source for initial condition
#### ie replicate Cicc figs 1, 5
    
if False:

    # siam inputs
    tl = 1.0;
    Vg = 5;
    Jeff = 2*tl*tl/Vg/2; # eff heisenberg

    # cicc inputs
    alat = 1.0; # should always cancel for E and kx0
    m = 1/(2*alat*alat*tl);
    rhoJ_int = 1.0; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJ_int*rhoJ_int*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = wfm.k_disp(E_rho-2*tl, alat, tl); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - wfm.E_disp(k_rho,alat, tl) ) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
    print("k*a = ",k_rho*alat);
    print("rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
    E_rho = E_rho - 2*tl; # measure from mu

    # choose boundary condition
    source = np.zeros(8);
    source[0] = 1;
    spinstate = "aaa"
    
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
        hmats, tmats = wfm.h_cicc_eff(Jeff, tl, i1, i2, Nsites);

        # get T from this setup
        Tvals.append(list(wfm.Tcoef(hmats, tmats, tl, E_rho , source)) );

    # package into one array
    Tvals = np.array(Tvals);
    info = np.zeros_like(kx0vals);
    info[0], info[1], info[2], info[3], info[4] = tl, Jeff, rhoJ_int, E_rho, k_rho; # save info we need
    data = [info, N0vals, kx0vals];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "dat/cicc/eff/"+spinstate+"/";
    fname +="gf_E"+str(E_rho)[:6]+"_k"+str(k_rho)[:4]+"_"+str(rhoJ_int)[:4]+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);

if False: # plot

    # plot each file given at command line
    fig, axes = plt.subplots();
    axes = [axes];
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):

        # unpack
        data = np.load(datafs[fi]);
        tl, Jeff, rhoJ_int, E_rho, k_rho = data[0,0], data[0,1], data[0,2], data[0,3], data[0,4],
        N0vals, kx0vals = data[1], data[2];
        Tvals = data[3:];

        # convert T
        Ttotals = np.sum(Tvals, axis = 0);

        # plot
        axes[0].plot(kx0vals/np.pi, Ttotals, label = "$\\rho  \, J a= $"+str(np.real(rhoJ_int)));

    # format and show
    #axes[0].axvline(np.pi, color = "black", linestyle = "dashed");
    #axes[0].axvline(2*np.pi, color = "black", linestyle = "dashed");
    axes[0].set_ylim(0.0,1.05);
    axes[0].set_xlabel("$kx_{0}/\pi$");
    #axes[0].set_title("Up electron scattering from $\Psi^+ $ impurities, J = "+str(np.real(Jeff)); );
    axes[0].set_ylabel("$T$");
    for ax in axes:
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.legend();
    plt.show();


##################################################################################
#### replicate cicc Fig 6 ie Tdown vs rhoJ

if False:

    # siam inputs
    tl = 1.0;
    Vg = 20;
    Jeff = 2*tl*tl/Vg; # eff heisenberg strength fixed

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[3] = 1;
    spinstate = "abb"

    # cicc inputs
    alat = 1.0; # should always cancel for E and kx0
    m = 1/(2*alat*alat*tl);

    # iter over rhoJ, getting T
    Tvals = [];
    rhoJvals = np.linspace(0.5,5.5,44)
    for rhoJ_int in rhoJvals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJ_int*rhoJ_int*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        k_rho = wfm.k_disp(E_rho-2*tl, alat, tl); # input E measured from 0 by -2*tl
        assert(abs((E_rho - 2*tl) - wfm.E_disp(k_rho,alat, tl) ) <= 1e-8 ); # check by getting back energy measured from bottom of band
        print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
        print("k*a = ",k_rho*alat);
        print("rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
        E_rho = E_rho - 2*tl; # measure from mu
        
        # location of impurities, fixed by kx0 = pi
        kx0 = np.pi;
        N0 = int(kx0/(k_rho*alat));
        if verbose: print("N0 = ",N0);

        # construct hams
        i1, i2 = 1, 1+N0;
        Nsites = i2+2; # 1 lead site on each side
        print("i1, i2, Nsites = ",i1, i2, Nsites)
        hmats, tmats = wfm.h_cicc_eff(Jeff, tl, i1, i2, Nsites);

        # get T from this setup
        Tvals.append(list(wfm.Tcoef(hmats, tmats, tl, E_rho , source)) );

    # save results
    Tvals = np.array(Tvals); # rows are diff rhoJ vals
    info = np.zeros_like(rhoJvals);
    info[0], info[1], info[2], info[3] = tl, Jeff, N0, kx0; # save info we need
    data = [info, rhoJvals]; # now want each column of data to be a rhoJ val
    for Ti in range(np.shape(Tvals)[1]): # all dofs
        data.append(Tvals[:,Ti]); # row of one dof, but each column is diff rhoJ
    # save data
    fname = "dat/cicc_eff/Fig6/"
    fname +="gf_"+spinstate+"_kx0"+str(kx0)[:4]+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);

if True: # plot above

    # load data
    fname = sys.argv[1]
    if verbose: print("Loading data from "+fname);
    data = np.load(fname);
    info = data[0];
    tl, Jeff, N0, kx0 = info[0], info[1], info[2], info[3];
    rhoJvals = data[1];
    Tvals = data[2:];
    Tup = Tvals[0] + Tvals[1] + Tvals[2] + Tvals[3];
    Tdown = Tvals[4] + Tvals[5] + Tvals[6] + Tvals[7];

    # plot
    fig, ax = plt.subplots();
    ax.scatter(rhoJvals, Tup, marker = 's', label = "$T_{up}$");
    ax.scatter(rhoJvals, Tdown, marker = 's', label = "$T_{down}$");
    ax.set_xlabel("$\\rho\,J a$");
    ax.set_ylabel("$T$");
    #ax.set_title("Fig. 6");
    #ax.set_ylim(0.0,0.25);
    ax.legend();
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();
    
    
    
