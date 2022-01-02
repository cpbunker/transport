'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of single electron off of two localized spin-1/2 impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
'''

from transport import wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# top level
plt.style.use('seaborn-dark-palette');
np.set_printoptions(precision = 4, suppress = True);
verbose = 5


##################################################################################
#### data and plots for cicc Fig 2b
    
if False: # original version of 2b (varying x0 by varying N)

    # tight binding params
    tl = 1.0;
    Jeff = 0.1;

    # cicc inputs
    rhoJa = 1.0; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = np.arccos((E_rho - 2*tl)/(-2*tl)); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - -2*tl*np.cos(k_rho)) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
    print("k*a = ",k_rho); # a =1
    print("rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
    E_rho = E_rho - 2*tl; # measure from mu

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";
    
    # mesh of x0s (= N0s * alat)
    kx0max = 2.1*np.pi;
    N0max = int(kx0max/(k_rho)); # a = 1
    if verbose: print("N0max = ",N0max);
    N0vals = np.linspace(2, N0max, 49, dtype = int); # always integer
    kx0vals = k_rho*N0vals; # a = 1

    # iter over all the differen impurity spacings, get transmission
    Tvals = []
    for N0 in N0vals:

        # construct hams
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, 1, N0, N0+2);

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho , source));

    # package into one array
    Tvals = np.array(Tvals);
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    info = np.zeros_like(kx0vals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, k_rho; # save info we need
    data = [info, kx0vals];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "dat/cicc/"+spinstate+"/";
    fname +="N_rhoJa"+str(int(rhoJa))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);
    raise(Exception);


if False: # vary kx0 by varing k at fixed N

    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # cicc inputs
    N_SR = 100 #100,199,989; # num sites in SR
                # N_SR = 99, J = 0.1 gives rhoJa \approx 1, Na \approx 20 angstrom
    ka0 = np.pi/(N_SR-1); # val of ka (dimensionless) s.t. kx0 = ka*N_SR = pi
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho))

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJa);

    # construct blocks of hamiltonian
    # num sites in SR, imps will be located at site 0, N_SR-1
    hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, 0, N_SR-1, N_SR);
        
    # get data
    # wfm.Data(): tblocks is hopping in SR, then th = hopping onto SR, then tl = in leads
    kalims = (0.0*ka0,2.1*ka0);
    kavals, Tvals = wfm.Data(source, np.zeros_like(hblocks[0]),-tl*np.eye(np.shape(hblocks[0])[0]),
                         hblocks, tblocks, np.zeros_like(hblocks[0]), tl, kalims, numpts = 49, retE = False, verbose = verbose);
    
    # package into one array
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    info = np.zeros_like(kavals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, ka0; # save info we need
    data = [info, kavals*(N_SR-1)];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of kaval, corresponding T vals
    # save data
    fname = "dat/cicc/"+spinstate+"/";
    fname +="ka_rhoJa"+str(int(rhoJa))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);
    raise(Exception);
    

if False: # vary kx0 by varying t', N and ka fixed
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1; # eff heisenberg

    # cicc quantitites
    N_SR = 100;
    kap0 = np.pi/(N_SR - 1); # a' is length defined by hopping t' btwn imps
                            # ka' = ka'0 = ka0 when t' = t so a' = a
    E_rho = 2*tl-2*tl*np.cos(kap0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka'0 = ",kap0);
        print("- rho*J*a = ", rhoJa);
    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # get data
    kaplims = (0.95*kap0,1.05*kap0);
    kapvals = np.linspace(*kaplims, 9);
    tpvals = tl*kap0*kap0/(kapvals*kapvals);
    Tvals = [];
    for tpi in range(len(tpvals)):

        # construct blocks of hamiltonian
        # now need to have 0's on each end !!!
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tpvals[tpi], 1, N_SR, N_SR+2);
        #if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks);

        # get data
        Energy = -2*tl*np.cos(kap0);
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, Energy, source));
    Tvals = np.array(Tvals);

    # plot
    Ttotals = np.sum(Tvals, axis = 1);
    plt.plot(kapvals*(N_SR - 1)/np.pi, Ttotals);
    plt.show();
    raise(Exception);


if False: # vary kx0 by varying Vgate
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1; # eff heisenberg

    # cicc quantitites
    N_SR = 100;
    ka0 = np.pi/(N_SR - 1); # a' is length defined by hopping t' btwn imps
                            # ka' = ka'0 = ka0 when t' = t so a' = a
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJa);
    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # get data
    kalims = (0.0*ka0,2.1*ka0);
    kavals = np.linspace(*kalims, 9);
    Vgvals = E_rho - tl*kavals*kavals;
    Tvals = [];
    for Vg in Vgvals:

        # construct blocks of hamiltonian
        # now need to have 0's on each end !!!
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, 1, N_SR, N_SR+2);
        for blocki in range(len(hblocks)): # add Vg in SR
            if(blocki > 0 and blocki < N_SR + 1): # if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                
        # get data
        Energy = -2*tl*np.cos(ka0);
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, Energy, source));
    Tvals = np.array(Tvals);

    # package into one array
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    info = np.zeros_like(kavals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, ka0; # save info we need
    data = [info, kavals*(N_SR-1)];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of kaval, corresponding T vals
    # save data
    fname = "dat/cicc/"+spinstate+"/";
    fname +="mu_rhoJa"+str(int(rhoJa))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);
    raise(Exception);
    
        
if True: # plot fig 2b data

    # plot each file given at command line
    fig, axes = plt.subplots();
    axes = [axes];
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):

        # unpack
        print("Loading data from "+datafs[fi]);
        data = np.load(datafs[fi]);
        tl, Jeff, rhoJa, k_rho = data[0,0], data[0,1], data[0,2], data[0,3];
        kNavals = data[1];
        Tvals = data[2:];

        # convert T
        Ttotals = np.sum(Tvals, axis = 0);

        # plot
        axes[0].plot(kNavals/np.pi, Ttotals, label = "$\\rho  \, J a= $"+str(int(rhoJa)));

    # format and show
    axes[0].set_xlim(0.0,2.1);
    axes[0].set_ylim(0.0,1.05);
    axes[0].set_xticks([0,1,2]);
    axes[0].set_xlabel("$k'a(N-1)/\pi$", fontsize = "x-large");
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    plt.legend(loc = "upper left", fontsize = "large");
    plt.savefig("my_fig_2a");


##################################################################################
#### N_SR = 2 calcs

if False: # vary kx0 by varying Vgate
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1; # eff heisenberg

    # cicc quantitites
    N_SR = 2;
    factor = 100;
    ka0 = np.pi/(N_SR - 1)/factor; # a' is length defined by hopping t' btwn imps
                            # ka' = ka'0 = ka0 when t' = t so a' = a
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJa);
    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # get data
    kalims = (0.0*ka0,(factor/1)*ka0); # first 1/10th of the zone
    kavals = np.linspace(*kalims, 499);
    Vgvals =  -2*tl*np.cos(ka0) + 2*tl*np.cos(kavals);
    Tvals = [];
    for Vg in Vgvals:

        # construct blocks of hamiltonian
        # now need to have 0's on each end !!!
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl*0.4, 1, N_SR, N_SR+2);
        for blocki in range(len(hblocks)): # add Vg in SR
            if(blocki > 0 and blocki < N_SR + 1): # if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                
        # get data
        Energy = -2*tl*np.cos(ka0);
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, Energy, source));
    Tvals = np.array(Tvals);

    # plot
    Ttotals = np.sum(Tvals, axis = 1);
    plt.plot(kavals/np.pi, Ttotals);
    plt.show();
    raise(Exception);


if False: # vary kx0 by varing k at fixed N, t' != t

    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1;
    tp = 0.9

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";

    # cicc inputs
    N_SR = 179 #100,199,989; # num sites in SR
                # N_SR = 99, J = 0.1 gives rhoJa \approx 1, Na \approx 20 angstrom
    kap0 = np.pi/(N_SR-1); # val of ka' (dimensionless) s.t. ka'(N_SR-1)=pi
    ka0 = kap0*np.sqrt(tp/tl)
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJap = Jeff/(np.pi*np.sqrt(tp*E_rho))

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJap);

    # construct blocks of hamiltonian
    # num sites in SR, imps will be located at site 0, N_SR-1
    hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tp, 0, N_SR-1, N_SR);
        
    # get data
    # wfm.Data(): tblocks is hopping in SR, then th = hopping onto SR, then tl = in leads
    kalims = (0.0*ka0,(N_SR-1)*ka0/5);
    kavals, Tvals = wfm.Data(source, np.zeros_like(hblocks[0]),-tl*np.eye(np.shape(hblocks[0])[0]),
                         hblocks, tblocks, np.zeros_like(hblocks[0]), tl, kalims, numpts = 49, retE = False, verbose = verbose);

    #
    Ttotals = np.sum(Tvals, axis = 1);
    plt.plot(kavals*(N_SR-1)/np.pi, Ttotals);
    plt.axvline(ka0*(N_SR-1)/np.pi);
    plt.show();
    raise(Exception);
    

if True: # cicc fig 6 / my fig 3

    # siam inputs
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # iter over rhoJ, getting T
    Tvals = [];
    rhoJvals = np.linspace(0.05,2.5,40)
    for rhoJa in rhoJvals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
        print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
        print("ka = ",k_rho);
        print("rhoJa = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
        E_rho = E_rho - 2*tl; # measure from mu
        
        # location of impurities, fixed by kx0 = pi
        kx0 = 0*np.pi;
        N0 = max(1,int(kx0/(k_rho)));
        if verbose: print("N0 = ",N0);

        # construct hams
        i1, i2 = 1, 1+N0;
        Nsites = i2+2; # 1 lead site on each side
        print("i1, i2, Nsites = ",i1, i2, Nsites)
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, Nsites);

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho , source));

    # plot
    fig, ax = plt.subplots();
    for el in range(3): ax.plot([-1],[-1]);
    Tvals = np.array(Tvals);
    ax.plot(rhoJvals, Tvals[:,4], label = "$|i\,>$");
    ax.plot(rhoJvals, Tvals[:,1]+Tvals[:,2], label = "$|+>$");
    ax.legend(loc = "upper left", fontsize = "large");

    # inset
    rhoEvals = Jeff*Jeff/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    axins = inset_axes(ax, width="40%", height="40%");
    for el in range(4): axins.plot([-1],[-1]);
    axins.plot(rhoEvals,Tvals[:,1]);
    axins.set_xlabel("$E+2*t_l$", fontsize = "x-large");
    axins.set_ylabel("$T_+$", fontsize = "x-large");
    axins.set_xlim(-0.01,0.4);
    axins.set_ylim(0,0.15);

    # format
    ax.set_xlabel("$\\rho\,J a$", fontsize = "x-large");
    ax.set_ylabel("$T$", fontsize = "x-large");
    ax.set_xlim(0.05,2.5);
    ax.set_ylim(0,1.05);
    plt.savefig("my_fig_3");
    raise(Exception);
    








#################################################################
#### misc

if False: # resonant Rabi with Jz = 0

    # tight binding params
    tl = 1.0;
    Jeff = 0.4;

    # cicc inputs
    rhoJa = 0.5; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = np.arccos((E_rho - 2*tl)/(-2*tl)); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - -2*tl*np.cos(k_rho)) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E = ",E_rho,"\nka = ",k_rho,"\nE/J = ",E_rho/Jeff,"\nrho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));

    Omega = k_rho*Jeff/(2*E_rho);
    Jtau = np.arccos(np.power(1+Omega*Omega,-1/2));
    print("Omega = ",Omega,"\nJtau = ",Jtau);
    
    # choose boundary condition
    source = np.zeros(8);
    source[4] = 1; # down up up
    
    # mesh of x0s (= N0s * alat)
    kx0max = 1.0*np.pi;
    N0max = int(kx0max/(k_rho)); # a = 1
    print("N0max = ",N0max);
    N0vals = np.linspace(0, N0max, 49, dtype = int); # always integer

    # iter over all the differen impurity spacings, get transmission
    Tvals = []
    for N0 in N0vals:

        # construct hams
        hblocks, tblocks = wfm.utils.h_cicc_hacked(Jeff, tl, N0+2);
        
        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho , source));

    # plot
    Tvals = np.array(Tvals);
    fig, axes = plt.subplots(2);
    axes[0].plot(N0vals, Tvals[:,4], label = "$|i\,>$");
    axes[0].plot(N0vals, Tvals[:,1]+Tvals[:,2], label = "$|+>$");
    axes[0].set_xlabel("$k(N-1)a/\pi$");
    #axes[1].plot(np.arccos(np.power(1+np.power(k_rho,2),-1/2)), Tvals[:,1]+Tvals[:,2]);
    plt.legend();
    plt.show();
    raise(Exception);
        


if False: # vary kx0 by varying t'
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    tp = 0.8; # hyb between imps
    Jeff = 0.1; # eff heisenberg

    # cicc quantitites
    
    for Np in [8,6,2,1]:
        fig, ax = plt.subplots();
        kaN = np.pi/Np
        ka0 = kaN*np.sqrt(tp/tl); # val of ka s.t. ka' = pi
                                    # a' is length defined by hopping t' btwn imps
        E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                        # measured from bottom of the band!!
        rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

        # diagnostic
        if(verbose):
            print("\nCiccarello inputs")
            print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
            print(" - ka0 = ",ka0);
            print("- rho*J*a = ", rhoJa);
        # choose boundary condition
        source = np.zeros(8);
        source[1] = 1/np.sqrt(2);
        source[2] = -1/np.sqrt(2);

        # construct blocks of hamiltonian
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tp, 0, Np, Np+1);
        #hblocks = np.append([np.zeros_like(hblocks[0])], hblocks, axis=0);
        #hblocks = np.append(hblocks, [np.zeros_like(hblocks[0])], axis=0);
        if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks);

        # get data
        kalims = (0*kaN,np.pi);
        kavals, Tvals = wfm.Data(source, np.zeros_like(hblocks[0]),-tl*np.eye(np.shape(hblocks[0])[0]),
                                 hblocks, tblocks, np.zeros_like(hblocks[0]), tl, kalims, numpts = 49, retE = False, verbose = verbose)
        Ttotals = np.sum(Tvals, axis = 1);
        kapvals = kavals*np.sqrt(tl/tp); # convert ka to ka'
        
        # plot data
        ax.plot(kavals/np.pi, Ttotals, label = Np);#"$\\rho Ja$ = "+str(int(10*rhoJa)/10));
        #if tp == 1.0: ax.axvline(Np*np.sqrt(tl/tp)/np.pi, linestyle = "dashed", color = "black", label = "$kNa'=\pi$");

        # format and show
        ax.set_xlabel("$ka/\pi$");
        ax.set_ylabel("$T$");
        ax.set_ylim(0,1.05);
        ax.set_title(Np);
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        plt.legend(loc = "upper left");
        plt.show();
    
