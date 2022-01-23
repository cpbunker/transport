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
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# top level
plt.style.use('seaborn-dark-palette');
np.set_printoptions(precision = 4, suppress = True);
verbose = 5


##################################################################################
#### data and plots for cicc Fig 2b (transparency)
    
if False: # original version of 2b (varying x0 by varying N)

    # tight binding params
    tl = 1.0;
    Jeff = 0.1;

    # cicc inputs
    rhoJa = 10.0; # integer that cicc param rho*J is set equal to
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
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
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

if False: # vary kx0 by varying Vgate
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1; # eff heisenberg

    # cicc quantitites
    N_SR = 199 #988;
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
    kavals = np.linspace(*kalims, 299);
    Vgvals = -2*tl*np.cos(ka0) + 2*tl*np.cos(kavals);
    Tvals = [];
    for Vg in Vgvals:

        # construct blocks of hamiltonian
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
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
    fname +="Vg_rhoJa"+str(int(rhoJa))+".npy";
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
    # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
    hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, 1, N_SR, N_SR+2);
    raise Exception("switch from Data() to kernel()")
        

    
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
        
if False: # plot fig 2b data

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
        print(">>>",np.shape(Ttotals));

        # plot
        axes[0].plot(kNavals/np.pi, Ttotals, label = "$\\rho  \, J a= $"+str(int(rhoJa)));

    # format and show
    axes[0].set_xlim(0.0,2.1);
    axes[0].set_ylim(0.0,1.05);
    axes[0].set_xticks([0,1,2]);
    axes[0].set_xlabel("$k'a(N-1)/\pi$", fontsize = "x-large");
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    plt.legend(loc = "upper left", fontsize = "large");
    plt.show();
    raise(Exception);

##################################################################################
#### molecular dimer regime (N = 2 fixed)

if False: # vary k'x0 by varying Vg for low energy detection, t', th != t;

    # incident state
    theta_param = 3*np.pi/4;
    phi_param = 0;
    source = np.zeros(8);
    source[1] = np.cos(theta_param);
    source[2] = np.sin(theta_param);
    spinstate = "psimin";

    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom
    Jeff = 0.1; # eff heisenberg
    th = 1.0
    tp = 1.0
    N_SR = 2;

    factor = 100;
    ka0 =  np.pi/(N_SR - 1)/factor; # free space wavevector, should be << pi
                                    # increasing just broadens the Vg peak
    kpa0 = np.pi/(N_SR - 1)/factor; # wavevector in gated SR
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
        print(" - ka0/pi = ",ka0/np.pi);
        print("- rho*J*a = ", rhoJa);

    # get data
    kpalims = (0.0*kpa0,(factor/2)*kpa0); # k'a in first 1/2 of the zone
    kpavals = np.linspace(*kpalims, 99);
    Vgvals =  -2*tl*np.cos(ka0) + 2*tp*np.cos(kpavals);
    Tvals = [];
    for Vg in Vgvals:

        # construct blocks of hamiltonian
        # t's vary, so have to construct hblocks, tblocks list by hand
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tp, 0, N_SR-1, N_SR);
        hblocks = np.append([np.zeros_like(hblocks[0])], hblocks, axis = 0); # LL block
        hblocks = np.append(hblocks, [np.zeros_like(hblocks[0])], axis = 0); # RL block
        tblocks = np.append([-th*np.eye(np.shape(hblocks)[1])], tblocks, axis = 0); # V hyb
        tblocks = np.append(tblocks,[-th*np.eye(np.shape(hblocks)[1])], axis = 0);
        for blocki in range(len(hblocks)): # add Vg in SR
            if(blocki > 0 and blocki < N_SR + 1): # gate voltage if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                
        # get data
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, -2*tl*np.cos(ka0), source));
    
    Tvals = np.array(Tvals);
    Ttotals = np.sum(Tvals, axis = 1);

    # plot
    fig, axes = plt.subplots(2)
    axes[0].plot(kpavals/np.pi, Ttotals);
    axes[1].plot(Vgvals, Ttotals);
    axes[0].set_xlabel("$k'a/\pi$", fontsize = "x-large");
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    axes[1].axvline(-2*tl*np.cos(ka0) + 2*tp, color = "black");
    plt.show();
    del Vg, Tvals, Ttotals, fig, axes;

    #### vary theta, phi
    #### -> detection !
    if((th == tl) and (tp == tl)):
        myVg = -2*tl*np.cos(ka0) + 2*tp;
    else:
        myVg = -0.80
    thetavals = np.linspace(0, np.pi, 49);
    phivals = np.linspace(0, np.pi, 49);
    Ttotals = np.zeros((len(thetavals), len(phivals)));

    # construct blocks of hamiltonian
    # lead blocks and Vhybs not filled yet !
    hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tp, 0, N_SR-1, N_SR);
    hblocks = np.append([np.zeros_like(hblocks[0])], hblocks, axis = 0); # LL block
    hblocks = np.append(hblocks, [np.zeros_like(hblocks[0])], axis = 0); # RL block
    tblocks = np.append([-th*np.eye(np.shape(hblocks)[1])], tblocks, axis = 0); # V hyb
    tblocks = np.append(tblocks,[-th*np.eye(np.shape(hblocks)[1])], axis = 0);
    for blocki in range(len(hblocks)): # add Vg in SR
        if(blocki > 0 and blocki < N_SR + 1): # gate voltage if in SR
            hblocks[blocki] += myVg*np.eye(np.shape(hblocks[0])[0])

    # iter over entanglement space
    for ti in range(len(thetavals)):
        for pi in range(len(phivals)):

            thetaval = thetavals[ti];
            phival = phivals[pi];
	
            source = np.zeros(8, dtype = complex);
            source[1] = np.cos(thetaval);
            source[2] = np.sin(thetaval)*np.exp(complex(0,phival));

            # get data
            Ttotals[ti, pi] = sum(wfm.kernel(hblocks, tblocks, tl, -2*tl*np.cos(ka0), source));

    # plot
    fig = plt.figure();
    ax = fig.add_subplot(projection = "3d");
    thetavals, phivals = np.meshgrid(thetavals, phivals);
    ax.plot_surface(thetavals/np.pi, phivals/np.pi, Ttotals.T, cmap = cm.coolwarm);
    ax.set_xlabel("$\\theta/\pi$", fontsize = "x-large");
    ax.set_ylabel("$\phi/\pi$", fontsize = "x-large");
    ax.set_xlim(0,1);
    ax.set_xticks([0,1/4,1/2,3/4,1]);
    ax.set_ylim(0,1);
    ax.set_yticks([0,1/4,1/2,3/4,1]);
    ax.set_zlim(0,1);
    ax.set_zticks([0,1]);
    print(tl, th, tp, myVg);
    plt.show();
    raise(Exception);


# still in dimer case, compare T vs rho J a peak under resonant
plot_6_resonant = False;
if plot_6_resonant: # cicc fig 6 (N = 2 still)

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
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2);

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho , source));

    # plot
    fig, ax = plt.subplots();
    Tvals = np.array(Tvals);
    ax.plot(rhoJvals, Tvals[:,4], label = "$|i\,>$", color = "y");
    ax.plot(rhoJvals, Tvals[:,1]+Tvals[:,2], label = "$|+>$", color = "indigo");
    ax.legend(loc = "upper left", fontsize = "large");

    # inset
    rhoEvals = Jeff*Jeff/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    axins = inset_axes(ax, width="50%", height="50%");
    axins.plot(rhoEvals,Tvals[:,1], color = "indigo");
    axins.set_xlabel("$E+2t_l$", fontsize = "x-large");
    axins.set_xlim(-0.01,0.4);
    axins.set_ylim(0,0.15);

    # format
    ax.set_xlabel("$\\rho\,J a$", fontsize = "x-large");
    ax.set_ylabel("$T$", fontsize = "x-large");
    ax.set_xlim(0.05,2.5);
    ax.set_ylim(0,1.05);
    if not plot_6_resonant:
        plt.show();


if plot_6_resonant: # add fig 6 data at switching resonance (Jz = 0)

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

    # modulate N, with both imps on all sites in SR, in the old way
    if not plot_6_resonant:
        
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
            # hacked = no Jz
            # when dimer=False, puts both imp hams on all SR sites, as in old way
            hblocks, tblocks = wfm.utils.h_cicc_hacked(Jeff, tl, N0+2, dimer = False);
            
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

        del hblocks, tblocks;

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
        hblocks, tblocks = wfm.utils.h_cicc_hacked(Jeff, tl, Nsites, dimer = True);

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho , source));

    # plot with and without Jz against rhoJa to compare
    if not plot_6_resonant:
        fig, ax = plt.subplots();
        axins = inset_axes(ax, width="50%", height="50%");
    Tvals = np.array(Tvals);
    ax.plot(rhoJvals, Tvals[:,4], color = "y", linestyle = "dashed");
    ax.plot(rhoJvals, Tvals[:,1]+Tvals[:,2], color = "indigo", linestyle = "dashed");
    ax.legend(loc = "upper left", fontsize = "large");

    # inset
    rhoEvals = Jeff*Jeff/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    axins.plot(rhoEvals,Tvals[:,1], color = "indigo", linestyle = "dashed");
    axins.set_xlabel("$E+2t_l$", fontsize = "x-large");
    xlim, ylim = (0,0.01), (0.05,0.15);
    axins.set_xlim(*xlim);
    axins.set_xticks([*xlim]);
    axins.set_ylim(*ylim);
    axins.set_yticks([*ylim]);

    # format
    ax.set_xlabel("$\\rho\,J a$", fontsize = "x-large");
    ax.set_ylabel("$T$", fontsize = "x-large");
    ax.set_xlim(0.05,2.5);
    ax.set_ylim(0,1.05);
    plt.show();
    
