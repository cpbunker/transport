'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 2:
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

# entanglement detection for nonlinear dimers
if False: 

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
        # t's vary, so have to construct by hand
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tp, 0, N_SR-1, N_SR);
        hblocks = np.append([np.zeros_like(hblocks[0])], hblocks, axis = 0); # LL block
        hblocks = np.append(hblocks, [np.zeros_like(hblocks[0])], axis = 0); # RL block
        for blocki in range(len(hblocks)): # add Vg in SR
            if(blocki > 0 and blocki < N_SR + 1): # gate voltage if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0]);

        # hopping
        tnn = np.append([-th*np.eye(len(source))], tnn, axis = 0); # V hyb
        tnn = np.append(tnn,[-th*np.eye(len(source))], axis = 0);
        tnnn = np.zeros_like(tnn)[:-1];
        tnnn = tnn[:-1]; # next nearest neighbor hopping allowed !
                
        # get data
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(ka0), source));
    
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
    assert False;

    #### vary theta, phi
    #### -> detection !
    if((th == tl) and (tp == tl)):
        myVg = -2*tl*np.cos(ka0) + 2*tp;
    else:
        myVg = -0.80;
    print(">>> myVg = ",myVg);
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
    ax.set_xlim(0,1);
    ax.set_xticks([0,1/4,1/2,3/4,1]);
    ax.set_ylabel("$\phi/\pi$", fontsize = "x-large");
    ax.set_ylim(0,1);
    ax.set_yticks([0,1/4,1/2,3/4,1]);
    ax.set_xlabel("$\\theta/\pi$", fontsize = "x-large");
    ax.set_zlim(0,1);
    ax.set_zticks([0,1]);
    ax.set_zlabel("$T$");
    plt.show();
    raise(Exception);


# entanglement generation for nonlinear dimers - not much interesting so far
if True: 

    # siam inputs
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "abb"

    # iter over rhoJ, getting T
    Tvals = [];
    xlims = 0.05, 4.0;
    rhoJvals = np.linspace(xlims[0], xlims[1], 299);
    #Elims = (1.0,2);
    #Evals = np.linspace(Elims[0], Elims[1], 9);
    #rhoJvals = Jeff/(np.pi*np.sqrt(tl*Evals));
    #print(Evals,"\n",rhoJvals); 
    for rhoJa in rhoJvals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        #E_rho = Evals[Ei]
        k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
        print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
        print("ka = ",k_rho);
        print("rhoJa = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
        E_rho = E_rho - 2*tl; # measure from mu
        
        # location of impurities, fixed by kx0 = pi
        kx0 = 0*np.pi;
        N0 = max(1,int(kx0/(k_rho)));
        assert(N0 == 1);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2);
        tnnn = np.zeros_like(tnn)[:-1];
        tnnn = tnn[:-1]; # next nearest neighbor hopping allowed !

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source));

    # plot
    fig, ax = plt.subplots();
    Tvals = np.array(Tvals);
    ax.plot(rhoJvals, Tvals[:,4], label = "$|i\,>$", color = "black");
    ax.plot(rhoJvals, Tvals[:,1]+Tvals[:,2], label = "$|+>$", color = "black", linestyle = "dashed");

    # inset
    rhoEvals = Jeff*Jeff/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    axins = inset_axes(ax, width="50%", height="50%");
    #axins.plot(rhoEvals,Tvals[:,4], color = "black"); # i state
    axins.plot(rhoEvals,Tvals[:,1]+Tvals[:,2], color = "black", linestyle = "dashed"); # + state
    axins.set_xlim(0,0.4);
    axins.set_xticks([0,0.4]);
    axins.set_xlabel("$E+2t_l$", fontsize = "x-large");
    axins.set_ylim(0,0.2);
    axins.set_yticks([0,0.2]);

    # format
    #ax.set_xlim(*xlims);
    #ax.set_xticks([0,1,2,3,4]);
    ax.set_xlabel("$\\rho\,J a$", fontsize = "x-large");
    ax.set_ylim(0,1.0);
    ax.set_yticks([0,1]);
    ax.set_ylabel("$T$", fontsize = "x-large");
    plt.show();
    raise(Exception);

