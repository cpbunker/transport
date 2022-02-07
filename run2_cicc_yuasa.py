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

# Yuasa: Jz = 0
if True: 

    # cicc inputs
    if False:
        rhoJa = 0.5; # integer that cicc param rho*J is set equal to
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        k_rho = np.arccos((E_rho - 2*tl)/(-2*tl)); # input E measured from 0 by -2*tl
        assert(abs((E_rho - 2*tl) - -2*tl*np.cos(k_rho)) <= 1e-8 ); # check by getting back energy measured from bottom of band
        print("E = ",E_rho,"\nka = ",k_rho,"\nE/J = ",E_rho/Jeff,"\nrho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));

        Omega = k_rho*Jeff/(2*E_rho);
        Jtau = np.arccos(np.power(1+Omega*Omega,-1/2));
        print("Omega = ",Omega,"\nJtau = ",Jtau);

    # tight binding params
    tl = 1.0;
    Jeff = 0.1;

    # choose boundary condition
    source = np.zeros(8); # incident up, imps = down, down
    source[4] = 1;
    spinstate = "baa"

    # iter over rhoJ, getting T
    Tvals = [];
    xlims = 0.05, 4.0;
    rhoJvals = np.linspace(xlims[0], xlims[1], 99)
    for rhoJa in rhoJvals:

        # energy and K fixed by J, rhoJ
        E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                # this E is measured from bottom of band !!!
        k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
        print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
        print("ka = ",k_rho);
        print("rhoJa = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
        E_rho = E_rho - 2*tl; # measure from mu
        myVg = E_rho # -2*tl*np.cos(k_rho) + 2*tl;

        # ham
        hblocks, tnn = wfm.utils.h_cicc_hacked(Jeff, tl, 4, dimer = True);
        hblocks[1] += myVg*np.eye(len(source));
        hblocks[2] += myVg*np.eye(len(source));
        tnnn = np.zeros_like(tnn)[:-1];

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source));

    # plot with and without Jz against rhoJa to compare
    fig, ax = plt.subplots();
    axins = inset_axes(ax, width="50%", height="50%");
    Tvals = np.array(Tvals);
    ax.plot(rhoJvals, Tvals[:,4], color = "black"); # |i>
    ax.plot(rhoJvals, Tvals[:,1]+Tvals[:,2], color = "black", linestyle = "dashed"); # |+>
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
    ax.set_xlim(*xlims);
    ax.set_ylim(0,1.0);
    plt.show();
    
