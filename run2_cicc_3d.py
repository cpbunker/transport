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
    rhoJa = 1.0; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = np.arccos((E_rho - 2*tl)/(-2*tl)); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - -2*tl*np.cos(k_rho)) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
    print("k*a = ",k_rho); # a = 1
    print("rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
    E_rho = E_rho - 2*tl; # measure from mu

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1/np.sqrt(2);
    source[2] = -1/np.sqrt(2);
    spinstate = "psimin";
    
    # mesh of x0s (= N0s * alat)
    numpts = (29, 49);
    kx0max = 1.0*np.pi;
    N0max = 1+int(kx0max/(k_rho)); # a = 1
    if verbose: print("N0max = ",N0max);
    N0vals = np.linspace(20, 1.5*N0max, numpts[0], dtype = int); # always integer
    kx0vals = k_rho*(N0vals-1); # a = 1
    # modulate optical distance vs N, k'
    Tvals = np.zeros((numpts[0]+1,numpts[1]+1),dtype = complex);
    for N0i in range(len(N0vals)): # iter over N

        N0 = N0vals[N0i];
        Tvals[N0i+1,0] = N0; # record N

        # iter over k'
        ka0 = np.pi/(N0 - 1);
        kpalims = (0.0*ka0,2.1*ka0);
        kpavals = np.linspace(*kpalims, numpts[1]);
        Vgvals = -2*tl*np.cos(ka0) + 2*tl*np.cos(kpavals);
        for Vgi in range(len(Vgvals)):

            Tvals[0,Vgi+1] = kpavals[Vgi];

            # construct blocks of hamiltonian
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
            hblocks, tblocks = wfm.utils.h_cicc_eff(Jeff, tl, 1, N0, N0+2);
            for blocki in range(len(hblocks)): # add Vg in SR
                if(blocki > 0 and blocki < N0 + 1): # if in SR
                    hblocks[blocki] += Vgvals[Vgi]*np.eye(np.shape(hblocks[0])[0])
                    
            # get data
            Energy = -2*tl*np.cos(ka0);
            print("N0, Energy = ",N0, Energy);
            print("rhoJa = ",Jeff/(np.pi*np.sqrt(tl*(Energy+2*tl))));
            Tvals[N0i+1,Vgi+1] = sum(wfm.kernel(hblocks, tblocks, tl, Energy, source));
        # end iter over k'
    # end iter over N

    # package into one array
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    fname = "dat/cicc/"+spinstate+"/";
    fname +="3d_rhoJa"+str(int(rhoJa))+".npy";
    np.save(fname,Tvals);
    if verbose: print("Saved data to "+fname);
    raise(Exception);

# 3d data modulating N, k'
        
if True: # plot fig 2b data

    # plot each file given at command line
    #fig, axes = plt.subplots();
    #axes = [axes];
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):

        # unpack
        print("Loading data from "+datafs[fi]);
        data = np.load(datafs[fi]);
        N0vals = data[1:,0];
        kpvals = data[0,1:];
        Tvals = data[1:,1:]; # 3d data vs N 9rows) and k' (columns)

        if False: # plot at fixed N
            for N0i in range(len(N0vals)):
                axes[0].plot(N0vals[N0i]*kpvals/np.pi, np.real(Tvals[N0i,:]));

        else: # plot 3d
            fig = plt.figure();
            ax3d = fig.add_subplot(projection = "3d");
            N0vals, kpvals = np.meshgrid(N0vals, kpvals);
            ax3d.plot_surface(np.real(N0vals),np.real(kpvals),np.real(Tvals.T),cmap = cm.coolwarm);
            ax3d.set_xlim(np.min(np.real(N0vals)), np.max(np.real(N0vals)));
            ax3d.set_xlabel("$N$");
            ax3d.set_ylim(np.min(np.real(kpvals)), np.max(np.real(kpvals)));
            ax3d.set_ylabel("$k'a$");
            ax3d.set_zlim(0,1);
            ax3d.set_zticks([0,1]);
            ax3d.set_zlabel("$T$");
            plt.show();
            

    # format and show
    axes[0].set_xlim(0.0,2.1);
    axes[0].set_xticks([0,1,2]);
    axes[0].set_xlabel("$ka(N-1)/\pi$", fontsize = "x-large");
    axes[0].set_ylim(0.0,1);
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    plt.show();
    raise(Exception);


