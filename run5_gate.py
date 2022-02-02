'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 5:
Scattering of single electron off of two localized spin-1/2 impurities
with mirror. N12 = dist btwn imps, N23 = dist btwn imp 2, mirror are
tunable in order to realize CNOT
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

# tight binding params
tl = 1.0; # hopping everywhere
Jeff = 0.1; # exchange strength
Vb = 4.0; # barrier in RL (i.e. perfect mirror)

##################################################################################
#### tune N12 and N23
    
if False: 

    # cicc inputs
    rhoJa = 2.0; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                    # this E is measured from bottom of band !!!
    k_rho = np.arccos((E_rho - 2*tl)/(-2*tl)); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - -2*tl*np.cos(k_rho)) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("\nCiccarello inputs:");
    print("- E, J, E/J = ",E_rho, Jeff, E_rho/Jeff);
    print("- k*a = ",k_rho); # a =1
    print("- rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
    E_rho = E_rho - 2*tl + complex(0,0); # measure from mu

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1;
    spinstate = "aab";

    # fix kx12 at pi
    N12 = int(np.pi/k_rho);
    
    # mesh of x23s (= N23*a)
    kx23max = 1.0*np.pi;
    N23max = int(kx23max/k_rho); # a = 1
    if verbose: print("N12 = ",N12,"\nN23max = ",N23max);
    N23vals = np.linspace(1, N23max, 99, dtype = int); # always integer
    kx23vals = k_rho*(N23vals); # a = 1

    # iter over impurity-mirror distances, get transmission
    Tvals = [];
    for N23 in N23vals:

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, 1, 1+N12, 1+N12+N23+1);
        tnnn = np.zeros_like(tnn)[:-1];
        
        # add barrier to RL
        hblocks[-1] += Vb*np.eye(len(source));

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho , source, reflect = True));

    # package into one array
    Tvals = np.array(Tvals);
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));   
    info = np.zeros_like(kx23vals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, k_rho; # save info we need
    data = [info, kx23vals];
    for sigmai in range(len(source)): # 
        data.append(Tvals[:,sigmai]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "dat/gate/"+spinstate+"/";
    fname +="N_rhoJa"+str(int(np.around(rhoJa)))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);


if True:   # plot each file given at command line
    
    fig, axes = plt.subplots();
    axes = [axes];
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):

        # unpack
        print("Loading data from "+datafs[fi]);
        data = np.load(datafs[fi]);
        tl, Jeff, rhoJa, k_rho = data[0,0], data[0,1], data[0,2], data[0,3];
        kNavals = data[1];
        Tvals = data[2:].T;

        # plot by channel
        print(">>>",np.shape(Tvals));
        labels = ["aaa","aab","aba","abb","baa","bab","bba","bbb"];
        for sigmai in range(np.shape(Tvals)[1]):
            axes[0].plot(kNavals/np.pi, Tvals[:,sigmai], label = labels[sigmai]);

    # format and show
    axes[0].set_xlim(0.0,1.1);
    axes[0].set_ylim(0.0,1);
    axes[0].set_xticks([0,1]);
    axes[0].set_xlabel("$kaN_{23}/\pi$", fontsize = "x-large");
    axes[0].set_ylabel("$T$", fontsize = "x-large");
    plt.legend();
    plt.show();
    raise(Exception);









