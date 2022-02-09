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

    # choose boundary condition
    source = np.zeros(8);
    source[1] = 1;
    spinstate = "aab";

    # fix kpa0 at 0
    N12 = 2
    factor = 988;
    ka0 =  np.pi/(N12 - 1)/factor; # free space wavevector, should be << pi
                                    # increasing just broadens the Vg peak
    kpa0 = np.pi/(N12 - 1)/factor; # wavevector in gated SR
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = Jeff/(np.pi*np.sqrt(tl*E_rho));
    myVg = -2*tl*np.cos(ka0) + 2*tl; # to get kpa0 = 0
    
    # mesh of x23s (= N23*a)
    kx23max = 1.0*np.pi;
    N23max = int(kx23max/ka0); # a = 1
    if verbose: print("rhoJa = ",rhoJa,"\nN12 = ",N12,"\nN23max = ",N23max);
    N23vals = np.linspace(1, N23max, 99, dtype = int); # always integer
    kx23vals = ka0*(N23vals); # a = 1

    # iter over impurity-mirror distances, get transmission
    Tvals = [];
    for N23 in N23vals:

        # construct hams
        print(N23);
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks also
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, 1, 1+N12, 1+N12+N23+1);
        tnnn = np.zeros_like(tnn)[:-1];

        # add gate voltage in SR
        for blocki in range(len(hblocks)):
            if(blocki > 0 and blocki < N12 + 1): 
                hblocks[blocki] += myVg*np.eye(np.shape(hblocks[0])[0])
        
        # add barrier to RL
        hblocks[-1] += Vb*np.eye(len(source));

        # get T from this setup
        Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(ka0), source, reflect = True));

    # package into one array
    Tvals = np.array(Tvals);
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));   
    info = np.zeros_like(kx23vals);
    info[0], info[1], info[2], info[3] = tl, Jeff, rhoJa, ka0; # save info we need
    data = [info, kx23vals];
    for sigmai in range(len(source)): # 
        data.append(Tvals[:,sigmai]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "dat/gate/"+spinstate+"/";
    fname = "";
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
        for sigmai in [1,2,4]:
            axes[0].plot(kNavals/np.pi, Tvals[:,sigmai], label = labels[sigmai]);

    # format and show
    axes[0].set_xlim(0.0,1.0);
    axes[0].set_xticks([0,1]);
    axes[0].set_xlabel("$ka(N_{B} - 2)/\pi$", fontsize = "x-large");
    axes[0].set_ylim(0.0,1);
    axes[0].set_yticks([0,1]);
    axes[0].set_ylabel("$R$", fontsize = "x-large");
    plt.show();
    raise(Exception);









