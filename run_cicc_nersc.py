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
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5


##################################################################################
#### data and plots for fixed E, k, but varying x0
#### effective ham, choose source for initial condition
#### ie replicate Cicc figs 1, 5
    
for rJ in [1.0,2.0]:

    # siam inputs
    tl = 1.0;
    Vg = 20;
    Jeff = 2*tl*tl/Vg; # eff heisenberg

    # cicc inputs
    alat = 1.0; # should always cancel for E and kx0
    m = 1/(2*alat*alat*tl);
    rhoJ_int = rJ; # integer that cicc param rho*J is set equal to
    E_rho = Jeff*Jeff/(rhoJ_int*rhoJ_int*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                            # this E is measured from bottom of band !!!
    k_rho = wfm.k_disp(E_rho-2*tl, alat, tl); # input E measured from 0 by -2*tl
    assert(abs((E_rho - 2*tl) - wfm.E_disp(k_rho,alat, tl) ) <= 1e-8 ); # check by getting back energy measured from bottom of band
    print("E, E - 2t, J, E/J = ",E_rho, E_rho -2*tl, Jeff, E_rho/Jeff);
    print("k*a = ",k_rho*alat);
    print("rho*J = ", (Jeff/np.pi)/np.sqrt(E_rho*tl));
    E_rho = E_rho - 2*tl; # measure from mu

    # choose boundary condition
    source = np.array([0,1,1,0,0,0,0,0])/np.sqrt(2);
    #source = np.array([0,1,0,0,0,0,0,0]);
    #source = source/np.dot(source, source); # normalize
    
    
    # mesh of x0s (= N0s * alat)
    kx0min, kx0max = 3*np.pi/4, 1.1*np.pi;
    kx0min, kx0max = 0.0, 2.1*np.pi
    N0min, N0max = int(kx0min/(k_rho*alat)), int(kx0max/(k_rho*alat));
    if verbose: print("N0 min, max = ",N0min, N0max);
    N0vals = np.linspace(N0min, N0max, int(N0max - N0min), dtype = int); # always integer
    N0vals = N0vals[ N0vals % 3 == 0]; # boolean mask for even only
    print(N0vals);
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
        Tvals.append(list(wfm.Tcoef(hmats, tmats, E_rho , source, verbose = 1)) );

    # package into one array
    Tvals = np.array(Tvals);
    info = np.zeros_like(kx0vals);
    info[0], info[1], info[2], info[3], info[4] = tl, Jeff, rhoJ_int, E_rho, k_rho; # save info we need
    data = [info, N0vals, kx0vals];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of N0val, k0val, corresponding T vals
    # save data
    fname = "";
    fname +="gf_E"+str(E_rho)[:6]+"_k"+str(k_rho)[:4]+"_"+str(rhoJ_int)[:4]+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);

    
