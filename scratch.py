'''
'''

from transport import wingreen, ops, fci_mod
import fcdmft

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
iE = 1e-3; # small imag part

#### test dot_spinful arrays
if(False):
    
    gE = np.array(range(3), dtype = complex);
    gf = np.array([[ [np.copy(gE), np.zeros_like(gE)],[np.zeros_like(gE),np.copy(gE)] ]]); # classic spin by norb by norb by nw array
    Sy = complex(0,1/2)*np.array([[[0,-1],[1,0]]]);
    Sz = (1/2)*np.array([[[1,0],[0,-1]]]);
    print("gf",np.shape(gf)," before:");
    for wi in range(len(gE)): print(gf[0,:,:,wi]);

    # try dot
    gf = fcdmft.dot_spinful_arrays(gf, Sy);
    print("gf",np.shape(gf)," after:");
    for wi in range(len(gE)): print(gf[0,:,:,wi]);

    assert(False); # stop

#### test new surface gf
if(False):

    Es = np.linspace(-2,2,30);
    iE = 1e-3;
    tl = 1.0
    mu = 0.0
    H = np.array([[[mu,0],[0,mu]]]); # set mu-2t = 0
    V = np.array([[[-tl,0],[0,-tl]]]); # spin conserving hopping
    gf = fcdmft.surface_gf(Es, iE, H, V, verbose = verbose);

    # plot results
    for i in range(np.shape(H)[1]):
        for j in range(np.shape(H)[2]):
            plt.plot(Es, (-1/np.pi)*np.imag(gf[0,i,j,:]), label = (i,j));
            print(">>>",(i,j), np.trapz((-1/np.pi)*np.imag(gf[0,i,j,:]),Es) );
    plt.legend();
    plt.show();

    #### understand how the mean field green's function works
    if(False):
        
        # higher level green's function in the scattering region
        SR_meanfield = dmft.dmft_solver.mf_kernel(H, dummy, chem_pot, nao, np.array([np.eye(n_imp_orbs)]), max_mem, verbose = 1);
        g_inta = dmft.dmft_solver.fci_gf(SR_meanfield, Es, iE, verbose = verbose);
        print(np.shape(g_nona), np.shape(g_inta));
        print(g_inta[:,:,0]);
        plt.plot(Es, g_inta[0,0,:]);
        plt.show();

    assert(False); # stop

################################################################
#### main code



