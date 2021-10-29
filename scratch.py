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
if(True):

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

# SIAM, ASU formalism
Vg = -0.05;
U = 1.0;
h1e = np.array([[Vg,0],[0,Vg]]); # on site energy
g2e = np.zeros((2,2,2,2));
g2e[0,0,1,1] += U; # coulomb
g2e[1,1,0,0] += U;
th = 0.4; # coupling between imp, leads
Vmat = np.array([[-th, 0],[-th,0]]); # ASU

# embed in semi infinite leads
# leads are noninteracting, nearest neighbor only
tl = 1.0; # lead hopping, rescales input energies
leadsite = fcdmft.site(np.array([0]), np.array([1.0]), iE, (0,1e6), "defH");
# this object contains all the physics of the leads

# pass to kernel
# kernel couples the scattering region, repped by h1e and g2e,
# to the semi infinite leads, repped by leadsite
# treats the SR with fci green's function
MBGF = fcdmft.kernel(h1e, g2e, Vmat, leadsite, verbose = verbose);

print("\n>> Results of MBGF calculation:");
print("\n - shape: ",np.shape(MBGF));
print("\n - operator: ", MBGF[:,:,0]);



'''
# Gimp with DMRG
bdims = np.array([300,400,500]);
noises = np.array([1e-4,1e-5,1e-6]);
G_imp = fcdmft.h1e_to_gf(hmat, gmat, (2,0), bdims, noises);

# green's function at scattering region
if(verbose): print("\n2. Impurity green's function:");
GFSR = wingreen.dos.junction_gf(np.copy(gf_noninteract), np.copy(thmat), np.copy(gf_noninteract), np.copy(thmat), Evals, hmat);
for el in GFSR: print(el);
'''
