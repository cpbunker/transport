'''
'''

from transport import wingreen, ops, fci_mod
import fcdmft

import numpy as np

# top level
verbose = 5;

# energy range
Emin, Emax = 0.01, 0.5;
Evals = np.linspace(Emin, Emax, 2, dtype = complex);
iE = 1e-3; # small imag part

# 2 site hubbard, ASU formalism
td = 1.0; # hopping between sites
Vg = -0.05;
U = 0.4;
hmat = ops.h_hub_1e(Vg, td); # 1e matrix
gmat = ops.h_hub_2e(U, U); # 2e matrix
th = 0.4; # coupling between sites
thmat = np.array([[-th,0,0,0],[0,-th,0,0],[-th,0,0,0],[0,-th,0,0]]); # coupling matrix

# embed in semi infinite leads
tl = 1.0; # lead hopping, rescales input energies
chain_length = 20; # num sites in tb chain
if(verbose): print("\n1. Hybridization");
lead_dos = wingreen.dos.surface_dos(chain_length, None, chain_length+1, Evals/tl, iE/tl, verbose = verbose);

# map semi infinite leads onto discrete bath
bath = fcdmft.hybridization(Evals, lead_dos, 10);









assert False;
# Gimp with DMRG
bdims = np.array([300,400,500]);
noises = np.array([1e-4,1e-5,1e-6]);
G_imp = fcdmft.h1e_to_gf(hmat, gmat, (2,0), bdims, noises);

# green's function at scattering region
if(verbose): print("\n2. Impurity green's function:");
GFSR = wingreen.dos.junction_gf(np.copy(gf_noninteract), np.copy(thmat), np.copy(gf_noninteract), np.copy(thmat), Evals, hmat);
for el in GFSR: print(el);
