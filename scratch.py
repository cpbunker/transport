'''
'''

from transport import wingreen, ops, fci_mod
import fcdmft

import numpy as np

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
iE = 1e-3; # small imag part

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


hyb = dyson(nonint of whole system, nonint of central region)
g_nonint_ofwhole system = inv( ({w-h, t,},{t, gsurf}}
                               g_cc = inv(w - h - t g_surf ^-1 t  )
                               second term = \Delta


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
