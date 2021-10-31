'''
Demonstration of my wrappers to Tianyu Zhu's fcdmft package

Dynamical Mean Field Theory:
Treat impurity region, which is highly correlated, at a very high level of
quantum chemistry. Treat environment at lower level. If the environment is
periodic and highly correlated, one can then do a self consistent loop. See
e. g. https://arxiv.org/pdf/1012.3609.pdf.

fcdmt package contains:
- drivers for dmft algorithm, including self consistent loop (dmft/gwdmft.py)
- utils for these drivers (dmft/gwdmft.py)
- solvers for many body physics (e. g. RHF, CC, FCI)

My fcdmft.kernel() function uses utils and solvers to put together a dmft driver
that skips the self consistent loop, i. e. for a non periodic system
'''

from transport import wingreen, ops, fci_mod
import fcdmft

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
iE = 1e-3; # small imag part

#### do SIAM in ASU formalism

# energy spectrum
Es = np.linspace(-2.0,0.0,101);
iE = 1e-2;

# anderson dot
Vg = -0.5;
U = 1.1;
h1e = np.array([[[Vg,0],[0,Vg]]]); # on site energy
g2e = np.zeros((1,2,2,2,2));
g2e[0][0,0,1,1] += U; # coulomb
g2e[0][1,1,0,0] += U;
th = 1.0; # coupling between imp, leads
coupling = np.array([[[-th, 0],[0,-th]]]); # ASU

# embed in semi infinite leads
# leads are noninteracting, nearest neighbor only
tl = 1.0; # lead hopping
HL = np.array([[[0,0],[0,0]]]);
VL = np.array([[[-tl,0],[0,-tl]]]); # spin conserving hopping

# pass to kernel
# kernel couples the scattering region, repped by h1e and g2e,
# to the semi infinite leads, repped by HL, VL
# treats the SR with fci green's function
MBGF = fcdmft.kernel(Es, iE, h1e, g2e, coupling, HL, VL, solver = 'fci', verbose = verbose);

# see results
print("\n>> Results of MBGF calculation:");
print("\n - shape: ",np.shape(MBGF));
print("\n - operator:\n", MBGF[0,:3,:3,0]);
plt.plot(Es, (-1/np.pi)*np.imag(MBGF[0,0,0,:]), label = "up" );
plt.plot(Es, (-1/np.pi)*np.imag(MBGF[0,1,1,:]), label = "down" );
plt.xlabel("$E$");
plt.ylabel("Density of states");
plt.legend();
plt.show();




