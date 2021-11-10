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

import fcdmft

import numpy as np
import matplotlib.pyplot as plt

#### top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 1;

##### 1: set up the impurity + leads system

# anderson dot
Vg = -0.5;
U = 0.5;
h1e = np.array([[[Vg,0],[0,Vg]]]); # on site energy
g2e = np.zeros((1,2,2,2,2));
g2e[0][0,0,1,1] += U; # coulomb
g2e[0][1,1,0,0] += U;

# embed in semi infinite leads (noninteracting, nearest neighbor only)
tl = 1.0; # lead hopping
Vb = 0.005; # bias
th = 0.4; # coupling between imp, leads
coupling = np.array([[[-th, 0],[0,-th]]]); # ASU

# left lead
H_LL = np.array([[[Vb/2,0],[0,Vb/2]]]);
V_LL = np.array([[[-tl,0],[0,-tl]]]); # spin conserving hopping
LLphys = (H_LL, V_LL, np.copy(coupling), Vb/2); # pack

# right lead
H_RL = np.array([[[-Vb/2,0],[0,-Vb/2]]]);
V_RL = np.array([[[-tl,0],[0,-tl]]]); # spin conserving hopping
RLphys = (H_RL, V_RL, np.copy(coupling), -Vb/2); # pack

# energy spectrum 
Es = np.linspace(-1.09*Vb, 1.1*Vb, 101);

#### 2: compute the many body green's function for imp + leads system

# kernel inputs
iE = (Es[-1] - Es[0])/(2*nbo);
kBT = 0.0;

# run kernel for MBGF
MBGF = fcdmft.kernel(Es, iE, h1e, g2e, LLphys, RLphys, verbose = verbose);

#### 3: use meir wingreen formula
jE = fcdmft.wingreen(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

# also try landauer formula
jE_land = fcdmft.landauer(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

plt.plot(Es, np.real(jE));
plt.plot(Es, np.real(jE_land));
plt.title((np.pi/Vb)*np.trapz(np.real(jE), Es) );
plt.show();


