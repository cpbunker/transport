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
#np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

##### 1: set up the impurity + leads system
nimp = 1; # number of impurities

# anderson dot
Vg = -0.1;
U = 0.3;
h1e = np.array([[[Vg,0],[0,Vg]]]); # on site energy
g2e = np.zeros((1,2,2,2,2));
g2e[0][0,0,1,1] += U; # coulomb
g2e[0][1,1,0,0] += U;
assert(np.shape(h1e)[1] == 2*nimp); # 2 spin orbs per imp

# embed in semi infinite leads (noninteracting, nearest neighbor only)
tl = 1.0; # lead hopping
Vb = 0.005; # bias
th = 1.0; # coupling between imp, leads
coupling = np.array([-th*np.eye(2*nimp)]);

# left lead
H_LL = np.array([np.zeros((2*nimp,2*nimp))]);
V_LL = np.array([-tl*np.eye(2*nimp)]); # spin conserving hopping
LLphys = (H_LL, V_LL, np.copy(coupling), Vb/2); # pack

# right lead
H_RL = np.array([np.zeros((2*nimp,2*nimp))]);
V_RL = np.array([-tl*np.eye(2*nimp)]); # spin conserving hopping
RLphys = (H_RL, V_RL, np.copy(coupling), -Vb/2); # pack

# energy spectrum 
Es = np.linspace(-1.09*Vb, 1.1*Vb, 101);

#### 2: compute the many body green's function for imp + leads system

# kernel inputs
nbo = 6; # num bath orbs
iE = (Es[-1] - Es[0])/nbo
kBT = 0.0;

# run kernel for MBGF
MBGF = fcdmft.kernel(Es, iE, h1e, g2e, LLphys, RLphys, n_bath_orbs = nbo, solver = 'cc', verbose = verbose);

#### 3: use meir wingreen formula

# matrix of spin current at all energies
jE = fcdmft.wingreen(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

# plot
plt.plot(Es, np.real(jE[0,0]+jE[1,1]));
plt.title((np.pi/Vb)*np.trapz(np.real(jE[0,0]+jE[1,1]), Es) );
plt.xlabel("Energy");
plt.ylabel("Current density");
plt.show();
assert False;

# also try landauer formula
jE_land = fcdmft.landauer(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

# spin 1/2 scatterer
Vg = -0.5;
J = 0.1;
h1e = np.zeros((1,2*nimp,2*nimp));
h1e[0][1,1] = Vg; # localized down spin
g2e = np.zeros((1,2*nimp,2*nimp,2*nimp,2*nimp));
g2e[0][0,0,2,2] = J/4;
g2e[0][2,2,0,0] = J/4;
g2e[0][0,0,3,3] = -J/4;
g2e[0][3,3,0,0] = -J/4;
g2e[0][1,1,2,2] = -J/4;
g2e[0][2,2,1,1] = -J/4;
g2e[0][1,1,3,3] = J/4;
g2e[0][3,3,1,1] = J/4;
g2e[0][0,1,2,3] = J/2;
g2e[0][2,3,0,1] = J/2;
g2e[0][1,0,2,3] = J/2;
g2e[0][2,3,1,0] = J/2;

# plot all spin currents
for sigma in [0,1]:
    for sigmap in [0,1]:
        plt.plot(Es, jE[sigma,sigmap], label = (sigma, sigmap));
plt.legend();
plt.show();
assert(False);
