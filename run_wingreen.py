'''

'''

import fcdmft

import numpy as np
import matplotlib.pyplot as plt

#### top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 1;


##### 1: set up the impurity + leads system

# anderson dot
Vg = -0.0;
U = 0.0;
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

# energy spectrum needs to handle close to 0 finely, but also extend to \pm 2
# also needs to avoid 0 and \pm Vb/2 (screened out)
bigstep, smallstep = 2/100, 1.5*Vb/100
Es = np.append(np.arange(-2.0,-1.5*Vb-smallstep, bigstep), np.arange(-1.5*Vb+smallstep, -Vb/100, smallstep));
Es = np.append(Es, np.arange(Vb/100, 1.5*Vb-smallstep, smallstep));
Es = np.append(Es, np.arange(1.5*Vb+smallstep, 2.0, bigstep)); # lends itself to 4 bath orbs
# screen out
for badnum in [0, Vb/2, -Vb/2]:
    if(np.min(abs(Es - badnum)) < 1e-10): assert False
Es = np.linspace(-1.09*Vb, 1.1*Vb, 101);

#### 2: compute the many body green's function for imp + leads system

# kernel inputs
nbo = 4;
iE = (Es[-1] - Es[0])/(2*nbo);
kBT = 0.0;

# run kernel for MBGF
MBGF = fcdmft.kernel(Es, iE, h1e, g2e, LLphys, RLphys, solver = 'fci', n_bath_orbs = nbo, verbose = verbose);

#### 3: use meir wingreen formula
jE = fcdmft.wingreen(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

# also try landauer formula
jE_land = fcdmft.landauer(Es, iE, kBT, MBGF, LLphys, RLphys, verbose = verbose);

plt.plot(Es, np.real(jE));
plt.plot(Es, np.real(jE_land));
plt.title((np.pi/Vb)*np.trapz(np.real(jE), Es) );
plt.show();


