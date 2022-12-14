'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) solved in time-dependent QM
using bardeen theory method in transport/bardeen
'''

from transport import bardeen

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
n_loc_dof = 2; # spin up and down
tL = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tL;
tR = 1.0*tL;
ts = (tinfty, tL, tinfty, tR, tinfty);
Vinfty = 1.0*tL;
VL = 0.0*tL;
VR = 0.0*tL;
Vs = (Vinfty, VL, Vinfty, VR, Vinfty);
Ninfty = 1;
NL = 1;
NR = 1*NL;
Ns = (Ninfty, NL, NR);

# central region
tC = 1*tL;
VC = 0.1*tL;
#VC = 1.0*tL;
NC = 1;
HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
for j in range(NC):
    HC[j,j] = VC;
for j in range(NC-1):
    HC[j,j+1] = -tC;
    HC[j+1,j] = -tC;

# central region prime
tCprime = tC;
VCprime = VC;
HCprime = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
for j in range(NC):
    HCprime[j,j] = VCprime;
for j in range(NC-1):
    HCprime[j,j+1] = -tCprime;
    HCprime[j+1,j] = -tCprime;
print(HCprime)

bardeen.kernel(*ts, *Vs, *Ns, HC, HCprime);





