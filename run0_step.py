'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a step potential

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from transport import wfm, fci_mod, ops
from transport.wfm import wfm_tight

import numpy as np
import matplotlib.pyplot as plt

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# tight binding params
tl = 1.0;
Vb = 0.01; # barrier

# blocks and inter block hopping
hSR = np.array([[Vb]]);
hLL = 0*np.eye(np.shape(hSR)[0]);
Vhyb = -tl*np.eye(np.shape(hSR)[0]);
VSR = [];
hRL = Vb*np.eye(np.shape(hSR)[0]);
hblocks = np.array([hLL, hSR, hRL]);
tblocks = np.array([np.copy(Vhyb), np.copy(Vhyb)])
if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

# source
source = np.zeros(np.shape(hSR)[0]);
source[0] = 1;

# sweep over range of energies
Elims = np.array([-2.5*tl,-1.98*tl]);
Es = np.linspace(Elims[0], Elims[1], 20, dtype = complex);
Esplus = np.real(Es + 2*tl);

# test with wfm discrete
Ts = [];
for E in Es:
    if( E == Es[-1]): # verbose
        Ts.append(wfm_tight.Tcoef(np.array([hLL[0,0],hSR[0,0],hRL[0,0]]), tl, E, verbose =verbose));
    else:
        Ts.append(wfm_tight.Tcoef(np.array([hLL[0,0],hSR[0,0],hRL[0,0]]), tl, E));
plt.scatter(Esplus, np.real(Ts), marker = 's');
plt.plot(Esplus, 4*np.lib.scimath.sqrt(Esplus*(Esplus-Vb))/np.power(np.lib.scimath.sqrt(Esplus) + np.lib.scimath.sqrt(Esplus - Vb),2));
plt.show();

# now do with actual wfm
Tvals = [];
for E in Es:
    if( E == Es[-1]): # verbose
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E, source, verbose = verbose));
    else:
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E, source));
Tvals = np.array(Tvals);

# plot Tvals vs E
fig, ax = plt.subplots();
ax.scatter(Esplus,np.real(Tvals[:,0]), marker = 's');
ax.plot(Esplus, 4*np.lib.scimath.sqrt(Esplus*(Esplus-Vb))/np.power(np.lib.scimath.sqrt(Esplus) + np.lib.scimath.sqrt(Esplus - Vb),2));

# format and show
#ax.set_ylim(0.0,0.25);
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel("$E+2t_l $");
ax.set_ylabel("$T$");
ax.set_title("Scattering off step $V_b$ = "+str(Vb));
plt.show();
   








