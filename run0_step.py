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

import numpy as np
import matplotlib.pyplot as plt

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# tight binding params
tl = 1.0;
Vb = 0.2; # barrier

# scattering region physics
hSR = np.array([[Vb]]);

# lead physics
hLL = 0*np.eye(np.shape(hSR)[0]);
hRL = Vb*np.eye(np.shape(hSR)[0]);

# source
source = np.zeros(np.shape(hSR)[0]);
source[0] = 1;

# package together hamiltonian blocks
hblocks = np.array([hLL, hSR, hRL]);

# construct hopping
tblocks = np.array([-tl*np.eye(np.shape(hSR)[0]),-tl*np.eye(np.shape(hSR)[0])]);
if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

# sweep over range of energies
# def range
Emin, Emax = -2*tl, -1.5*tl;
numE = 10;
Evals = np.linspace(Emin, Emax, numE, dtype = complex);
Tvals = [];
for Ei in range(len(Evals) ):
    Tvals.append(wfm.kernel(hblocks, tblocks, tl, Evals[Ei], source));
Tvals = np.array(Tvals);
print(">>>",np.shape(Tvals));

# plot Tvals vs E
fig, ax = plt.subplots();
ax.scatter(Evals + 2*tl,Tvals[:,0], marker = 's',label = "$T$");

# format and show
#ax.set_ylim(0.0,0.25);
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel("$E+2t_l $");
ax.set_ylabel("$T$");
ax.set_title("Scattering off step $V_b$ = "+str(Vb));
plt.show();
   








