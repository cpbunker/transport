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

# blocks and inter block hopping
hSR = np.array([[Vb]]);
hLL = 0*np.eye(np.shape(hSR)[0]);
Vhyb = -tl*np.eye(np.shape(hSR)[0]);
VSR = [];
hRL = Vb*np.eye(np.shape(hSR)[0]);
print(hLL, "\n", hSR, "\n", hRL);

# source
source = np.zeros(np.shape(hSR)[0]);
source[0] = 1;

# sweep over range of energies
# def range
Elims = np.array([-2*tl,-1*tl]);
Evals, Tvals = wfm.Data(source, hLL, Vhyb, [hSR], VSR, hRL, tl, Elims, verbose = verbose);

# plot Tvals vs E
fig, ax = plt.subplots();
ax.scatter(Evals + 2*tl,np.real(Tvals[:,0]), marker = 's');
#ax.scatter(Evals + 2*tl,np.imag(Tvals[:,0]), marker = 's');

# format and show
#ax.set_ylim(0.0,0.25);
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel("$E+2t_l $");
ax.set_ylabel("$T$");
ax.set_title("Scattering off step $V_b$ = "+str(Vb));
plt.show();
   








