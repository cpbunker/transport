'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a spin-1/2 impurity

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
Jeff = 0.2;
Delta = 0.04; # zeeman splitting on imp

# 2nd qu'd operator for S dot s
h1e = np.zeros((4,4))
g2e = ops.h_kondo_2e(Jeff, 0.5); # J, spin
states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

# leads and zeeman splitting
hzeeman = np.array([[2*Delta/2, 0, 0, 0],
                [0,0.02, 0, 0],
                [0, 0, 2*Delta/2, 0],
                [0, 0, 0,0.02]]); # zeeman splitting
hLL = np.copy(hzeeman);
hRL = np.copy(hzeeman);

# source = up electron, down impurity
source = np.zeros(np.shape(hSR)[0]);
source[1] = 1;

# package together hamiltonian blocks
hblocks = np.array([hLL, hSR, hRL]);

# construct hopping
tblocks = np.array([-tl*np.eye(*np.shape(hSR)),-tl*np.eye(*np.shape(hSR))]);
if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

# sweep over range of energies
# def range
Emin, Emax = -1.99*tl, -1.9*tl;
numE = 20;
Evals = np.linspace(Emin, Emax, numE, dtype = complex);
Tvals = [];
for Ei in range(len(Evals) ):
    Tvals.append(wfm.kernel(hblocks, tblocks, tl, Evals[Ei], source));

# plot Tvals vs E
Tvals = np.array(Tvals);
fig, ax = plt.subplots();
ax.scatter(Evals + 2*tl,Tvals[:,1], marker = 's',label = "$T$");
ax.scatter(Evals + 2*tl,Tvals[:,2], marker = 's',label = "$T_{flip}$");

# menezes prediction in the continuous case
# all the definitions, vectorized funcs of E
kappa = np.lib.scimath.sqrt(Evals);
jprime = Jeff/(4*kappa);
l1, = ax.plot(np.linspace(Emin,Emax,100)+2*tl, Jeff*Jeff/(16*(np.linspace(Emin,Emax,100)+2*tl)), label = "Predicted");

# format and plot
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel("$E+2t_l $");
ax.set_ylabel("$T$");
ax.set_title("Up electron scattering from down impurity");
ax.legend(title = "$J = $"+str(Jeff)+"\n$\Delta$ = "+str(Delta));
plt.show();



if(False): # hubbard that downfolds into J

    # scattering region, Vg and U on second level only
    h_SR = np.array([[0,0,-tl,tl,0,0], # up down, -
                     [0,-Vg,0, 0,0,0], # up, up
                     [-tl,0,-Vg, 0,0,-tl], # up, down
                     [tl,0, 0, -Vg,0, tl], # down, up
                     [0,0,0,0,0,0],    # down, down
                     [0,0,-tl,tl,0,U-2*Vg]]); # -, up down
                     

    # leads also have gate voltage
    h_LL = -Vg*np.eye(*np.shape(h_SR));
    h_LL[0,0] = 0;
    h_RL = -Vg*np.eye(*np.shape(h_SR));
    h_RL[0,0] = 0;

    # shift by gate voltage
    h_LL += Vg*np.eye(*np.shape(h_LL));
    h_SR += Vg*np.eye(*np.shape(h_SR));
    h_RL += Vg*np.eye(*np.shape(h_RL));

    # source = up electron, down impurity
    source = np.zeros(np.shape(h_SR)[0]);
    source[2] = 1;
    flipi = 3;
   








