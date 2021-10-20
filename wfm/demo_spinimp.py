'''
Christian Bunker
M^2QM at UF
September 2021

Steady state transport of a single electron through a one dimensional wire
A single spin impurity is embedded in the wire, forming the scattering potential

Demonstrates different ways of treating this spin impurity using wave
function matching (wfm.py module)

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

import wfm
import ops
import fci_mod

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

# top level
colors = seaborn.color_palette("dark");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
option = sys.argv[1];

# tight binding params
tl = 1.0;
Vg = 20.0;
U = 100.0;
Jeff = 2*tl*tl*U/(-Vg*(U-Vg));

#### different ways of doing the scattering region

if option == "J": # just do J straightforwardly

    # scattering region
    h_SR = (Jeff/4)*np.array([[-1,2], # up down
                          [2,-1]]); # down up

    # leads
    h_LL = np.zeros_like(h_SR);
    h_RL = np.zeros_like(h_SR);

    # source = up electron, down impurity
    source = np.zeros(np.shape(h_SR)[0]);
    source[0] = 1;
    flipi = 1;

elif option == "Jdown": # hubbard that downfolds into J

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

elif option == "Kondo": # kondo coupling in 2nd qu'd form

    # 2nd qu'd operator for S dot s
    g2e = ops.h_kondo_2e(Jeff, 0.5); # J, spin
    h1e = np.zeros((np.shape(g2e)[0], np.shape(g2e)[1]));

    # convert to determinantal form
    states_1p = [[0,1],[2,3]]; # [site one e up, down], [site 2 e up, down]
    h_SR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p, verbose = verbose);

    # leads
    h_LL = np.zeros_like(h_SR);
    h_RL = np.zeros_like(h_SR);
    
    # source = up electron, down impurity
    source = np.zeros(np.shape(h_SR)[0]);
    source[1] = 1;
    flipi = 2;

else: raise Exception("Option not supported");

# package together hamiltonian blocks
hblocks = np.array([h_LL, h_SR, h_RL]);

# construct hopping
tblocks = np.array([-tl*np.eye(*np.shape(h_SR)),-tl*np.eye(*np.shape(h_SR))]);
if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

if False: # test at max verbosity
    myT = wfm.Tcoef(hblocks, tblocks, tl, -1.99, source, verbose = 5);
    if verbose: print("******",myT);

# sweep over range of energies
# def range
Emin, Emax = -1.99, -2.0 + 0.1
N = 10;
Evals = np.linspace(Emin, Emax, N, dtype = complex);
Tvals = [];
for Ei in range(len(Evals) ):
    Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Evals[Ei], source));

# plot Tvals vs E
Tvals = np.array(Tvals);
fig, ax = plt.subplots();
ax.scatter(Evals + 2*tl,Tvals[:,flipi], marker = 's',label = "$T_{flip}$");

# menezes prediction in the continuous case
# all the definitions, vectorized funcs of E
kappa = np.lib.scimath.sqrt(Evals);
jprime = Jeff/(4*kappa);
l1, = ax.plot(np.linspace(Emin,Emax,100)+2*tl, Jeff*Jeff/(16*(np.linspace(Emin,Emax,100)+2*tl)), label = "Predicted");

# format and plot
ax.set_ylim(0.0,0.25);
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel("$E+2t_l $");
ax.set_ylabel("$T$");
ax.set_title("Up electron scattering from down impurity");
ax.legend(title = "$t_{l} = $"+str(tl)+"\n$V_g = $"+str(Vg)+"\n$U = $"+str(U)+"\n$J_{eff} = $"+str(Jeff));
plt.show();
   








