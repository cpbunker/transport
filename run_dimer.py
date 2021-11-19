'''
Christian Bunker
M^2QM at UF
October 2021

Steady state transport of a single electron through a one dimensional wire
Part of the wire is scattering region, where the electron spin degrees of
freedom can interact with impurity spin degrees of freedom

Impurity hamiltonians calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm

import numpy as np
import matplotlib.pyplot as plt

import sys

#### run code

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
option = sys.argv[1];

# tight binding params
tl = 1.0; # 2e hopping, in meV
JK = -0.0;

# construct h_SR and define source
h1e, g2e = wfm.utils.second_q_ham(JK, JK); # second qu'd form
parts = np.array([1,1,1]); # one e, 2 separate impurities
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz

# prep system
mT = 1/2; # total z spin
mystates, myis, mystrs = wfm.utils.subspace(mT); # subspace
h_SR = fci_mod.single_to_det(h1e, g2e, parts, states, dets_interest = mystates);
source = np.zeros(np.shape(h_SR)[0]);
source[1] = 1/np.sqrt(2);
source[2] = 1/np.sqrt(2);
print(h_SR);
mysource = r"$0.71|up, \frac{1}{2}, -\frac{1}{2} \rangle + 0.71|up, -\frac{1}{2}, \frac{1}{2} \rangle$"

#### what to iter over

if option == "E": # iter over energy

    # sweep from bottom of band up
    Evals = np.linspace(-2*tl, -2*tl + 0.6*tl, 20);

    # package h, t block matrices
    hblocks = np.array([np.zeros_like(h_SR), h_SR, np.zeros_like(h_SR)]);
    tblocks = np.array([-tl*np.eye(*np.shape(h_SR)),-tl*np.eye(*np.shape(h_SR))]);

    # get coefs at each energy
    Tvals = [];
    for Ei in range(len(Evals) ):
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Evals[Ei], source));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    xlab = "$E+2t_l $"
    for pi in range(len(myis)):
        ax.scatter(Evals + 2*tl,Tvals[:,myis[pi]], marker = 's',label = mystrs[pi]);
                                                    
    
elif option == "N": # stretch SR, switzer style

    # fix energy near bottom of band
    Energy = -2*tl + 0.5;
    ka = np.arccos(Energy/(-2*tl));

    # iter over N
    Nmax = 10;
    Nvals = np.linspace(0,Nmax,30,dtype = int);
    Tvals = [];
    for N in Nvals:

        # package as block hams 
        # number of blocks depends on N
        hblocks = [np.zeros_like(h_SR)]
        tblocks = [-tl*np.eye(*np.shape(h_SR)) ];
        for Ni in range(N):
            hblocks.append(np.copy(h_SR));
            tblocks.append(-tl*np.eye(*np.shape(h_SR)) );
        hblocks.append(np.zeros_like(h_SR) );
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    xlab = "$N$"
    for pi in range(len(myis)):
        ax.plot(Nvals,Tvals[:,pi], label = mystrs[pi]);


# format and plot
#ax.set_ylim(0.0,0.25);
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel(xlab);
ax.set_ylabel("$T$");
ax.set_title("source = "+mysource+", $J_{K} = $"+str(JK));
plt.legend();
plt.show();




