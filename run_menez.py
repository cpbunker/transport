'''
Christian Bunker
M^2QM at UF
September 2021

Access electron transport regime for a 2 quantum dot model e.g. ciccarello's paper
Initial state:
- itinerant electron up on LL, move to RL in time
- 1 down e confined to each dot

Difference from 2 dot scattering is that there is a conduction site in btwn dots now 
'''

import wfm

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys

##################################################################################
#### make contact with menezes

# top level
plt.style.use('seaborn-dark-palette');
colors = seaborn.color_palette("dark");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5

# siam inputs
tl = 1.0;
Vg = 100;

# plot at different J
fig, axes = plt.subplots();
axes = [axes];
for J in [0.1,0.4]:

    Jmat = J*np.array([ [1/2,-1/2],[-1/2,1/2] ]);

    # menezes just has single delta potential interaction
    h_menez = np.array([np.zeros_like(Jmat), Jmat, np.zeros_like(Jmat) ]);

    # construct hopping
    Tmat = -tl*np.array([ [1,0],[0,1] ]);
    tl_arr = np.array([ np.copy(Tmat), np.copy(Tmat) ]);
    print(h_menez, tl_arr);

    # define source, ie what is incident at left bdy, in this case an up electron
    sourcei = 0; # up itinerant e is 0th basis slot

    if False: # test at max verbosity
        myT = wfm.Tcoef(h_menez, tl_arr, -1.99, sourcei, verbose = 5);
        print("******",myT);

    # sweep over range of energies
    # def range
    Emin, Emax = -2,-2+0.2
    N = 10;
    Evals = np.linspace(Emin, Emax, N, dtype = complex);
    Tupvals = np.zeros_like(Evals);
    Tdownvals = np.zeros_like(Evals);

    # sweep thru E
    for Ei in range(len(Evals) ):
        Tupvals[Ei], Tdownvals[Ei] = wfm.Tcoef(h_menez, tl_arr, Evals[Ei], sourcei);

    # plot
    s2 = axes[0].scatter(Evals+2*tl, Tdownvals, marker = 's', label = "$T_{down},\, J = $"+str(J));

    # menezes prediction in the continuous case
    # all the definitions, vectorized funcs of E
    newEvals = np.linspace(0.0,0.2,100);
    kappa = np.lib.scimath.sqrt(newEvals);
    jprime = J/(4*kappa);
    l1, = axes[0].plot(newEvals, J*J/(16*newEvals));

# format
axes[0].set_title("Up electron scattering from a down spin impurity");
axes[0].set_ylabel("$T$");
axes[0].set_xlabel("$E + 2t_l$");
axes[0].set_ylim(0.0,1.05);
plt.legend();
for ax in axes:
    #ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
plt.show();




    








