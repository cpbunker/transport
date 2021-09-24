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

# convert to eff heisenberg interaction
J = -2*tl*tl/Vg;
Jmat = J*np.array([ [1,-1],[-1,1] ]);

# menezes just has single delta potential interaction
h_menez = np.array([np.zeros_like(Jmat), Jmat, np.zeros_like(Jmat) ]);

# construct hopping
Tmat = -tl*np.array([ [1,0],[0,1] ]);
tl_arr = np.array([ np.copy(Tmat), np.copy(Tmat) ]);
print(h_menez, tl_arr);

# define source, ie what is incident at left bdy, in this case an up electron
sourcei = 0; # up itinerant e is 0th basis slot

if True: # test at max verbosity
    myT = wfm.Tcoef(h_menez, tl_arr, -1.99, sourcei, verbose = 5);
    print("******",myT);

# sweep over range of energies
# def range
Emin, Emax = -2,-2+0.02
N = 20;
Evals = np.linspace(Emin, Emax, N, dtype = complex);
Tupvals = np.zeros_like(Evals);
Tdownvals = np.zeros_like(Evals);

# sweep thru E
for Ei in range(len(Evals) ):
    Tupvals[Ei], Tdownvals[Ei] = wfm.Tcoef(h_menez, tl_arr, Evals[Ei], sourcei);

# plot
fig, axes = plt.subplots(2, sharex = True);
s1 = axes[0].scatter(Evals+2*tl, Tupvals, color = colors[0], marker = 's', label = "$T_{up}$");
s2 = axes[1].scatter(Evals+2*tl, Tdownvals, color = colors[1], marker = 's', label = "$T_{down}$");

# menezes prediction in the continuous case
# all the definitions, vectorized funcs of E
newEvals = np.linspace(0.0,0.04,100);
kappa = np.lib.scimath.sqrt(newEvals);
jprime = J/(4*kappa);
l1, = axes[1].plot(newEvals, J*J/(4*newEvals), color = colors[1], label = "Continuous");

# format
axes[0].set_title("Incident up electron scattering from a spin impurity");
axes[0].set_ylabel("$T$");
axes[1].set_xlabel("$E + 2t_l$");
plt.legend([s1,s2,l1],['$T_{up}$','$T_{down}$','continuous'],title = "$J = $"+str(J)+"\n$t_l = $"+str(tl));
for ax in axes:
    #ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
plt.show();




    








