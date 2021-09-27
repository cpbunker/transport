'''
Christian Bunker
M^2QM at UF
September 2021

Greens function transport through a square barrier
Look for delta function limit
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

# define source, ie what is incident at left bdy, in this case an up electron
sourcei = 0; # all electrons up ie spinless system

# sweep over range of energies
# def range
Emin, Emax = -2.0,-2+2.0;
N = 30;
Evals = np.linspace(Emin, Emax, N, dtype = complex);

# multiple alphas
# plot
Js = [0.1, 1.0];
fig, axes = plt.subplots(len(Js), sharex = True);
handles, labels = [], [];
for Ji in range(len(Js)):

    # potential
    J = Js[Ji];
    h_delta = J*np.eye(8);
    h = np.array([np.zeros_like(h_delta), h_delta, np.zeros_like(h_delta) ]);

    # hopping
    tl = 1.0;
    tl_arr = np.array([ tl*np.eye(8), tl*np.eye(8) ]);

    # get T(E) data
    Tvals = []
    for Ei in range(len(Evals) ):
        Tvals.append(list(wfm.Tcoef(h, tl_arr, Evals[Ei], sourcei)) );
    Tvals = np.array(Tvals);
    Ttotals = np.sum(Tvals, axis = 1);

    # plot
    s1 = axes[Ji].scatter(Evals+2*tl, Ttotals, color = colors[Ji], marker = 's', label = "$J = $"+str(J));
    m = 0.5; # this works ad hoc
    predict = 1/(1+m*J*J/(2*(Evals+2*tl)));
    l1, = axes[Ji].plot(Evals + 2*tl, predict, color = colors[Ji]);
    handles.append(s1);
    labels.append( "$J = $"+str(J) );

# format
axes[0].set_title("Scattering from $J\delta (x)$ potential");
axes[0].set_ylabel("$T$");
axes[-1].set_xlabel("$E + 2t_l, t_l \equiv 1.0$");
axes[0].legend(handles = handles, labels = labels);
for ax in axes:
    #ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
plt.show();




    








