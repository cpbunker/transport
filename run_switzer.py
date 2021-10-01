'''
Christian Bunker
M^2QM at UF
October 2021

Transmit an itinerant electron through eric's model
'''

import wfm

import numpy as np
import matplotlib.pyplot as plt
import sys

##################################################################################
#### make contact with menezes

# top level
#plt.style.use('seaborn-dark-palette');
#colors = seaborn.color_palette("dark");
colors = ['tab:blue','tab:red','tab:green','tab:blue'];
np.set_printoptions(precision = 4, suppress = True);
verbose = 5

# define params according to Eric's paper
JH = 1.0;
JK2 = 1.0;
JK3 = 1.0;
D = 1.0;

# eff params
JK = (JK2 + JK3)/2;
DeltaK = JK2 - JK3;




# format
axes[0].set_title("");
axes[0].set_ylabel("$T$");
axes[0].set_xlabel("$E + 2t_l$");
axes[0].set_ylim(0.0,1.05);
plt.legend();
for ax in axes:
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
plt.show();

    








