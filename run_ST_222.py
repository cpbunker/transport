'''
Christian Bunker
M^2QM at UF
September 2021

Access electron transport regime for a 2 quantum dot model e.g. ciccarello's paper
Want single itinerant electron to start at LL in up state, move to RL in time
Want 1 e confined to each dot in down state

Prelim goal:
- prep in up down down state
- let itinerant e move from left to right
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (2,2);
nelecs = (3,0); # one electron on each dot and one itinerant
ndots = 2;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 2.0; # need strong coupling for transport
td = 3.0; # "
Vb = 10.0;
mu = 1.0;
Vgs = [-10.0];
U =  20.0;
B = 1.0*tl;
theta = 0.0;

#time info
dt = 0.01;
tf = 1.0;

if get_data: # must actually compute data

    for i in range(len(Vgs)): # iter over Vg vals;
        Vg = Vgs[i];
        params = tl, th, td, Vb, mu, Vg, U, B, theta;
        siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, prefix = "dat/cicc/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2:]
    labs = Vgs # one label for each Vg
    splots = ['occ','delta_occ','Sz','Szleads','E']; # which subplots to plot
    title = "Current between impurity and left, right leads"
    plot.CompObservables(datafs, labs, splots = splots, mytitle = title, leg_title = "$V_g$", leg_ncol = 1, whichi = 0);

    








