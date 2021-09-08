'''
Christian Bunker
M^2QM at UF
September 2021

Access electron transport regime for a 2 quantum dot model e.g. ciccarello's paper
Initial state:
- itinerant electron up on LL, move to RL in time
- 1 down e confined to each dot

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
nleads = (1,1);
nelecs = (3,0); # one electron on each dot and one itinerant
ndots = 2;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.01;
mu = 0.0;
Vgs = [-5.0];
# U fixed to 6*abs(Vg)
B = 10.0*tl;
theta = 0.0;

#time info
dt = 0.01;
tf = 40.0;

if get_data: # must actually compute data

    for i in range(len(Vgs)): # iter over Vg vals;
        Vg = Vgs[i];
        params = tl, th, td, Vb, mu, Vg, abs(6*Vg), B, theta;
        siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = "abb", prefix = "dat/cicc/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2:]
    labs = Vgs # one label for each Vg
    splots = ['Jup','Jdown','occ','delta_occ','Sz','Szleads','E']; # which subplots to plot
    title = "Current between impurity and left, right leads"
    plot.CompObservables(datafs, labs, splots = splots, mytitle = title, leg_title = "$V_g$", leg_ncol = 2, whichi = 0);

    








