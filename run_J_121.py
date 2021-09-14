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
verbose = 4;
nleads = (1,1);
nelecs = (3,0); # one electron on each dot and one itinerant
ndots = 2;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = "a00";

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.01;
mu = 0.0;
Vg = -10.0;
U = abs(4*Vg);
B = 10.0*tl;
theta = 0.0;

#time info
dt = 0.01;
tf = 1.0;

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = spinstate, prefix = "dat/temp/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2]
    splots = ['occ','Sz','E']; # which subplots to plot
    title = "1 LL site, 2 dots, 1 RL site, initial state |"+spinstate+" $\\rangle$";
    paramstr = "$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    plot.PlotObservables(datafs, sites = ['LL','LD','RD','RL'], splots = splots, mytitle = title, paramstr = paramstr);

    








