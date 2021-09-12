'''
Christian Bunker
M^2QM at UF
September 2021

Access electron transport regime for a 2 quantum dot model e.g. ciccarello's paper
Initial state:
- itinerant electron up on LL, move to RL in time
- 1 down e confined to each dot
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (2,3);
nelecs = (1,0); # one electron on each dot and one itinerant
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = "a";

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = 0.0;
U = abs(4*Vg);
B = 10.0*tl;
theta = 0.0;

#time info
dt = 0.01;
tf = 6.0;

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = spinstate, prefix = "dat/cicc/diffusion/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2]
    splots = ['occ']; # which subplots to plot
    title = "Delocalized electron through a featureless dot";
    paramstr = "$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    plot.PlotObservables(datafs, ['L1','L2','R1','R2','R3','R4'], splots = splots, mytitle = title, paramstr = paramstr);

    








