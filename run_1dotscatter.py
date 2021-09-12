'''
Christian Bunker
M^2QM at UF
September 2021

Itinerant electron scattering from a single dot
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (2,2);
nelecs = (2,0); # one electron on dot and one itinerant
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = "ab";

# phys params, must be floats
tl = 1.0;
th = 0.2; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = -5.0;
U = abs(4*Vg);
B = 1100;
theta = 0.0;

#time info
dt = 0.01;
tf = 500.0;

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = spinstate, prefix = "dat/cicc/1dotscatter/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2]
    splots = ['occ','Sz','E']; # which subplots to plot
    title = "Heisenberg scattering";
    paramstr = "$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    plot.PlotObservables(datafs, sites = ['L1','LL','D','R1','R2'], splots = splots, mytitle = title, paramstr = paramstr);

    








