'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for simulating current through single dot SIAM, weak biasing, using td-fci
compare w/ different U
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 3;
nleads = (2,2);
nelecs = (sum(nleads)+1,0); # half filling
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = tl/10; # can scale down and same effects are seen. Make sure to do later
td = 0.0;
Vb = -1/100*tl;
mu = 10.0
Vg = mu
Us = [0.0,1.0, 10.0];
B = 1.0;
theta = 0.0;

#time info
dt = 0.004;
tf = 5.0;

if get_data: # must actually compute data

    for i in range(len(Us)): # iter over Vg vals;
        U = Us[i]
        params = tl, th, td, Vb, mu, Vg, U, B, theta;
        siam_current.DotData(nleads, nelecs, ndots, tf, dt, prefix = "dat/param_tuning/tuneU/", phys_params=params, namevar="U", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2:];
    labs = Us;
    splots = ['J','Jup','Jdown','Sz']; # which subplots to plot
    title = "Coulomb blockade for a down spin impurity, $V_g = \mu = 10.0$"
    plot.CompObservables(datafs, labs, leg_title = "U", mytitle = title, whichi = 0, splots = splots);

    








