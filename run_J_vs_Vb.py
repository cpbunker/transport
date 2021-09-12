'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for simulating current through single dot SIAM, weak biasing, using td-fci
Model runner for anything in single dot weak biasing regime
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### 

# top level
verbose = 3;
nleads = (3,3);
nelecs = (1,0); 
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = "1a"

# phys params, must be floats
tl = 1.0;
th = 1.0;
td = 0.0; # only one dot
Vbs = [0.01, 0.05, 0.1, 0.5, 1.0]
Vbs = [0.5];
mu = 0.0;
Vg = 0.0
U =  abs(4*Vg)
B = 100.0;
theta = 0.0;

#time info
dt = 0.01
tf = 10.0;

if get_data: # must actually compute data

    for i in range(len(Vbs)): # iter over Vg vals;
        Vb = Vbs[i];
        params = tl, th, td, Vb, mu, Vg, U, B, theta;
        siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, prefix = "dat/param_tuning/Vb/", spinstate = spinstate, namevar = "Vb", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2:]
    labs = Vbs # one label for each Vg
    splots = ['J','occ','Sz']; # which subplots to plot
    title = "Itinerant electron with diffuseness 2"
    for wi in range(1):
        plot.CompObservables(datafs, labs, splots = splots, whichi = wi, mytitle = title, leg_title = "$V_b$", leg_ncol = 1);

    








