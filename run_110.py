'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for benchmarking td-FCI
'''

import td_fci
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt
    
import sys

#### 1_1_0 system to match analytical results

verbose = 3;
nleads = (1,0);
nelecs = (1,1);
splots = ['occ','Sz']; # which subplots to make

#time info
dt = 0.04;
tf = 3.14

if ( bool(int(sys.argv[1]) ) ):

    # benchmark with spin free code
    th = 1.0;
    Vb = -0.1;
    U = 0.0;
    params = 0.0, th, Vb, 0.0, U; # featureless dot # no mu or mag field in spin free

    # since spinfree, have to run up and down calculations separately
    # get observables for up case
    obs_up = td_fci.SpinfreeTest(nleads, (1,0), tf, dt, phys_params = params, verbose = verbose);

    # get observables for down case
    obs_down = td_fci.SpinfreeTest(nleads, (0,1), tf, dt, phys_params = params, verbose = verbose);

    # combine into actual observables of a 1 up 1 down system
    observables = [obs_up[0], obs_up[1]+obs_down[1], obs_up[2], obs_down[2], obs_up[4]+obs_down[4], obs_up[5]+obs_down[5],obs_up[6]+obs_down[6],obs_up[7]+obs_down[7],obs_up[8]+obs_down[8],obs_up[9]+obs_down[9] ];
    fname = "dat/bench/spinfree/110_updown.npy"
    np.save(fname, observables);
    print("Saving data to "+fname);

else:
    fname = "dat/bench/spinfree/110_updown.npy";
    plot.PlotObservables(fname, nleads = nleads, splots = splots);



