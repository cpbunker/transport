'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for prepping dot spin state with B field, getting current output
'''


import siam_current

import numpy as np

import sys

##################################################################################
#### prepare dot in diff spin states

# top level params from command line
get_data = bool(float(sys.argv[1]));

if get_data: # must actually compute data

    # rest of top levels needed for getting data
    verbose = int(sys.argv[2]);
    nleads = (int(sys.argv[3]),int(sys.argv[4]));
    nelecs = (sum(nleads)+1,0); # half filling

    # time info
    dt = 0.004
    tf = float(sys.argv[5]);

    # phys params, must be floats
    tl = 1.0;
    th = tl/10; # can scale down and same effects are seen. Make sure to do later
    Vb = -1/100*tl
    mu = 10.0*tl
    Vg = mu;
    U = 100.0*tl;
    thetas = np.array([float(sys.argv[6])] ); # take theta vals from command line one at time for parallelization
    Bs = [tl*5];

    for i in range(len(Bs)): # iter over B, theta inputs
        B, theta = Bs[i], thetas[i];
        params = tl, th, Vb, mu, Vg, U, B, theta, phi;
        fname = siam_current.DotData(nleads, nelecs, tf, dt, prefix = "", phys_params=params, verbose = verbose);

else:
    import plot # do here for compatibility

    # only command line arg needed is data file names
    datafs = sys.argv[2:];
    splots = ['Jup','Jdown','delta_occ']; # which subplots 
    thetas = np.pi*np.linspace(0,8,9,dtype=int)/8
    title = "";
    plot.CompObservables(datafs, thetas, splots = splots, mytitle = title, leg_title = "$\\theta$");


    








