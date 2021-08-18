'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for prepping dot spin state with B field, getting current output
'''

import siam_current

import numpy as np

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 3;
nleads = (2,2);
nelecs = (sum(nleads)+1,0); # half filling
get_data = False; # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = tl/10; # can scale down and same effects are seen. Make sure to do later
Vb = -1/100*tl;
mu = 10.0
Vg = mu
Us = [0.0, 5.0,10.0, 15.0, 20.0]
B = 5*tl;
theta = 0.0;
phi = 0.0;

#time info
dt = 0.004;
tf = 5.0;

if get_data: # must actually compute data

    for i in range(len(Us)): # iter over Vg vals;
        U = Us[i]
        params = tl, th, Vb, mu, Vg, U, B, theta, phi;
        siam_current.DotData(nleads, nelecs, tf, dt, prefix = "dat/tuneU/", phys_params=params, verbose = verbose);

else:

    import plot

    # plot results
    datafs = [];
    labs = [];
    splots = ['Jtot','J']; # which subplots to plot

    for i in range(len(Us)):
        U = Us[i];
        datafs.append("dat/tuneU/fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_U"+str(U)+".npy");
        labs.append("U = "+str(U) );
    
    plot.CompObservables(datafs, nleads, Vg, labs, whichi = 0, splots = splots);

    








