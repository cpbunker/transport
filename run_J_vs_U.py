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
Us = [0.0,1.0, 2.0, 10.0]
B = 5*tl;
theta = 0.0;
phi = 0.0;

#time info
dt = 0.004;
tf = 10.0;

if get_data: # must actually compute data

    for i in range(len(Us)): # iter over Vg vals;
        U = Us[i]
        params = tl, th, Vb, mu, Vg, U, B, theta, phi;
        siam_current.DotData(nleads, nelecs, tf, dt, prefix = "dat/param_tuning/tuneU/", phys_params=params, namevar="U", verbose = verbose);

else:

    import plot

    # plot results
    datafs = [];
    labs = [];
    splots = ['J','Jup','Jdown','Sz']; # which subplots to plot

    for i in range(len(Us)):
        U = Us[i];
        datafs.append("dat/param_tuning/tuneU/fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_U"+str(U)+".npy");
        labs.append("U = "+str(U) );

    title = "Coulomb barrier for a down spin impurity, $V_g = \mu = 10.0$"
    plot.CompObservables(datafs, nleads, Vg, labs, mytitle = title, whichi = 0, splots = splots);

    








