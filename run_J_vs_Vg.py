'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for prepping dot spin state with B field, getting current output
Now assuming polarizer btwn Rlead, dot
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
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = tl/10; # can scale down and same effects are seen. Make sure to do later
Vb = -1/100*tl;
mu = 2.0*tl;
U =  0.0
B = 5.0*tl;
theta = 0.0;
delta_Vg = 2.0;
Vg_step = 0.2;
Vgs = np.linspace(mu+B/2 - delta_Vg,mu+B/2 + delta_Vg,1+int(abs(2*delta_Vg)/Vg_step));

#time info
dt = 0.04;
tf = 5.0;

if get_data: # must actually compute data

    for i in range(len(Vgs)): # iter over Vg vals;
        Vg = Vgs[i];
        params = tl, th, Vb, mu, Vg, U, B, theta;
        siam_current.DotData(nleads, nelecs, tf, dt, params, prefix = "dat/zeeman/U0/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[2:]
    labs = Vgs # one label for each Vg
    splots = ['Jup','Jdown','occ','Sz']; # which subplots to plot
    title = "Current between impurity and left, right leads"
    plot.CompObservables(datafs, labs, splots = splots, mytitle = title, leg_title = "$V_g$", leg_ncol = 3);

    








