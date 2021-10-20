'''
Christian Bunker
M^2QM at UF
September 2021

Itinerant electron scattering from two sequentially placed quantum dots
'''

import siam_current

import numpy as np

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (2,2);
nelecs = (3,0); # one electron on each dot and one itinerant
ndots = 2;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = sys.argv[2];

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = -10.0;
U = abs(4*Vg);
B = 10;
theta = 0.0;

#time info
dt = 0.01;
tf = 400.0;

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = spinstate, prefix = "dat/2dotscatter/", verbose = verbose);

else:

    import plot

    # plot results
    datafs = sys.argv[3]
    splots = ['occ','Sz','concur']; # which subplots to plot
    title = str(nleads[0])+" LL sites, "+str(ndots)+" dots, "+str(nleads[1])+" RL sites, initial state |"+spinstate+" $\\rangle$";
    paramstr = "$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    plot.PlotObservables(datafs, sites = ['L1','L2','LD','RD','R1','R2'], splots = splots, mytitle = title, paramstr = paramstr);

    








