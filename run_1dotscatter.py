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
th = 0.5; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = -10.0;
U = abs(4*Vg);
B = 1000;
theta = 0.0;

#time info
dt = 0.01;
tf = 120.0;

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = spinstate, prefix = "dat/1dotscatter/th/", namevar = "th", verbose = verbose);

else:

    import plot
    import matplotlib.pyplot as plt

    # plot results
    datafs = sys.argv[2:]
    splots = ['occ','Sz','E']; # which subplots to plot
    mysites = ['L1','L2','D','R1','R2'];
    title = "Itinerant up electron, up impurity";
    paramstr = "$t_h$ = "+str(th)+"\n$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    plot.PlotObservables(datafs[0], sites = mysites, splots = splots, mytitle = title, paramstr = paramstr);

    # find T, R
    Ts, Rs = plot.TandR(datafs, nleads, mysites);
    thybs = [1.0, 0.9, 0.8,0.7,0.6,0.5];
    plt.plot(thybs, Ts);
    plt.plot(thybs, Rs);
    plt.show();

    








