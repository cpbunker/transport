'''
Christian Bunker
M^2QM at UF
September 2021

Itinerant electron scattering from a single dot

Scattering mechanism is Heisenberg exchange J=t^2/Vg
'''

import siam_current

import numpy as np
import scipy

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (4,4);
nelecs = (2,0); # one electron on dot and one itinerant
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = "ab";

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = -40.0;
U = 80.0
B = 1000;
theta = 0.0;

#time info
dt = 0.005;
tf = 150.0;

# dmrg info
bdims = [700, 800, 900, 1000];
noises = [1e-4, 1e-5, 1e-6, 0.0];

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotDataDmrg(nleads, nelecs, ndots, tf, dt, params, bdims, noises, spinstate = spinstate, prefix = "", namevar = "Vg", verbose = verbose);

else:

    import plot
    import matplotlib.pyplot as plt

    # plot results
    datafs = sys.argv[2:]
    splots = ['lead_occ','lead_Sz','E']; # which subplots to plot
    mysites = ['L1','L2','L3','L4','D','R1','R2','R3','R4'];
    title = "Itinerant electron scatters from spin impurity";
    paramstr = "$t_h$ = "+str(th)+"\n$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    plot.PlotObservables(datafs[0], sites = mysites, splots = splots, mytitle = title, paramstr = paramstr);

    








