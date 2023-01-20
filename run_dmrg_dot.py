'''
Christian Bunker
M^2QM at UF
September 2021

Half filled 1D TB band of itinerant e's interacting with a single quantum dot
under a small bias voltage.
solved in time-dependent DMRG (approximate many body QM) method in transport/tddmrg
'''
from transport import tddmrg
from transport.tddmrg import wrappers
from transport import fci_mod
from transport.fci_mod import plot

import numpy as np

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (3,3);
nelecs = (2,0); # one electron on dot and one itinerant
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
spinstate = "ab";
myprefix = "data/dot/"

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
timestep = 0.1;
timefinal = 1.0;

# dmrg info
bdims = [700, 800, 900, 1000];
noises = [1e-4, 1e-5, 1e-6, 0.0];

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    tddmrg.wrappers.siam_data(nleads, nelecs, ndots, timefinal, timestep, params, bdims, noises, spinstate = spinstate, prefix = myprefix, verbose = verbose);

else:

    # plot results
    datafs = sys.argv[2:]
    splots = ['lead_occ','lead_Sz','E']; # which subplots to plot
    mysites = ['L1','L2','L3','L4','D','R1','R2','R3','R4'];
    title = "Itinerant electron scatters from spin impurity";
    paramstr = "$t_h$ = "+str(th)+"\n$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)
    fci_mod.plot.PlotObservables(datafs[0], sites = mysites, splots = splots, mytitle = title, paramstr = paramstr);

    








