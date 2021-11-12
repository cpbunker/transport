'''
'''

from transport import tddmrg, tdfci, fci_mod, ops

import numpy as np
import matplotlib.pyplot as plt
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
get_data = int(sys.argv[1]);
fci = True;

# SIAM hyperparams
nleads = (4,4);
ndots = 1;
tf = 1.0;
dt = 0.1;
bdims = [200,300,400,500];
noises = [1e-3,1e-4,1e-5,0];

# SIAM params
#        tl,  th,  td,  Vb,   mu,  Vg,   U,   B,   theta
params = 1.0, 0.4, 0.0, 0.00, 0.0, -0.5, 0.0, 0.0, 0.0;

# occupied spin orbs
source = [6,9]; # m=+2 incident up, imp down

# run or plot
if get_data:

    # set up SIAM in spatial basis
    h1e, g2e, _ = ops.dot_hams(nleads, ndots, params, verbose = verbose);

    # convert leads from spatial to k space basis
    h1e = fci_mod.cj_to_ck(h1e, nleads);

    h1e = np.zeros((4,4));
    h1e[0,0] = -1;
    g2e = np.zeros((4,4,4,4));
    source = [];

    # time propagation
    if fci:
        tdfci.Data(source, nleads, h1e, g2e, tf, dt, verbose = verbose);
    else:
        tddmrg.Data(source, nleads, h1e, g2e, tf, dt, bdims, noises, verbose = verbose);   

else:

    # plot
    from transport import plot
    states = ["Inc -2","Inc -1", "Inc +1","Inc +2","SR","Tra -2","Tra -1", "Tra +1", "Tra +2"];
    plot.PlotObservables(sys.argv[2], states, splots = ['occ','Sz']);





