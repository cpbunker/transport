'''
'''

from transport import tddmrg, fci_mod, ops

import numpy as np
import matplotlib.pyplot as plt
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
get_data = int(sys.argv[1]);

# SIAM hyperparams
nleads = (4,4);
nelecs = (sum(nleads),0);
ndots = 1;
tf = 0.1;
dt = 0.01;
bdims = [700,800,900,1000];
noises = [1e-3,1e-4,1e-5,0];

# SIAM params
#        tl,  th,  td,  Vb,   mu,  Vg,   U,   B,   theta
params = 1.0, 0.4, 0.0, 0.00, 0.0, -0.5, 0.0, 0.0, 0.0;

# occupied spin orbs
source = [3,9]; # m=+2 incident up, imp down

# run or plot
if get_data:

    # set up SIAM in spatial basis
    h1e, g2e, _ = ops.dot_hams(nleads, nelecs, ndots, params, verbose = verbose);

    # convert leads from spatial to k space basis
    h1e = fci_mod.cj_to_ck(h1e, nleads);
    print(h1e[4:10,4:10]); assert False;

    # td fci
    print(np.shape(h1e), np.shape(g2e));
    tddmrg.Data(source, nleads, h1e, g2e, tf, dt, bdims, noises, verbose = verbose);   

else:

    # plot
    pass;





