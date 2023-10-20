'''
Christian Bunker
M^2QM at UF
August 2021

1_1_1 system to match analytical results
'''

from transport import tdfci
from transport.tdfci import wrappers
from transport import fci_mod

import numpy as np
    
import sys

#### top level
verbose = 4;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists
nleads = (2,2);
nelecs = (2,0); # one electron on dot and one itinerant
ndots = 1;
myspinstate = "ab";
myprefix = "data/fci_dot/";

# physical params
tl = 1.0
th = 1.0;
td = 1.0;
Vb = -0.0;
mu = 0.0
Vg = -40.0;
U = 80.0;
B = 1000.0;
theta = 0.0;

#time info
dt = 0.1;
tf = 1.0;

if(get_data): # command line tells whether to get data
    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    tdfci.wrappers.SiamData(nleads, nelecs, ndots, tf, dt, phys_params = params, spinstate = myspinstate, prefix = myprefix, namevar = "Vg", verbose = verbose);

else: # plot data
    datafs = sys.argv[2:];
    labs = [];
    datafs.append("dat/analyt/fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg"+str(Vg)+".npy");
    labs.append("U = "+str(U));
    splots = ['Jup','Jdown','occ','delta_occ','Sz','Szleads']; # which subplots to make
    fci_mod.plot.CompObservables(datafs, labs, splots = splots, mytitle = "Featureless impurity");
