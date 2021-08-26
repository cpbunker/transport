'''
Christian Bunker
M^2QM at UF
August 2021

1_1_1 system to match analytical results
'''

import td_fci
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt
    
import sys

#### top level
verbose = 4;
nleads = (2,2);
nelecs = (5,0);
splots = ['J','delta_occ']; # which subplots to make
tl = 1.0
th = 0.1;
Vb = -0.01;
mu = 0.0
Vg = 0.0;
U = 0.0;
B = 5.0;
theta = 0.0;
phi = 0.0

#time info
dt = 0.01;
tf = 2*3.14;

if( int(sys.argv[1])): # command line tells whether to get data

    params = tl, th, Vb, mu, Vg, U, B, theta, phi;
    siam_current.DotData(nleads, nelecs, tf, dt, phys_params = params, prefix = "dat/analyt/", namevar = "Vg", verbose = verbose);

if( int(sys.argv[2])):
    datafs = [];
    labs = [];
    datafs.append("dat/analyt/fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg"+str(Vg)+".npy");
    labs.append("U = "+str(U));
    plot.CompObservables(datafs, nleads, "", labs,whichi = 0, mytitle = "Featureless impurity",splots = splots);
