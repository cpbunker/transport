'''
Christian Bunker
M^2QM at UF
August 2021

1_1_0 system to match analytical results
'''

import td_fci
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt
    
import sys

#### top level
verbose = 3;
nleads = (1,0);
nelecs = (1,1);
splots = ['Jtot','occ','Sz','Szleads']; # which subplots to make
th = 1.0;
Vb = -0.01;
Vg = 0.0;
B = 5.0;
theta = 0.0;

#time info
dt = 0.004;
tf = 2*3.14;

#### benchmark with spinfree td fci
if( int(sys.argv[1])): # run data

    # spin free can only do U=0
    U = 0.0;
    params = 0.0, th, Vb, 0.0, U; # featureless dot # no mu or mag field in spin free

    # since spinfree, have to run up and down calculations separately
    # get observables for up case
    obs_up = td_fci.SpinfreeTest(nleads, (1,0), tf, dt, phys_params = params, verbose = verbose);

    # get observables for down case
    obs_down = td_fci.SpinfreeTest(nleads, (0,1), tf, dt, phys_params = params, verbose = verbose);

    # combine into actual observables of a 1 up 1 down system
    observables = [obs_up[0], obs_up[1]+obs_down[1], obs_up[2], obs_down[2], obs_up[4]+obs_down[4], obs_up[5]+obs_down[5],obs_up[6]+obs_down[6],obs_up[7]+obs_down[7],obs_up[8]+obs_down[8],obs_up[9]+obs_down[9] ];
    fname = "dat/bench/spinfree/110_updown.npy"
    np.save(fname, observables);
    print("Saving data to "+fname);

if( int(sys.argv[2])): # plot
    fname = "dat/bench/spinfree/110_updown.npy";
    plot.PlotObservables(fname, nleads = nleads, splots = splots);

#### sweep U vals with ASU formalism

nleads_ASU = (1,1);
nelecs_ASU = (2,0);
Us = np.array([0.0, 10.0, 20.0, 50.0]);

if( int(sys.argv[3])): # command line tells whether to get data

    for U in Us: # sweep U vals

        params = 0.0, th, Vb, 0.0, Vg, U, B, theta, 0.0;
        siam_current.DotData(nleads_ASU, nelecs_ASU, tf, dt, phys_params = params, prefix = "dat/param_tuning/Usweep/110_Vbias01", verbose = verbose);

if( int(sys.argv[4])):
    datafs = [];
    labs = [];
    for i in range(len(Us)):
        U = Us[i];
        datafs.append("dat/param_tuning/Usweep/110_Vbias01/fci_1_1_1_e"+str(sum(nelecs_ASU))+"_B"+str(B)+"_t"+str(theta)+"_U"+str(U)+".npy");
        labs.append("U = "+str(U));
    plot.CompObservables(datafs, nleads_ASU, "", labs,whichi = 3, mytitle = "U sweep",splots = splots);
