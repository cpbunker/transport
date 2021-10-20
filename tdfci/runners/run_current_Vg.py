'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for getting, analyzing, plotting data for a dot
(1 site impurity w/ Vg, U) across sweep of Vg values

'''

import plot
import siam_current

import numpy as np

#################################################
#### inputs

# top level inputs
get_data = False;
plot_J = True;
plot_Fourier = False;
verbose = 5;
splots = ['Jtot','Sz','occ','delta_occ','Szleads'];
Rlead_pol = 1;

# time
tf = 5.0;
dt = 0.04;

# physical inputs
nleads = (2,2);
nimp = 1;
nelecs = (sum(nleads) + nimp, 0);
Vgs = [-0.9,-0.7,-0.5,-0.3, -0.1,0.5];  # should be list
Vgs = [-0.5,-0.3, -0.1,0.5];  # should be list
B = 5.0; # prep dot in down state always. NB starting thyb at 1e-5
theta = 0.0;

#################################################
#### get data for current thru dot

if get_data:
    for Vg in Vgs:
        myparams = 1.0, 0.4, -0.005, 0.0, Vg, 8*1.0, B, theta; # std inputs except Vg
        prefix = "VgSweep/";
        if(Rlead_pol == 1): prefix = "VgSweepSpinpolLargeU/";
        siam_current.DotData(nleads, nelecs, tf, dt, phys_params = myparams, Rlead_pol = Rlead_pol, prefix = prefix, verbose = verbose);


#################################################
#### plot current data

# plot inputs
title = "Dot impurity:\n"+str(nleads[0])+" left sites, "+str(nleads[1])+" right sites, $t_{hyb} = 10^{-5}$ -> 0.4, B = "+str(B); 

if plot_J:
    # plot J vs t, E vs t, fourier freqs, ind'ly or across Vg, mu sweep
    folder = "dat/DotData/VgSweep/"; 
    if( Rlead_pol == 1): folder = "dat/DotData/VgSweepSpinpolLargeU/";
    plot.CurrentPlot(folder, nleads, nimp, nelecs, Vgs, B, theta, whichi = 0, splots=splots, mytitle = title);


if plot_Fourier:
    # plot J vs t with energy spectrum, ∆E -> w, and fourier modes
    plot.FourierEnergyPlot(folder, nleads, nimp, nelecs, mus, Vgs, energies, mytitle = title);









