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
Vg = -10.0;
U = 40.0
B = 1000;
theta = 0.0;

#time info
dt = 0.01;
tf = 50.0;

if get_data: # must actually compute data

    params = tl, th, td, Vb, mu, Vg, U, B, theta;
    siam_current.DotData(nleads, nelecs, ndots, tf, dt, params, spinstate = spinstate, prefix = "", namevar = "Vg", verbose = verbose);

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

    # find T, R against th
    if False:
        Ts, Rs = plot.TandR(datafs, nleads, mysites);
        thybs = [0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        Us = [20, 40, 60, 80];
        plt.plot(thybs, Ts, label = "T");
        plt.plot(thybs, Rs, label = "R");

        # fit R to x^4
        def func(x,a,b):
            return a*np.power(x,b) + 0.1;
        fit, _ = scipy.optimize.curve_fit(func, thybs, Rs);
        print(fit);
        plt.plot(thybs, func(thybs, *fit), color = 'black', linestyle = 'dashed' );
        plt.legend();
        plt.xlabel("$U$");
        plt.show();

    # find T, R against U
    if False:
        Ts, Rs = plot.TandR(datafs, nleads, mysites);
        Us = [20, 40, 60, 80];
        plt.plot(Us, Ts, label = "T");
        plt.plot(Us, Rs, label = "R");

        # fit R to x^4
        def func(x,a,b):
            return a*np.power(x,4) + b;

        plt.legend();
        plt.xlabel("$U$");
        plt.show();

    








