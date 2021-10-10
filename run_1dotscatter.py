'''
Christian Bunker
M^2QM at UF
September 2021

Itinerant electron scattering from a single dot

Scattering mechanism is Heisenberg exchange J=t^2/Vg
'''

import ops
import siam_current

import numpy as np
import scipy

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (2,2);
nelecs = (2,0); # one electron on dot and one itinerant
ndots = 1;
get_data = int(sys.argv[1]); # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = -20.0;
Vgvals = [-20.0,-18.0,-16.0,-14.0,-12.0,-10.0,-8.0,-6.0];
U = 80.0
B = 100;
theta = 0.0;

#time info
dt = 0.01;
tf = 5.0;

if get_data: # must actually compute data
    for Vg in Vgvals:
        # custom ham
        h1e, g2e, _ = ops.spin_imp_hams(nleads, nelecs, (tl, 0.0, td, 0.0, mu, Vg, U, B, theta), verbose = verbose);
        h1e_neq, _, _ = ops.spin_imp_hams(nleads, nelecs, (tl, th, td, Vb, mu, Vg, U, 0.0, theta), verbose = verbose);
        siam_current.CustomData(h1e, g2e, h1e_neq, nelecs, tf, dt, fname = "1dot_Vg"+str(Vg)+"_U"+str(U)+".npy", verbose = verbose);   
    
else:

    import plot
    import matplotlib.pyplot as plt

    # plot results
    datafs = sys.argv[2:]
    splots = ['Jup','Jdown','occ','Sz']; # which subplots to plot
    mysites = ['L1','L2','D1','D2','R1','R2'];
    title = "Single impurity scattering";
    paramstr = "$t_h$ = "+str(th)+"\n$V_b$ = "+str(Vb)+"\n$V_g$ = "+str(Vg)+"\n$U$ = "+str(U)

    plot.PlotObservables(datafs[-1], sites = mysites, splots = splots, paramstr = paramstr);

    # find T, R against th
    if True:
        plt.style.use("seaborn-dark-palette");
        Ts, Rs = plot.TandR(datafs, nleads, mysites);

        fig, ax = plt.subplots();
        ax.scatter(Vgvals, Ts, marker = 's', label = "T");
        #ax.scatter(Vgs, Rs, marker = 's', label = "R");

        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        plt.xlabel("$V_g$");
        plt.ylabel("$T_{down}$");
        plt.legend();
        plt.show();

        '''
        def func(x,a,b):
            return a*np.power(x,b) + 0.1;
        fit, _ = scipy.optimize.curve_fit(func, thybs, Rs);
        print(fit);
        plt.plot(thybs, func(thybs, *fit), color = 'black', linestyle = 'dashed' );
        plt.legend();
        plt.xlabel("$U$");
        plt.show();
        '''

    








