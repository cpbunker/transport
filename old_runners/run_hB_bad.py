'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for prepping dot spin state with B field
'''

import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 4;
nleads = (2,2);
nelecs = (sum(nleads)+1,0); # half filling
get_data = False # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 0.4;
Vb = -0.005;
mu = 0.0;
Vg = -0.5;
U = 1.0
Bs = [tl*5, tl*5, tl*5,tl*5,tl*5];
thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi];

#time info
dt = 0.01;
tf = 5.0;

datafs = [];
if get_data: # must actually compute data

    for i in range(len(Bs)): # iter over B, theta inputs
        B, theta = Bs[i], thetas[i];
        params = tl, th, Vb, mu, Vg, U, B, theta;
        fname = siam_current.DotData(nleads, nelecs, tf, dt, phys_params=params, verbose = verbose);
        datafs.append(fname);

else: # already there
    splots = ['Jtot','J','Sz']; # which subplots to plot
    for i in range(len(Bs)):
        datafs.append("dat/DotData/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(Bs[i])+"_t"+str(thetas[i])[:3]+"_Vg"+str(Vg)+".npy");
        
    plot.CompObservablesB(datafs, nleads, Bs,thetas, Vg, splots = splots);

    








