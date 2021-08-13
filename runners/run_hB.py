'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for prepping dot spin state with B field, comparing results
'''

import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 2;
nleads = (2,2);
nelecs = (sum(nleads)+1,0); # half filling

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

# get data for each theta value, store in txt file
for i in range(len(Bs)): # iter over B, theta inputs
    B, theta = Bs[i], thetas[i];
    params = tl, th, Vb, mu, Vg, U, B, theta;
    fname = siam_current.DotData(nleads, nelecs, tf, dt, phys_params=params, verbose = verbose);
    datafs.append(fname);

# compare data
splots = ['Jtot','J','Sz']; # which subplots to plot
plot.CompObservablesB(datafs, nleads, Bs,thetas, Vg, splots = splots);

    








