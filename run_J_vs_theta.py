'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for prepping dot spin state with B field, getting current output
Now assuming polarizer btwn Rlead, dot
'''

import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 3;
nleads = (3,3);
nelecs = (sum(nleads)+1,0); # half filling
get_data = False; # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = tl/10; # can scale down and same effects are seen. Make sure to do later
Vb = -2.0*tl
mu = 0.0;
Vg = -3.0*tl;
U = -2*Vg;
Bs = [tl*5, tl*5, tl*5,tl*5,tl*5];
thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi];
phi = 0.0;

#time info
dt = 0.04;
tf = 3.0;

datafs = [];
labs = [];
splots = ['Jtot','J','delta_occ','delta_Sz','Szleads']; # which subplots 
if get_data: # must actually compute data

    for i in range(len(Bs)): # iter over B, theta inputs
        B, theta = Bs[i], thetas[i];
        params = tl, th, Vb, mu, Vg, U, B, theta, phi;
        fname = siam_current.DotData(nleads, nelecs, tf, dt, prefix = "dat/shadow/", phys_params=params, verbose = verbose);

for i in range(len(Bs)):
    datafs.append("dat/shadow/fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(Bs[i])+"_t"+str(thetas[i])[:3]+"_Vg"+str(Vg)+".npy");
    labs.append("$\\theta$ = "+str(thetas[i])[:3] );
    
plot.CompObservables(datafs, nleads, Vg, labs, whichi = 0, splots = splots);


    








