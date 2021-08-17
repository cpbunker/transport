'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for prepping dot spin state with B field, getting current output
'''

import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### prepare dot in diff spin states

# top level
verbose = 5;
nleads = (2,2);
nelecs = (sum(nleads)+1,0); # half filling
get_data = True; # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = tl/10; # can scale down and same effects are seen. Make sure to do later
Vb = -1/100*tl;
mumin, mumax = -20,50
mus = np.linspace(-20,50,11);
# Vg will be = mu
U = 100*tl;
B = 5*tl;
theta = 0.0;
phi = 0.0;

#time info
dt = 0.0004;
tf = 5.0;

if get_data: # must actually compute data

    for i in range(len(mus)): # iter over Vg vals;
        mu = mus[i];
        Vg = mu
        params = tl, th, Vb, mu, Vg, U, B, theta, phi;
        siam_current.DotData(nleads, nelecs, tf, dt, prefix = "", phys_params=params, verbose = verbose);


# plot results
datafs = [];
labs = [];
splots = ['Jtot','J']; # which subplots to plot

else:
    for i in range(len(Vgs)):
        Vg = Vgs[i];
        datafs.append("fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_Vg"+str(Vg)+".npy");
        labs.append("Vg = "+str(Vg) );
    
    plot.CompObservables(datafs, nleads, Vg, labs, whichi = 0, splots = splots);

    








