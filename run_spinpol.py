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
verbose = 3;
nleads = (4,4);
nelecs = (sum(nleads)+1,0); # half filling
get_data = True; # whether to run computations, if not data already exists

# phys params, must be floats
tl = 1.0;
th = 0.1; # can scale down and same effects are seen. Make sure to do later
Vb = -0.5;
mu = 0.0;
Vg = -1.73
U = -2*Vg;
Bs = [tl*5, tl*5, tl*5,tl*5,tl*5, tl*5, tl*5, tl*5, tl*5];
thetas = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi];
Bs = [tl*5, tl*5,tl*5, tl*5,tl*5];
thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4,np.pi];
Rlead_pol = 1
#assert( Vg == -0.5*U);

#time info
dt = 0.04;
tf = 4.0;

datafs = [];
if get_data: # must actually compute data

    for i in range(len(Bs)): # iter over B, theta inputs
        B, theta = Bs[i], thetas[i];
        params = tl, th, Vb, mu, Vg, U, B, theta;
        fname = siam_current.DotData(nleads, nelecs, tf, dt, phys_params=params, Rlead_pol = Rlead_pol, prefix = "spinpolsweep/", verbose = verbose);

if True: # already there
    splots = ['Jtot','occ','delta_occ','Sz','delta_Sz','Szleads']; # which subplots to plot
    for i in range(len(Bs)):
        datafs.append("dat/DotData/spinpolsweep/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(Bs[i])+"_t"+str(thetas[i])[:3]+"_Vg"+str(Vg)+".npy");
        
    plot.CompObservablesB(datafs, nleads, Bs,thetas, Vg, whichi = 0, splots = splots);
    plot.CompConductancesB(datafs, thetas, [1.5,2.5], Vb);

    








