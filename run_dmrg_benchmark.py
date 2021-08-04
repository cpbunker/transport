'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for benchmarking td-FCI
'''

import td_fci
import td_dmrg
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt


##################################################################################
#### 2_1_2 spinpol system

verbose = 5;
nleads = (3,3);
nelecs = (4,3); # half filling
nelecs_ASU = (sum(nleads)+1,0); # all spin up formalism
splots = ['Jtot','occ','delta_occ','Sz','E']; # which subplots to make
B = 5.0
theta = 0.0
Rlead_pol = 1;

#time info
dt = 0.01;
tf = 5.0;

# benchmark with spin free, fci code, std inputs
#f = td_fci.SpinfreeTest(nleads, nelecs, tf, dt, phys_params = None, verbose = verbose);
#f = "dat/SpinfreeTest/"+str(nleads[0])+"1"+str(nleads[1])+"_e"+str(sum(nelecs))+".npy"
#plot.PlotObservables(f, nleads = nleads, thyb = (1e-5,0.4), splots = splots);


# ASU fci code
params = 1.0, 0.4, -0.005, 0.0, -0.5, 1.0, B, theta
#siam_current.DotData(nleads,nelecs_ASU,tf,dt,phys_params = params, Rlead_pol = Rlead_pol, verbose = verbose);
f = "dat/DotData/spinpol/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg-0.5.npy"
plot.PlotObservables(f, nleads = nleads, splots = splots);

# test ASU, dmrg code with std inputs

params = 1.0, 0.4, -0.005, 0.0, -0.5, 1.0, B, theta
#siam_current.DotDataDmrg(nleads,nelecs_ASU,tf,dt,phys_params = params, Rlead_pol = Rlead_pol, verbose = verbose);
f = "dat/DotDataDMRG/spinpol/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg-0.5.npy"
plot.PlotObservables(f, nleads = nleads, splots = splots);





