'''
Christian Bunker
M^2QM at UF
August 2021

Things I don't yet understand about td-DMRG vs td-FCI
'''

import td_fci
import td_dmrg
import siam_current
import plot

import numpy as np
import matplotlib.pyplot as plt


##################################################################################
#### 212 system

verbose = 5;
nleads = (2,2);
nelecs = (3,2); # half filling
nelecs_ASU = (sum(nleads)+1,0); # all spin up formalism
splots = ['Jtot','occ','delta_occ','Sz','E']; # which subplots to make
B = 5.0
theta = 0.0
Rlead_pol = 1;

# ASU td fci results
f = "dat/DotData/spinpol/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg-0.5.npy"
plot.PlotObservables(f, nleads = nleads, splots = splots);

# ASU td dmrg results
f = "dat/DotDataDMRG/spinpol/"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)+"_t"+str(theta)+"_Vg-0.5.npy"
f = siam_current.DotDataDmrg(nleads, nelecs_ASU, 5.0, 0.01, phys_params = (1.0, 0.4, -0.005, 0.0, -0.5, 1.0, B, theta), Rlead_pol = 1, bond_dims_i = 450);
plot.PlotObservables(f, nleads = nleads, splots = splots);





