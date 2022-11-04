'''
Christian Bunker
M^2QM at UF
October 2022

Examining time-dep perturbation theory using td-FCI tools
'''

from transport import stm_utils, tdfci

import numpy as np
import matplotlib.pyplot as plt

# top level
verbose = 5

# sample states
samp_energies = np.array([10]); # in eV;
N_samp = len(samp_energies);

# tip states
tip_energies = np.array([4,8]);
N_tip = len(tip_energies);
Norbs = N_samp + N_tip;

# initial state ( e in sample)
h1e, g2e, i_state, mol_inst, scf_inst = stm_utils.initial_state(samp_energies, tip_energies, verbose = verbose);

# perturb by coupling the sample states to tip states
thyb = 0.1; # in eV
for sampi in range(N_samp):
    for tipi in range(N_tip):
        h1e[sampi, N_samp+tipi] += thyb;
        h1e[N_samp+tipi, sampi] += thyb;
print(h1e);

# time propagate
# since energy is in eV, time is in hbar/eV = 6.58*10^-16 sec
# since perturbation is 1/10 eV, timescale should ??
tf = 10;
dt = 0.1;
civecs, observables = tdfci.kernel(h1e, g2e, i_state, mol_inst, scf_inst, tf, dt, verbose = verbose);

# plot
fig, axes = plt.subplots(Norbs);
for orbi in range(Norbs):
    axes[orbi].plot(observables[:,0], observables[:,2+orbi]);
plt.show();







