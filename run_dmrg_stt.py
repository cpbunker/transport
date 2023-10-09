'''
Christian Bunker
M^2QM at UF
October 2023


'''

from transport import tddmrg, fci_mod
from transport.fci_mod import ops, ops_dmrg

import numpy as np
import matplotlib.pyplot as plt

import sys
import time
import json

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
data_file = sys.argv[1];
spinless = True;
do_fci = True;
# if this is a .txt, then this is a json with numerical params for new data run
# if this is a .npy, it is where some data is already stored

# physical params
timefinal = 0.1;
timestep = 0.01;
params_dict = json.load(open(data_file));
nelec_tup = (params_dict["Ne"],0)

#### do fci as a test #### !!!!!!

# build up t<0 Hamiltonian
h_base, g_base = tddmrg.Hsys_base(data_file);
polarize_aa, polarize_bb = tddmrg.Hsys_polarizer(data_file);
hinitial_aa, hinitial_bb = h_base+polarize_aa, h_base+polarize_bb;
print("h_up,up = \n",hinitial_aa);
print("h_down,down = \n",hinitial_bb);

# fci gd state
from pyscf import fci
harg, garg = hinitial_aa, g_base
fci_solver = fci.direct_nosym.FCI();
Es, psiEs = fci_solver.kernel(harg, garg, len(harg), nelec_tup);
psiEs = psiEs.T;
psi0 = psiEs[0];

# ground state observables
x, y = fci_mod.ops.charge_vs_site(psi0, h_base.shape[0]);

# plot
fig, ax = plt.subplots();
ax.plot(x,y, label="$N_e = $ {:.2f}".format(np.sum(y)));
ax.set_xlabel("Site");
ax.set_ylabel("$|\psi|^2$");
ax.set_title("FCI");
plt.legend();
plt.show();

assert False
##################################################################################
#### run for new data
if(data_file[-4:] == ".txt"):
    start = time.time();

    # dmrg info
    bdims = [700, 800, 900, 1000];
    noises = [1e-4, 1e-5, 1e-6, 0.0];

    # build up t<0 Hamiltonian
    h_base, g_base = tddmrg.Hsys_base(data_file);
    polarize_aa, polarize_bb = tddmrg.Hsys_polarizer(data_file);
    hinitial_aa, hinitial_bb = h_base+polarize_aa, h_base+polarize_bb;
    print("h_up,up = \n",hinitial_aa);
    print("h_down,down = \n",hinitial_bb);

    # DMRG initial Matrix Product State
    hinitial_tup = (hinitial_aa, hinitial_bb);
    hbase_tup = (np.copy(hbase), np.copy(hbase));
    ginitial_tup = (g_base, g_base, g_base);
    if (not spinless):
        harg, garg, hneq = hinitial_tup, ginitial_tup, hbase_tup;
    else:
        harg, garg, hneq = hinitial_aa, g_base, hbase;      
    dmrg_ham, dmrg_mpo, dmrg_mps = fci_mod.arr_to_mpo(harg, garg,
                            nelec_tup, bdims[0], spinless = spinless, cutoff = 1e-6);
    dmrg_mpe = fci_mod.mpo_to_mpe(dmrg_mpo, dmrg_mps);
    
    # DMRG algorithm modifies the Matrix Product State in place
    dmrg_output = dmrg_mpe.dmrg(bdims=bdims, noises=noises, iprint=0);

    # time dep DMRG algorithm
    tddmrg.kernel(harg, garg, hneq, nelec_tup, bdims, timefinal, timestep, verbose=verbose);

    # finish
    stop = time.time();
    print("\nTotal time = ",stop-start);

    # visualize
    x, y = fci_mod.ops_dmrg.charge_vs_site(dmrg_mps, h_base.shape[0], dmrg_ham);

    # plot
    fig, ax = plt.subplots();
    ax.plot(x,y);
    ax.set_xlabel("Site");
    ax.set_ylabel("$|\psi|^2$");
    plt.show();









