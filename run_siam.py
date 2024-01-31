'''
Christian Bunker
M^2QM at UF
January 2024

Use density matrix renormalization group (DMRG) code from Huanchen Zhai (block2)
to study the single impurity Anderson model (SIAM)
Reference results:
Garnet: https://doi.org/10.1063/5.0059257
Adrian: https://doi.org/10.1103/PhysRevB.73.195304
'''

from transport import tddmrg
from transport.tddmrg import plot

import numpy as np
import matplotlib.pyplot as plt

import time
import json
import sys
import os
print(">>> PWD: ",os.getcwd());

##################################################################################
#### wrappers

def check_observables(params_dict,psi,eris_or_driver, none_or_mpo,the_time):
    '''
    Print update on selected observables
    '''
    print("\nTime = {:.2f}".format(the_time));
    # check gd state
    check_E_dmrg = tddmrg.compute_obs(psi, none_or_mpo, eris_or_driver);
    print("Total energy = {:.6f}".format(check_E_dmrg));
    impo = eris_or_driver.get_identity_mpo()
    check_norm = eris_or_driver.expectation(psi, impo, psi)
    print("WF norm = {:.6f}".format(check_norm));

    # fermionic charge and spin in LL, Imp, RL
    Impsite = params_dict["NL"]
    sites_for_spin = [0, Impsite, Impsite+params_dict["NR"]];
    for sitei in sites_for_spin:
        sz_mpo = tddmrg.get_sz(eris_or_driver, sitei);
        sz_val = tddmrg.compute_obs(psi, sz_mpo, eris_or_driver);
        occ_mpo = tddmrg.get_occ(eris_or_driver, sitei);
        occ_val = tddmrg.compute_obs(psi, occ_mpo, eris_or_driver);
        print("<n  j={:.0f} = {:.6f}".format(sitei, occ_val));
        print("<sz j={:.0f} = {:.6f}".format(sitei, sz_val));

    # current through Imp
    Jimp_val = tddmrg.pcurrent_wrapper(psi, eris_or_driver, Impsite);
    print("<J  j={:.0f} = {:.6f}".format(Impsite, Jimp_val));
                           
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name)); print(">>> Params = ",params);

# unpacking
myNL, myNR = params["NL"], params["NR"];

# checks
pass;

#### Initialization
####
####
init_start = time.time();
    
# init ExprBuilder object with terms that are there for all times
H_driver, H_builder = tddmrg.H_SIAM_builder(params, scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder

# add in t<0 terms
H_driver, H_mpo_initial = tddmrg.H_SIAM_polarizer(params, (H_driver,H_builder), verbose=verbose);
    
# gd state
gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                         bond_dim=params["bdim_0"][0] )
gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,#tol=1e-24, # <------ !!!!!!
    bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], cutoff=params["cutoff"],
    iprint=2); # set to 2 to see Mmps
print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

init_end = time.time();
print(">>> Init compute time = "+str(init_end-init_start));

#### Observables
####
####
mytime=0;

# plot observables
check_observables(params, gdstate_mps_inst, H_driver, H_mpo_initial, mytime);
plot.snapshot_bench(gdstate_mps_inst, H_driver,
        params, json_name, time = mytime); # This will fail

#### Time evolution
####
####
H_driver_dyn, H_builder_dyn = tddmrg.H_SIAM_builder(params, scratch_dir=json_name, verbose=verbose);
H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose);
tddmrg.kernel(params, H_driver_dyn, H_mpo_dyn,gdstate_mps_inst,
                  check_observables,json_name,verbose=2) # set to 2 to see mmps


