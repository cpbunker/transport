'''
Christian Bunker
M^2QM at UF
February 2024


'''

from transport import tddmrg, tdfci
from transport.tddmrg import plot
from transport.tdfci import utils

import numpy as np
import matplotlib.pyplot as plt

import time
import json
import sys
import os
print(">>> PWD: ",os.getcwd());

##################################################################################
#### wrappers

def check_observables(params_dict,psi,eris_or_driver, none_or_mpo, the_time, block):
    '''
    Print update on selected observables
    '''
    if(not isinstance(block, bool)): raise TypeError;
    if(not block): raise NotImplementedError;
    print("\nTime = {:.2f}".format(the_time));

    # check gd state
    check_E_dmrg = tddmrg.compute_obs(psi, none_or_mpo, eris_or_driver);
    print("Total energy = {:.6f}".format(check_E_dmrg));
    impo = eris_or_driver.get_identity_mpo()
    check_norm = eris_or_driver.expectation(psi, impo, psi)
    print("WF norm = {:.6f}".format(check_norm));

    # fermionic spin, impurity spin, current through all impurity sites
    central_sites = np.arange(params_dict["NL"],params_dict["NL"]+params_dict["NFM"]);
    for sitei in central_sites:
        occ_mpo = tddmrg.get_occ(eris_or_driver, sitei, block);
        occ_val = tddmrg.compute_obs(psi, occ_mpo, eris_or_driver);
        sz_mpo = tddmrg.get_sz(eris_or_driver, sitei, block);
        sz_val = tddmrg.compute_obs(psi, sz_mpo, eris_or_driver);
        Sdz_mpo = tddmrg.get_Sd_mu(eris_or_driver, sitei, block);
        Sdz_val = tddmrg.compute_obs(psi, Sdz_mpo, eris_or_driver);
        Jimp_val = tddmrg.conductance_wrapper(psi, eris_or_driver, sitei, block);
        Jimp_val *= np.pi*params_dict["th"]/params_dict["Vb"];
        print("<occ j={:.0f}> = {:.6f}".format(sitei, occ_val));
        print("<sz  j={:.0f}> = {:.6f}".format(sitei, sz_val));
        print("<Sdz j={:.0f}> = {:.6f}".format(sitei, Sdz_val));
        print("<J   j={:.0f}>/Vb = {:.6f}".format(sitei, Jimp_val));
                           
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name)); print(">>> Params = ",params);
is_block = True;

# unpacking
myNL, myNFM, myNR = params["NL"], params["NFM"], params["NR"];
myNe = myNL+myNFM+myNR; # total num electrons. For fci, should all be input as spin up
nloc = 2; # spin dofs

# checks
pass;

#### Initialization
####
####
init_start = time.time();
    
# init ExprBuilder object with terms that are there for all times
H_driver, H_builder = tddmrg.H_SIETS_builder(params, is_block, scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder

# add in t<0 terms
H_driver, H_mpo_initial = tddmrg.H_SIETS_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);
    
# gd state
if(is_block):
    gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                         bond_dim=params["bdim_0"][0] )
    gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,#tol=1e-24, # <------ !!!!!!
        bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], 
        cutoff=params["cutoff"], iprint=2); # set to 2 to see Mmps
    eris_or_driver = H_driver;
    print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

init_end = time.time();
print(">>> Init compute time = "+str(init_end-init_start));

#### Observables
####
####
mytime=0;

# plot observables
check_observables(params, gdstate_mps_inst, H_driver, H_mpo_initial, mytime, is_block);
plot.snapshot_bench(gdstate_mps_inst, eris_or_driver,
        params, json_name, mytime, is_block); 

#### Time evolution
####
####
H_driver_dyn, H_builder_dyn = tddmrg.H_SIETS_builder(params, is_block, scratch_dir=json_name, verbose=verbose);
if(is_block):
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose);
    tddmrg.kernel(params, H_driver_dyn, H_mpo_dyn,gdstate_mps_inst,
                  check_observables,json_name, verbose=2) # set to 2 to see mmps
