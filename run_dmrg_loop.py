'''
Christian Bunker
M^2QM at UF
October 2023

Use density matrix renormalization group (DMRG) code from Huanchen Zhai (block2)
to study a 1D array of localized spins interacting with itinerant electrons in a
nanowire. In spintronics, this system is of interest because elecrons can impart
angular momentum on the localized spins, exerting spin transfer torque (STT).
'''

import numpy as np
import matplotlib.pyplot as plt

import time
import json
import sys
import os
print(">>> PWD: ",os.getcwd());

##################################################################################
#### wrappers

def get_energy_fci(h1e, g2e, nelec, nroots=1, verbose=0):
    # convert from arrays to uhf instance
    mol_inst, uhf_inst = utils.arr_to_uhf(h1e, g2e, len(h1e), nelec, verbose = verbose);
    # fci solution
    E_fci, v_fci = utils.scf_FCI(mol_inst, uhf_inst, nroots);
    if(nroots>1): E_fci, v_fci = E_fci[0], v_fci[0];
    # ci object
    CI_inst = tdfci.CIObject(v_fci, len(h1e), nelec);
    return CI_inst, E_fci, uhf_inst;

def check_observables(the_sites,psi,eris_or_driver,block):
    if(not block):
        # site 0 spin
        s0_eris = tddmrg.get_sz(len(eris_or_driver.h1e[0]), eris_or_driver, the_sites[0], block);
        gd_s0 = tdfci.compute_obs(psi, s0_eris);
        print("Site {:.0f} <Sz> (FCI) = {:.6f}".format(the_sites[0],gd_s0));
        # site 5 (ie the impurity site) spin
        sdot_eris = tddmrg.get_sz(len(eris_or_driver.h1e[0]), eris_or_driver, the_sites[1], block);
        gd_sdot = tdfci.compute_obs(psi, sdot_eris);
        print("Site {:.0f} <Sz> (FCI) = {:.6f}".format(the_sites[1],gd_sdot));       
        # concurrence between
        C_ci = tddmrg.concurrence_wrapper(psi, eris_or_driver, the_sites, False);
        print("C"+str(the_sites)+" = ",C_ci);
    else:
        s0_mpo = tddmrg.get_sz(eris_or_driver.n_sites*2, eris_or_driver, the_sites[0], block);
        gd_s0_dmrg = tddmrg.compute_obs(psi, s0_mpo, eris_or_driver);
        print("Site {:.0f} <Sz> (DMRG) = {:.6f}".format(the_sites[0],gd_s0_dmrg));
        sdot_mpo = tddmrg.get_sz(eris_or_driver.n_sites*2, eris_or_driver, the_sites[1], block);
        gd_sdot_dmrg = tddmrg.compute_obs(psi, sdot_mpo, eris_or_driver);
        print("Site {:.0f} <Sz> (DMRG) = {:.6f}".format(the_sites[1], gd_sdot_dmrg));
        # concurrence between 
        C_dmrg = tddmrg.concurrence_wrapper(psi, eris_or_driver, the_sites, True);
        print("C"+str(the_sites)+" = ",C_dmrg);

def time_evol_wrapper(params_dict,driver_inst, mpo_inst, psi, save_name):
    '''
    '''
    evol_start = time.time();
    time_step = params_dict["time_step"];
    time_update = params_dict["tupdate"];
    time_update = time_step*int(abs(time_update/time_step)+0.1); # discrete number
    total_time = 0.0;
    Nupdates = params_dict["Nupdates"];

    # time evolve with repeated snapshots
    tevol_mps_inst = psi;
    for _ in range(Nupdates):
        total_time += time_update;

        # time evol
        tevol_mps_inst = driver_inst.td_dmrg(mpo_inst, tevol_mps_inst, 
                delta_t=complex(0,time_step), target_t=complex(0,time_update),
                bond_dims=params_dict["bdim_t"], iprint=0);

        # observables
        check_observables(params_dict["ex_sites"],tevol_mps_inst,driver_inst,True);
        plot.snapshot_bench(None, tevol_mps_inst, None, driver_inst, params_dict, save_name, time=total_time);

    evol_end = time.time();
    print(">>> Time evol compute time (DMRG only) = {:.2f}".format(evol_end-evol_start));

##################################################################################
#### run code

# top level
verbose = 1; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name));
do_fci = bool(int(sys.argv[2]));
do_dmrg = bool(int(sys.argv[3]));
assert(do_fci or do_dmrg);
print(">>> Do FCI  = ",do_fci);
print(">>> Do DMRG = ",do_dmrg);
from transport import tdfci, tddmrg
from transport.tdfci import utils, plot

# some unpacking
myNL, myNFM, myNR, myNe = params["NL"], params["NFM"], params["NR"], params["Ne"],
mynelec = (myNFM+myNe,0);
my_sites = params["ex_sites"];

# checks
espin = myNe*np.sign(params["Be"]);
locspin = myNFM*np.sign(params["BFM"]);
myTwoSz = params["TwoSz"];
special_cases = ["BFM_first", "Bsd", "Bcentral", "Bsd_x","lead_penalty"];
special_cases_flag = False;
for case in special_cases:
    if(case in params.keys()):print(">>> special case: ",case); special_cases_flag = True;
if(not special_cases_flag): assert(espin+locspin == myTwoSz);

#### Initialization
####
####
init_start = time.time();

if(do_fci): # fci gd state

    # construct arrays with terms there for all times
    H_1e, H_2e = tddmrg.Hsys_builder(params, False, verbose=verbose);

    # add in t<0 terms
    H_1e, H_2e = tddmrg.Hsys_polarizer(params, False, (H_1e, H_2e), verbose=verbose);
    print("H_1e = ");print(H_1e[2*myNL:2*(myNL+2*myNFM+2),2*myNL:2*(myNL+2*myNFM+2)]);
H_eris, gdstate_ci_inst = None, None;

if(do_dmrg): # dmrg gd state
    
    # init ExprBuilder object with terms that are there for all times
    H_driver, H_builder = tddmrg.Hsys_builder(params, True, scratch_dir=json_name, verbose=0); # returns DMRGDriver, ExprBuilder

    # add in t<0 terms
    H_driver, H_mpo_initial = tddmrg.Hsys_polarizer(params, True, (H_driver,H_builder), verbose=0);

    # gd state
    gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                             bond_dim=params["bdim_0"][0] )
    gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,
        bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], cutoff=params["cutoff"],
        iprint=2); # set to 2 to see Mmps
    print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

    # check gd state
    check_E_dmrg = tddmrg.compute_obs(gdstate_mps_inst, H_mpo_initial, H_driver);
    print("Manually computed energy (DMRG) = {:.6f}".format(check_E_dmrg));

else:
    H_driver, gdstate_mps_inst = None, None;

init_end = time.time();
print(">>> Init compute time (FCI = "+str(do_fci)+", DMRG="+str(do_dmrg)+") = "+str(init_end-init_start));

# plot observables
if(do_dmrg): check_observables(my_sites, gdstate_mps_inst, H_driver, True);
plot.snapshot_bench(gdstate_ci_inst, gdstate_mps_inst, H_eris, H_driver, params, json_name, time = 0.0);

#### Time evolution
####
if(do_fci):
    H_1e_dyn, H_2e_dyn = tddmrg.Hsys_builder(params, False, verbose=verbose);
    print("H_1e_dyn = ");print(H_1e_dyn[2*myNL:2*(myNL+2*myNFM+2),2*myNL:2*(myNL+2*myNFM+2)]);
H_eris_dyn, tevol_ci_inst = None, None;
if(do_dmrg): 
    H_driver_dyn, H_builder_dyn = tddmrg.Hsys_builder(params, True, scratch_dir = json_name, verbose=verbose);
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=0);
    time_evol_wrapper(params, H_driver_dyn, H_mpo_dyn, gdstate_mps_inst,json_name)







