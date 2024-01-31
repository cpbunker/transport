'''
Christian Bunker
M^2QM at UF
January 2024

Use density matrix renormalization group (DMRG) code from Huanchen Zhai (block2)
to study a 1D array of localized spins interacting with itinerant electrons in a
nanowire. In spintronics, this system is of interest because elecrons can impart
angular momentum on the localized spins, exerting spin transfer torque (STT).
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

def get_energy_fci(h1e, g2e, nelec, nroots=1, verbose=0):
    # convert from arrays to uhf instance
    mol_inst, uhf_inst = utils.arr_to_uhf(h1e, g2e, len(h1e), nelec, verbose = verbose);
    # fci solution
    E_fci, v_fci = utils.scf_FCI(mol_inst, uhf_inst, nroots);
    if(nroots>1): E_fci, v_fci = E_fci[0], v_fci[0];
    # ci object
    CI_inst = tdfci.CIObject(v_fci, len(h1e), nelec);
    return CI_inst, E_fci, uhf_inst;

def check_observables(params_dict,psi,eris_or_driver, none_or_mpo,the_time):
    '''
    Print update on selected observables
    '''
    the_sites = params_dict["ex_sites"];
    print("\nTime = {:.2f}".format(the_time));
    # check gd state
    check_E_dmrg = tddmrg.compute_obs(psi, none_or_mpo, eris_or_driver);
    print("Total energy = {:.6f}".format(check_E_dmrg));
    impo = eris_or_driver.get_identity_mpo()
    check_norm = eris_or_driver.expectation(psi, impo, psi)
    print("WF norm = {:.6f}".format(check_norm));
    if(the_sites):
        # site spins
        s0_mpo = tddmrg.get_Sd_mu(eris_or_driver, the_sites[0]);
        gd_s0_dmrg = tddmrg.compute_obs(psi, s0_mpo, eris_or_driver);
        print("<Sz d={:.0f}> = {:.6f}".format(the_sites[0],gd_s0_dmrg));
        sdot_mpo = tddmrg.get_Sd_mu(eris_or_driver, the_sites[1]);
        gd_sdot_dmrg = tddmrg.compute_obs(psi, sdot_mpo, eris_or_driver);
        print("<Sz d={:.0f}> = {:.6f}".format(the_sites[1], gd_sdot_dmrg));
        # concurrence between 
        C_dmrg = tddmrg.concurrence_wrapper(psi, eris_or_driver, the_sites);
        print("<C"+str(the_sites)+"> = {:.6f}".format(C_dmrg));

def time_evol_wrapper(params_dict, driver_inst, mpo_inst, psi, save_name, verbose=0):
    '''
    '''
    print("\n\nSTART TIME EVOLUTION (te_type = "+params_dict["te_type"]+")\n\n","*"*50,"\n\n")
    evol_start = time.time();
    time_step = params_dict["time_step"];
    time_update = params_dict["tupdate"];
    time_update = time_step*int(abs(time_update/time_step)+0.1); # discrete number
    total_time = 0.0;
    Nupdates = params_dict["Nupdates"];

    # time evolve with repeated snapshots
    tevol_mps_inst = psi;
    for timei in range(Nupdates):
        if(timei in [0]): the_verbose=verbose;
        else: the_verbose=0; # ensures verbosity only on initial time steps
        total_time += time_update;

        # time evol
        krylov_subspace = 20; # default
        if(params["te_type"] == "tdvp"): krylov_subspace = 40;
        tevol_mps_inst = driver_inst.td_dmrg(mpo_inst, tevol_mps_inst, 
                delta_t=complex(0,time_step), target_t=complex(0,time_update),
                bond_dims=params_dict["bdim_t"], cutoff=params_dict["cutoff"], te_type=params["te_type"],krylov_subspace_size=krylov_subspace,
                final_mps_tag=str(int(100*total_time)), iprint=the_verbose);

        # observables
        check_observables(params_dict,tevol_mps_inst,driver_inst,mpo_inst,total_time);
        plot.snapshot_bench(tevol_mps_inst, driver_inst, params_dict, save_name, time=total_time);

    evol_end = time.time();
    print(">>> Time evol compute time = {:.2f}".format(evol_end-evol_start));

                           
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name)); print(">>> Params = ",params);

# unpacking
myNL, myNFM, myNR, myNe = params["NL"], params["NFM"], params["NR"], params["Ne"],
myTwoSz = params["TwoSz"];
myNbuffer = 0;
if "Nbuffer" in params.keys(): myNbuffer = params["Nbuffer"];

# checks
my_sites = params["ex_sites"]; # j indices
for j in my_sites: assert(j in np.arange(myNbuffer+myNL,myNbuffer+myNL+myNFM)); # must be FM sites or conc will fail
espin = myNe*np.sign(params["Be"]);
locspin = myNFM*np.sign(params["BFM"]);
special_cases = ["BFM_first", "Bsd", "Bcentral", "Bsd_x","noFM"];
special_cases_flag = False;
for case in special_cases:
    if(case in params.keys()):print(">>> special case: ",case); special_cases_flag = True;
if(not special_cases_flag): assert(espin+locspin == myTwoSz);

#### Initialization
####
####
init_start = time.time();
    
# init ExprBuilder object with terms that are there for all times
H_driver, H_builder = tddmrg.H_STT_builder(params, scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder

# add in t<0 terms
H_driver, H_mpo_initial = tddmrg.H_STT_polarizer(params, (H_driver,H_builder), verbose=verbose);
    
# gd state
gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                         bond_dim=params["bdim_0"][0] )
gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,#tol=1e-24, # <------ !!!!!!
    bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], cutoff=params["cutoff"],
    iprint=2); # set to 2 to see Mmps
print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

# orbital interactions and reordering
if False: 
    # have to change sym type to core.SymmetryTypes.SZ in driver constructor
    int_matrix = H_driver.get_orbital_interaction_matrix(gdstate_mps_inst);
    fig, ax = plt.subplots()
    ax.matshow(int_matrix, cmap='ocean_r')
    plt.show()
    assert False

init_end = time.time();
print(">>> Init compute time = "+str(init_end-init_start));

#### Observables
####
####
mytime=0;

# plot observables
check_observables(params, gdstate_mps_inst, H_driver, H_mpo_initial, mytime);
plot.snapshot_bench(gdstate_mps_inst, H_driver,
        params, json_name, time = mytime);

#### Time evolution
####
####
H_driver_dyn, H_builder_dyn = tddmrg.H_STT_builder(params, scratch_dir=json_name, verbose=verbose);
H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose);
time_evol_wrapper(params, H_driver_dyn, H_mpo_dyn,
                  gdstate_mps_inst,json_name,verbose=2) # set to 2 to see mmps


