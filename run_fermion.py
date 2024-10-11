'''
Christian Bunker
M^2QM at UF
August 2024

Use density matrix renormalization group (DMRG) code from Huanchen Zhai (block2)
to study a 1D wire with itinerant fermions
'''

from transport import tddmrg
from transport.tddmrg import plot

from pyblock2.driver import core
import numpy as np
import matplotlib.pyplot as plt

import time
import json
import sys
import os
print(">>> PWD: ",os.getcwd());

##################################################################################
#### wrappers

def get_orbital_entropies_use_npdm():
    raise NotImplementedError;

def check_observables(params_dict,psi,eris_or_driver, none_or_mpo,the_time,block):
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

    # divide sites
    Nsites = params_dict["NL"] + 2*params_dict["NFM"] + params_dict["NR"];
    central_d = np.arange(params_dict["NL"],params_dict["NL"]+2*params_dict["NFM"],2)+1;
    central_j = central_d - 1;
    all_j = np.append(np.arange(params_dict["NL"]), np.append(central_j, np.arange(central_d[-1]+1, central_d[-1]+1+params_dict["NR"])));

    # itinerant e's
    for j in all_j: 
        occ_mpo = tddmrg.get_occ(eris_or_driver, j, block);
        occ_dmrg = tddmrg.compute_obs(psi, occ_mpo, eris_or_driver);
        print("<occ  j={:.0f}> = {:.6f}".format(j, occ_dmrg));

    # central d site (localized spin) observables
    for d in central_d:
        sz_mpo = tddmrg.get_sz(eris_or_driver, d, block);
        sz_dmrg = tddmrg.compute_obs(psi, sz_mpo, eris_or_driver);
        #print("<sz   d={:.0f}> = {:.6f}".format(d, sz_dmrg));
        sz2_mpo = tddmrg.get_sz2(eris_or_driver, d, block);
        sz2_dmrg = tddmrg.compute_obs(psi, sz2_mpo, eris_or_driver);
        print("<sz   d={:.0f}> = {:.6f}, <sz^2 d={:.0f}> = {:.6f}".format(d, sz_dmrg, d, sz2_dmrg), "(-> 0.25 means localization)");
        #print("<sz^2 d={:.0f}> = {:.6f} (Need 0.25 for localization)".format(d, sz2_dmrg));

    # (S1+S2)^2
    S2_dmrg = tddmrg.S2_wrapper(psi, eris_or_driver, central_d[:2], is_impurity=False, block=block);
    print("<(S1+S2)^2>= {:.6f}".format(S2_dmrg));

    # one orbital von Neumann entropies
    ents1 = tddmrg.oneorb_entropies_wrapper(psi, eris_or_driver, central_d, np.ones_like(central_d), block);
    ents2 = tddmrg.twoorb_entropies_impurity(psi, eris_or_driver, central_d, block);

    # mutual information. Use convention that MI[p,q] >= 0, ie 2006 White Eq (8)
    # NB same as (-1) * 2013 Reiher Eq (13)
    minfo = 0.5 * (ents1[:, None] + ents1[None, :] - ents2) * (1 - np.identity(len(ents1))); 

    # results
    which_mis = 1*central_d;
    if("Bsd" in params_dict.keys()): which_mis = [central_j[0], central_d[0]];
    dsite_mask = np.array([True if site in which_mis else False for site in range(Nsites)]);
    print("ODM1 =\n", ents1[dsite_mask]);
    print("ODM2 =");
    for site in range(Nsites):
        if(dsite_mask[site]):
            print(ents2[site][dsite_mask], " d = {:.0f}".format(site));
    print("MI = (max = {:.6f})".format(np.log(2)));
    dsite_mask = np.array([True if site in which_mis else False for site in range(Nsites)]);
    for site in range(Nsites):
        if(dsite_mask[site]):
            print(minfo[site][dsite_mask], " d = {:.0f}".format(site));

    return;

##################################################################################
#### run code
if(__name__ == "__main__"):

    # top level
    verbose = 2; assert verbose in [1,2,3];
    np.set_printoptions(precision = 6, suppress = True);
    json_name = sys.argv[1];
    params = json.load(open(json_name)); print(">>> Params = ",params);
    is_block = True;

    # unpacking
    myNe, myNFM, myTwoSz =  params["Ne"], params["NFM"], params["TwoSz"];

    # checks
    special_cases = ["BFM_first", "Bsd"];
    special_cases_flag = False;
    for case in special_cases:
        if(case in params.keys()):print(">>> special case: ",case); special_cases_flag = True;
    if(not special_cases_flag): 
        espin = myNe*np.sign(params["Be"]);
        locspin = myNFM*np.sign(params["BFM"]);
        assert(espin+locspin == myTwoSz);

    #### Initialization
    ####
    ####
    init_start = time.time();

    # init ExprBuilder object with terms that are there for all times
    H_driver, H_builder = tddmrg.H_fermion_builder(params, is_block, scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder

    # add in t<0 terms
    H_driver, H_mpo_initial = tddmrg.H_fermion_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);

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
    check_observables(params, gdstate_mps_inst, H_driver, H_mpo_initial, mytime, is_block);
    #plot.snapshot_bench(gdstate_mps_inst, H_driver,
    #        params, json_name, mytime, is_block);

    #### Time evolution
    ####
    ####
    H_driver_dyn, H_builder_dyn = tddmrg.H_fermion_builder(params, is_block, 
        scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose); # we skip t<0 terms now
    tddmrg.kernel(params, H_driver_dyn, H_mpo_dyn,
                  gdstate_mps_inst,check_observables,None,json_name,verbose=2) # set to 2 to see mmps


