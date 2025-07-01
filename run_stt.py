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

    # rice mele or not
    if(params_dict["sys_type"] in ["STT_RM"]): RMdofs = 2;
    elif(params_dict["sys_type"] in ["STT"]): pass;

    # divide sites                              
    if(params_dict["sys_type"] in ["STT"]):
        Nsites = params_dict["NL"] + params_dict["NFM"] + params_dict["NR"];
        #all_j = np.arange(0, Nsites);
        if("MSQ_spacer" in params_dict.keys()):
            central_j = np.array([params_dict["NL"],params_dict["NL"]+params_dict["NFM"]-1]);
        else:
            central_j = np.arange(params_dict["NL"],params_dict["NL"]+params_dict["NFM"]);
    elif(params_dict["sys_type"] in ["STT_RM"]):
        central_j = [];
        for j in np.arange(params_dict["NL"],params_dict["NL"]+params_dict["NFM"]):
            central_j.append(RMdofs*j);
            central_j.append(RMdofs*j+1);
    else: raise NotImplementedError;

    # impurity Sz 
    if(np.any(central_j)):
        for dsite in central_j: # site spins
            s0_mpo = tddmrg.get_Sd_mu(eris_or_driver, dsite, block);
            gd_s0_dmrg = tddmrg.compute_obs(psi, s0_mpo, eris_or_driver);
            print("<Sz d={:.0f}> = {:.6f}".format(dsite, gd_s0_dmrg));
    if(len(central_j)==2):
        # (S1+S2)^2
        S2_dmrg = tddmrg.S2_wrapper(psi, eris_or_driver, central_j, is_impurity=True, block=block);
        print("<(S1+S2)^2> = {:.6f}".format(S2_dmrg));
        if("phient" in params_dict.keys()):
            print("1 + cos(phi)= {:.6f}".format(1+np.cos(params_dict["phient"])));

        # mutual info
        minfo = tddmrg.mutual_info_wrapper(psi, eris_or_driver, central_j, True, block);
        print("MI[{:.0f},{:.0f}] = {:.6f} (max = {:.6f})".format(*central_j, minfo, np.log(2)));

    return;
                           
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
is_block = True; # whether to use block2 to do DMRG, or do FCI instead
# open parameters JSON
json_name = sys.argv[1];
try:
    try: 
        params = json.load(open(json_name+".txt"));
    except: 
        params = json.load(open(json_name));
        json_name = json_name[:-4];
    print(">>> Params = ",params);
except:
    raise Exception(json_name+" cannot be found");

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
if(params["sys_type"] in ["STT_RM"]): locspin = locspin*2;
special_cases = ["specialcase","BFM_first", "Bsd", "Bcentral", "Bsd_x","noFM","notspinpol"];
special_cases_flag = False;
for case in special_cases:
    if(case in params.keys()):print(">>> special case: ",case); special_cases_flag = True;
if(not special_cases_flag): assert(espin+locspin == myTwoSz);
del myNL, myNFM, myNR, myNe, myTwoSz, myNbuffer
#### Initialization
####
####
init_start = time.time();
    
# init ExprBuilder object with terms that are there for all times
if(params["sys_type"] in ["STT"]):
    H_driver, H_builder = tddmrg.H_STT_builder(params, is_block, scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder
elif(params["sys_type"] in ["STT_RM"]):
    H_driver, H_builder = tddmrg.H_STTRM_builder(params, is_block, scratch_dir=json_name, verbose=verbose); 

# add in t<0 terms
if(params["sys_type"] in ["STT"]):
    H_driver, H_mpo_initial = tddmrg.H_STT_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);
elif(params["sys_type"] in ["STT_RM"]):
    H_driver, H_mpo_initial = tddmrg.H_STTRM_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);
    
# gd state
gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                         bond_dim=params["bdim_0"][0] )
                  # NB the above method creates an MPS with attribute dot=2, meaning H_driver.dmrg method 
                  # (does not apply to H_driver.td_dmrg method) will do TWO-SITE sweep algorithm
gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,#tol=1e-24, # <------ !!!!!!
    bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], cutoff=params["cutoff"],
    iprint=0); # set to 2 to see Mmps
print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

init_end = time.time();
print(">>> Init compute time = "+str(init_end-init_start));

#### Observables
####
####
mytime=0;

# plot observables
check_observables(params, gdstate_mps_inst, H_driver, H_mpo_initial, mytime, is_block);
plot.snapshot_bench(gdstate_mps_inst, H_driver,
        params, json_name, mytime, is_block);

#### Time evolution
####
####
if(params["sys_type"] in ["STT"]):
    H_driver_dyn, H_builder_dyn = tddmrg.H_STT_builder(params, is_block, scratch_dir=json_name, verbose=verbose);
elif(params["sys_type"] in ["STT_RM"]):
    H_driver_dyn, H_builder_dyn = tddmrg.H_STTRM_builder(params, is_block, scratch_dir=json_name, verbose=verbose);

# construct matrix product operator
assert(is_block);
H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose);

# time evolve
tddmrg.kernel(params, H_driver_dyn, H_mpo_dyn,
                  gdstate_mps_inst,check_observables,tddmrg.plot.snapshot_bench,json_name,verbose=0) # set to 2 to see mmps


