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


def get_energy_fci(h1e, g2e, nelec, nroots=1, verbose=0):
    # convert from arrays to uhf instance
    mol_inst, uhf_inst = utils.arr_to_uhf(h1e, g2e, len(h1e), nelec, verbose = verbose);
    # fci solution
    E_fci, v_fci = utils.scf_FCI(mol_inst, uhf_inst, nroots);
    if(nroots>1): E_fci, v_fci = E_fci[0], v_fci[0];
    # ci object
    CI_inst = tdfci.CIObject(v_fci, len(h1e), nelec);
    return CI_inst, E_fci, uhf_inst;

def check_observables(params_dict,psi,eris_or_driver, none_or_mpo, the_time, block):
    '''
    Print update on selected observables
    '''
    assert(isinstance(block, bool));
    print("\nTime = {:.2f}".format(the_time));
    if(not block): return; # hacky
    RMflag = False;
    if("RM" in params_dict["sys_type"]):
        assert("v" in params_dict.keys());
        RMflag = True;
        RMdofs = 2;

    # check gd state
    check_E_dmrg = tddmrg.compute_obs(psi, none_or_mpo, eris_or_driver);
    print("Total energy = {:.6f}".format(check_E_dmrg));
    impo = eris_or_driver.get_identity_mpo()
    check_norm = eris_or_driver.expectation(psi, impo, psi)
    print("WF norm = {:.6f}".format(check_norm));

    # observables
    Impi = params_dict["NL"]
    if(params_dict["sys_type"] in ["SIETS_RM"]): # impurity spin
        for_spin = np.arange(Impi, Impi+params_dict["NFM"]);
        for sitei in for_spin: 
            if(RMflag): mus = [RMdofs*sitei, RMdofs*sitei+1];
            else: mus = [sitei];
            for mu in mus:
                Sd_mpo = tddmrg.get_Sd_mu(eris_or_driver, mu, block);
                Sd_val = tddmrg.compute_obs(psi, Sd_mpo, eris_or_driver);
                print("<Sz d={:.0f} = {:.6f}".format(mu, Sd_val));
    else: # fermionic charge and spin in LL, Imp, RL
        for_spin = [0, Impi, Impi+params_dict["NR"]];
        for sitei in for_spin:
            if(RMflag): mus = [RMdofs*sitei, RMdofs*sitei+1];
            else: mus = [sitei];
            for mu in mus:
                sz_mpo = tddmrg.get_sz(eris_or_driver, mu, block);
                sz_val = tddmrg.compute_obs(psi, sz_mpo, eris_or_driver);
                occ_mpo = tddmrg.get_occ(eris_or_driver, mu, block);
                occ_val = tddmrg.compute_obs(psi, occ_mpo, eris_or_driver);
                print("<n  site={:.0f} = {:.6f}".format(mu, occ_val));
                print("<sz site={:.0f} = {:.6f}".format(mu, sz_val));
                
    if(len(np.arange(Impi, Impi+params_dict["NFM"]))==1): # SR has 1 RM block -> 2 impurities
        # (S1+S2)^2
        S2_dmrg = tddmrg.S2_wrapper(psi, eris_or_driver, [RMdofs*Impi,RMdofs*Impi+1], is_impurity=True, block=block);
        print("<(S1+S2)^2> = {:.6f}".format(S2_dmrg));

        # mutual info
        minfo = tddmrg.mutual_info_wrapper(psi, eris_or_driver, [RMdofs*Impi,RMdofs*Impi+1], True, block);
        print("MI[{:.0f},{:.0f}] = {:.6f} (max = {:.6f})".format(*[RMdofs*Impi,RMdofs*Impi+1], minfo, np.log(2)));

    # conductancethrough Imp
    if(RMflag): sites_for_G = [2*Impi, 2*Impi+1];
    else: sites_for_G = [Impi];
    for i in sites_for_G:
        Gval = tddmrg.conductance_wrapper(psi, eris_or_driver, i, block);
        Gval *= np.pi*params_dict["th"]/params_dict["Vb"];
        print("<G  site={:.0f}> = {:.6f}".format(i, Gval));
                           
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
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
is_block = True;
if("tdfci" in params.keys()):
    if(params["tdfci"]==1): is_block=False;
    
# Rice-Mele model?
is_RM = False;
if("RM" in params["sys_type"]):
    assert("v" in params.keys());
    is_RM = True;

# unpacking
nloc = 2; # spin dofs
myNL, myNR = params["NL"], params["NR"];
if("NFM" in params.keys() and (not is_RM)): raise NotImplementedError;
myNe = myNL+1+myNR; # total num electrons. For fci, should all be input as spin up
if("Ne_override" in params.keys()):
    assert("Ne" not in params.keys());
    myNe = params["Ne_override"];

# checks
pass;

#### Initialization
####
####
init_start = time.time();
    
# init ExprBuilder object with terms that are there for all times
if(is_RM):
    H_driver, H_builder = tddmrg.H_RM_builder(params, is_block, scratch_dir=json_name, verbose=verbose); 
else:
    H_driver, H_builder = tddmrg.H_SIAM_builder(params, is_block, scratch_dir=json_name, verbose=verbose); 
    # if is_block: returns DMRGDriver, ExprBuilder
    # else: returns h1e, g2e

# add in t<0 terms
if(is_RM):
    H_driver, H_mpo_initial = tddmrg.H_RM_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);
else:
    H_driver, H_mpo_initial = tddmrg.H_SIAM_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);
    # if is_block: returns DMRGDriver, MPO for t<0 hamiltonian
    # else: returns h1e, g2e
    
# gd state
if(is_block):
    gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                         bond_dim=params["bdim_0"][0] )
    gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,#tol=1e-24, # <------ !!!!!!
        bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], 
        cutoff=params["cutoff"], iprint=2); # set to 2 to see Mmps
    eris_or_driver = H_driver;
    print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));
else:
    H_1e, H_2e = np.copy(H_driver), np.copy(H_mpo_initial); # H_SIAM_polarizer output with block=False
    if(is_RM): block2site=2;
    else: block2site = 1;
    print("H_1e = ");
    print(H_1e[:nloc*myNL*block2site,:nloc*myNL*block2site]);
    print(H_1e[nloc*(myNL-1)*block2site:nloc*(myNL+1+1)*block2site,nloc*(myNL-1)*block2site:nloc*(myNL+1+1)*block2site]);
    print(H_1e[nloc*(myNL+1)*block2site:,nloc*(myNL+1)*block2site:]); 
    assert False;

    # gd state
    gdstate_mps_inst, gdstate_E, gdstate_scf_inst = get_energy_fci(H_1e, H_2e, (myNe, 0), nroots=1, verbose=0);
    H_eris = tdfci.ERIs(H_1e, H_2e, gdstate_scf_inst.mo_coeff);
    eris_or_driver = H_eris;
    print("Ground state energy (FCI) = {:.6f}".format(gdstate_E));


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
if(is_RM):
    H_driver_dyn, H_builder_dyn = tddmrg.H_RM_builder(params, is_block, scratch_dir=json_name, verbose=verbose);
else:
    H_driver_dyn, H_builder_dyn = tddmrg.H_SIAM_builder(params, is_block, scratch_dir=json_name, verbose=verbose);
if(is_block):
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose);
    tddmrg.kernel(params, H_driver_dyn, H_mpo_dyn,gdstate_mps_inst,
                  check_observables,tddmrg.plot.snapshot_bench,json_name, verbose=2) # set to 2 to see mmps

else: # td-FCI !
    time_step, time_stop = params["time_step"], params["tupdate"];
    H_1e_dyn, H_2e_dyn = np.copy(H_driver_dyn), np.copy(H_builder_dyn); # output with block=True
    print("H_1e_dyn = ");print(H_1e_dyn[:nloc*myNL,:nloc*myNL]);print(H_1e_dyn[nloc*(myNL-1):nloc*(myNL+1+1),nloc*(myNL-1):nloc*(myNL+1+1)]);print(H_1e_dyn[nloc*(myNL+1):,nloc*(myNL+1):]); 

    # repeated time evols
    Nupdates = params["Nupdates"];
    H_eris_dyn = tdfci.ERIs(H_1e_dyn, H_2e_dyn, gdstate_scf_inst.mo_coeff);
    t_ci_inst = gdstate_mps_inst; del gdstate_mps_inst;
    for update in range(1,Nupdates+1):
        mytime += time_stop;
        t_ci_inst = tdfci.kernel(t_ci_inst, H_eris_dyn, time_stop, time_step);
        check_observables(params, t_ci_inst, H_driver, H_mpo_initial, mytime, is_block);
        plot.snapshot_bench(t_ci_inst, eris_or_driver, params, json_name, mytime, is_block);

