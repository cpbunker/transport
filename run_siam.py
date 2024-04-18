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
        sz_mpo = tddmrg.get_sz(eris_or_driver, sitei, block);
        sz_val = tddmrg.compute_obs(psi, sz_mpo, eris_or_driver);
        occ_mpo = tddmrg.get_occ(eris_or_driver, sitei, block);
        occ_val = tddmrg.compute_obs(psi, occ_mpo, eris_or_driver);
        print("<n  j={:.0f} = {:.6f}".format(sitei, occ_val));
        print("<sz j={:.0f} = {:.6f}".format(sitei, sz_val));

    # current through Imp
    Jimp_val = tddmrg.conductance_wrapper(psi, eris_or_driver, Impsite, block);
    Jimp_val *= np.pi*params_dict["th"]/params_dict["Vb"];
    print("<J  j={:.0f}>/Vb = {:.6f}".format(Impsite, Jimp_val));
                           
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name)); print(">>> Params = ",params);
is_block = True;
if("tdfci" in params.keys()):
    if(params["tdfci"]==1): is_block=False;

# unpacking
myNL, myNR = params["NL"], params["NR"];
myNe = myNL+1+myNR; # total num electrons. For fci, should all be input as spin up
nloc = 2; # spin dofs

# checks
pass;

#### Initialization
####
####
init_start = time.time();
    
# init ExprBuilder object with terms that are there for all times
H_driver, H_builder = tddmrg.H_SIAM_builder(params, is_block, scratch_dir=json_name, verbose=verbose); # returns DMRGDriver, ExprBuilder

# add in t<0 terms
H_driver, H_mpo_initial = tddmrg.H_SIAM_polarizer(params, (H_driver,H_builder), is_block, verbose=verbose);
    
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
    print("H_1e = ");
    print(H_1e[:nloc*myNL,:nloc*myNL]);
    print(H_1e[nloc*(myNL-1):nloc*(myNL+1+1),nloc*(myNL-1):nloc*(myNL+1+1)]);
    print(H_1e[nloc*(myNL+1):,nloc*(myNL+1):]); 

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
H_driver_dyn, H_builder_dyn = tddmrg.H_SIAM_builder(params, is_block, scratch_dir=json_name, verbose=verbose);
if(is_block):
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=verbose);
    tddmrg.kernel(params, H_driver_dyn, H_mpo_dyn,gdstate_mps_inst,
                  check_observables,json_name, verbose=2) # set to 2 to see mmps
else:
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

