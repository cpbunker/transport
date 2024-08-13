'''
Christian Bunker
M^2QM at UF
August 2024

Use density matrix renormalization group (DMRG) code from Huanchen Zhai (block2)
to study a 1D wire with itinerant fermions
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

    # central site spins
    central_sites = np.arange(0,params_dict["Ncent"]);
    all_sites = np.arange(0,params_dict["Nsites"]);
    for j in all_sites:
        occ_mpo = tddmrg.get_occ(eris_or_driver, j, block);
        occ_dmrg = tddmrg.compute_obs(psi, occ_mpo, eris_or_driver);
        print("<occ j={:.0f}> = {:.6f}".format(j, occ_dmrg));
        sz_mpo = tddmrg.get_sz(eris_or_driver, j, block);
        sz_dmrg = tddmrg.compute_obs(psi, sz_mpo, eris_or_driver);
        print("<sz  j={:.0f}> = {:.6f}".format(j, sz_dmrg));
    # (S1+S2)^2
    S2_mpo = tddmrg.get_S2(eris_or_driver, central_sites, fermion=True, block=block);
    S2_dmrg = tddmrg.compute_obs(psi, S2_mpo, eris_or_driver);
    print("<(S1+S2)^2> = {:.6f}".format(S2_dmrg));

##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name)); print(">>> Params = ",params);
is_block = True;

# unpacking
myNe, myTwoSz =  params["Ne"], params["TwoSz"];

# checks
espin = myNe*np.sign(params["Be"]);
special_cases = ["Be_first"];
special_cases_flag = False;
for case in special_cases:
    if(case in params.keys()):print(">>> special case: ",case); special_cases_flag = True;
if(not special_cases_flag): assert(espin == myTwoSz);

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

# orbital entanglement
if(True):
    odm1 = H_driver.get_orbital_entropies(gdstate_mps_inst, orb_type=1);
    odm2 = H_driver.get_orbital_entropies(gdstate_mps_inst, orb_type=2);
    minfo = 0.5 * (odm1[:, None] + odm1[None, :] - odm2) * (1 - np.identity(len(odm1))); # Vedral Eq (?)
    print(minfo);

    assert False
    # have to change sym type to core.SymmetryTypes.SZ in driver constructor
    int_matrix = H_driver.get_orbital_interaction_matrix(gdstate_mps_inst);
    fig, ax = plt.subplots()
    ax.matshow(int_matrix, cmap='ocean_r')
    plt.show()
    assert False

