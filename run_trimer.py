'''
Christian Bunker
M^2QM at UF
February 2024
'''

from transport import tdfci, tddmrg
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

def H_builder(params_dict, block, scratch_dir="tmp",verbose=0):
    '''
    Builds the parts of the STT Hamiltonian which apply at all t
    The physical params are contained in a .json file. They are all in eV.
    They are:
    tl (lead hopping), th (lead-impurity hopping), Vg (gate voltage on impurity),
    U (Coulomb repulsion on impurity), Vb (bias between left and right leads.
    Vb>0 means that left lead is higher chem potential than right, leading to
    rightward/positive current).

    NL (number sites in left lead),  NR (number of sites in right lead).
    There is always exactly 1 impurity, so Nsites=NL+1+NR
    NB this system is assumed half-filled, so Ne=Nsites.
    The total Sz of the electrons is always 0, so Ne_up=Ne_down=Ne//2
    NB this requires that Ne%2==0

    There is NO supersiting in this system

    Returns: a tuple of DMRGDriver, ExprBuilder objects
    '''

    # load data from json
    tl, Vg, U = params_dict["tl"], params_dict["Vg"], params_dict["U"];
    Nsites, Ne = params_dict["Nsites"], params_dict["Ne"];

    # classify site indices (spin not included)
    all_sites = np.array([j for j in range(Nsites)]);

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
        else: raise NotImplementedError;
        builder = driver.expr_builder();
        print("\n",40*"#","\nConstructed builder\n",40*"#","\n");
    else:   
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        h1e, g2e = np.zeros((Nspinorbs, Nspinorbs),dtype=float), np.zeros((Nspinorbs, Nspinorbs, Nspinorbs, Nspinorbs),dtype=float);


    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        if(block):
            builder.add_term("cd",[j,j+1],-tl); 
            builder.add_term("CD",[j,j+1],-tl);
            builder.add_term("cd",[j+1,j],-tl);
            builder.add_term("CD",[j+1,j],-tl);
        else:
            h1e[nloc*j+0,nloc*(j+1)+0] += -tl;
            h1e[nloc*(j+1)+0,nloc*j+0] += -tl;
            h1e[nloc*j+1,nloc*(j+1)+1] += -tl;
            h1e[nloc*(j+1)+1,nloc*j+1] += -tl;
    # cyclic hopping
    if (True and Nsites>2):
        j, jp1 = all_sites[-1], all_sites[0];
        if(block):
            builder.add_term("cd",[j,jp1],-tl); 
            builder.add_term("CD",[j,jp1],-tl);
            builder.add_term("cd",[jp1,j],-tl);
            builder.add_term("CD",[jp1,j],-tl);
        else:
            h1e[nloc*j+0,nloc*(jp1)+0] += -tl;
            h1e[nloc*(jp1)+0,nloc*j+0] += -tl;
            h1e[nloc*j+1,nloc*(jp1)+1] += -tl;
            h1e[nloc*(jp1)+1,nloc*j+1] += -tl;

    # Vg 
    for j in all_sites: #[0,2]:
        if(block):
            builder.add_term("cd",[j,j], Vg);
            builder.add_term("CD",[j,j], Vg);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vg;
            h1e[nloc*j+1,nloc*j+1] += Vg;

    # U
    for j in all_sites: #[1]:
        if(block):
            builder.add_term("cdCD",[j,j,j,j], U);
        else:
            g2e[nloc*j+0,nloc*j+0,nloc*j+1,nloc*j+1] += U;
            g2e[nloc*j+1,nloc*j+1,nloc*j+0,nloc*j+0] += U; # switch electron labels

    # tiny bit of spin polarization and spin mixing
    Bx, Bz = 1e-7, 1e-6;
    for j in all_sites[:1]:
        if(block):
            raise NotImplementedError;
        else:
            h1e[nloc*j+0,nloc*j+0] += Bz/2;
            h1e[nloc*j+1,nloc*j+1] += -Bz/2;
            h1e[nloc*j+0,nloc*j+1] += Bx/2;
            h1e[nloc*j+1,nloc*j+0] += Bx/2;

    if(block): return driver, builder;
    else: return h1e, g2e;

def get_energy_fci(h1e, g2e, nelec, nroots=1, tol = 1e-2, verbose=0):
    # convert from arrays to uhf instance
    mol_inst, uhf_inst = utils.arr_to_uhf(h1e, g2e, len(h1e), nelec, verbose = verbose);
    # fci solution
    E_fci, v_fci = utils.scf_FCI(mol_inst, uhf_inst, nroots);
    # truncate to gd state manifold
    if(nroots > 1):
        E_fci_orig = np.copy(E_fci);
        v_fci = v_fci[abs(E_fci - E_fci[0]) < tol];
        E_fci = E_fci[abs(E_fci - E_fci[0]) < tol];
        print(E_fci_orig,"\, --->",E_fci);
    else:
        E_fci = np.array([E_fci]);
    # ci object
    CI_list = [];
    for fcii in range(len(E_fci)):
        CI_list.append( tdfci.CIObject(v_fci[fcii], len(h1e), nelec));
    return CI_list, E_fci, uhf_inst;

def check_observables(params_dict,psi,eris_or_driver, none_or_mpo, the_time, block):
    '''
    Print update on selected observables
    '''
    print("\nTime = {:.2f}".format(the_time));
    if(not block): compute_func = tdfci.compute_obs;
    else: raise NotImplementedError

    # check gd state
    check_E_dmrg = compute_func(psi, eris_or_driver, none_or_mpo);
    print("Total energy = {:.8f}".format(check_E_dmrg));
    check_norm = psi.dot( psi)
    print("WF norm = {:.8f}".format(check_norm));

    # fermionic charge and spin in LL, Imp, RL
    for sitei in range(params_dict["Nsites"]):
        sz_mpo = tddmrg.get_sz(eris_or_driver, sitei, block);
        sz_val = compute_func(psi, sz_mpo, eris_or_driver);
        occ_mpo = tddmrg.get_occ(eris_or_driver, sitei, block);
        occ_val = compute_func(psi, occ_mpo, eris_or_driver);
        print("<n  j={:.0f} = {:.8f}".format(sitei, occ_val));
        print("<sz j={:.0f} = {:.8f}".format(sitei, sz_val));
                          
##################################################################################
#### run code

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 8, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name)); print(">>> Params = ",params);
block = False;

# total num electrons. For fci, should all be input as spin up
myNsites, myNe = params["Nsites"], params["Ne"];
nloc = 2; # spin dofs
init_start = time.time();
    
# build H, get gd state
if(block):
    pass;
else:
    H_mpo_initial = None;
    H_1e, H_2e = H_builder(params, block, scratch_dir=json_name, verbose=verbose);
    print("H_1e =\n", H_1e); 
    gdstate_psi, gdstate_E, gdstate_scf_inst = get_energy_fci(H_1e, H_2e, (myNe, 0), nroots=20, verbose=0);
    print(type(gdstate_psi))
    print(type(gdstate_E))
    print(type(gdstate_scf_inst))
    H_eris = tdfci.ERIs(H_1e, H_2e, gdstate_scf_inst.mo_coeff);
    eris_or_driver = H_eris;
init_end = time.time();
print(">>> Init compute time = "+str(init_end-init_start));

# plot observables
mytime=0;
for statei in range(len(gdstate_E)):
    print("\nGround state energy (FCI) = {:.8f}".format(gdstate_E[statei]))
    check_observables(params, gdstate_psi[statei], eris_or_driver, H_mpo_initial, mytime, block);




