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

from hub_ring import H_builder, get_energy_fci, check_observables
                          
##################################################################################
#### run code
if(__name__ == "__main__"):
    
    # top level
    verbose = 2; assert verbose in [1,2,3];
    np.set_printoptions(precision = 6, suppress = True);
    json_name = sys.argv[1];
    params = json.load(open(json_name)); print(">>> Params = ",params);
    is_block = False;

    # total num electrons. For fci, should all be input as spin up
    myNsites, myNe = params["Nsites"], params["Ne"];
    nloc = 2; # spin dofs
    mynroots = 20;
    init_start = time.time();

    # iter over U
    Uvals = np.linspace(0,10,11);
    Evals = np.zeros((len(Uvals),mynroots),dtype=float);
    for Uvali in range(len(Uvals)):
        # override json
        params_over = params.copy();
        params_over["U"] = Uvals[Uvali];
        
        # build H, get gd state
        if(is_block):
            pass;
        else:
            H_mpo_initial = None;
            H_1e, H_2e = H_builder(params_over, is_block, scratch_dir=json_name, verbose=verbose);
            print("H_1e =\n", H_1e); 
            gdstate_psi, gdstate_E, gdstate_scf_inst = get_energy_fci(H_1e, H_2e,
                                    (myNe, 0), nroots=mynroots, tol=1e6, verbose=0);
            H_eris = tdfci.ERIs(H_1e, H_2e, gdstate_scf_inst.mo_coeff);
            eris_or_driver = H_eris;
        Evals[Uvali] = gdstate_E[:];
        init_end = time.time();
        print(">>> Init compute time = "+str(init_end-init_start));

        #observables
        mytime=0;
        for statei in range(len(gdstate_E)):
            print("\nGround state energy (FCI) = {:.8f}".format(gdstate_E[statei]))
            #check_observables(params_over, gdstate_psi[statei], eris_or_driver, H_mpo_initial, mytime, is_block);

    # plot E
    fig, ax = plt.subplots();
    for rooti in range(mynroots):
        ax.plot(Uvals, Evals[:,rooti]);
    ax.set_xlabel("$U/t$");
    ax.set_ylabel("$E_{trimer}$");
    plt.show();
        
