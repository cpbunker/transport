'''
Christian Bunker
M^2QM at UF
February 2024
'''

from transport import tdfci, tddmrg
from transport.tdfci import utils

import numpy as np
import matplotlib.pyplot as plt
import scipy

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
        from pyblock2.driver import core
        TwoSz = params_dict["TwoSz"];
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
            
    # last-> first hopping to complete ring
    if(params_dict["is_ring"] and Nsites>2):
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
            
    # triangulating hopping (see Greiner Fig 1a)
    if(params_dict["is_triangular"] and Nsites==4):
        tp = params_dict["tp"];
        j, jp1 = all_sites[1], all_sites[3];
        if(block):
            builder.add_term("cd",[j,jp1],-tp); 
            builder.add_term("CD",[j,jp1],-tp);
            builder.add_term("cd",[jp1,j],-tp);
            builder.add_term("CD",[jp1,j],-tp);
        else:
            h1e[nloc*j+0,nloc*(jp1)+0] += -tp;
            h1e[nloc*(jp1)+0,nloc*j+0] += -tp;
            h1e[nloc*j+1,nloc*(jp1)+1] += -tp;
            h1e[nloc*(jp1)+1,nloc*j+1] += -tp;

    # Vg 
    for j in all_sites: 
        if(block):
            builder.add_term("cd",[j,j], Vg);
            builder.add_term("CD",[j,j], Vg);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vg;
            h1e[nloc*j+1,nloc*j+1] += Vg;

    # U
    for j in all_sites: 
        if(block):
            builder.add_term("cdCD",[j,j,j,j], U);
        else:
            g2e[nloc*j+0,nloc*j+0,nloc*j+1,nloc*j+1] += U;
            g2e[nloc*j+1,nloc*j+1,nloc*j+0,nloc*j+0] += U; # switch electron labels

    # tiny bit of spin polarization and spin mixing
    # when present, breaks fourfold degeneracy of E levels, but ensures z is special direction so total sz is \pm 0.5
    Bx, Bz = params_dict["Bx"], params_dict["Bz"];
    for j in all_sites:
        if(block):
            builder.add_term("cd",[j,j], Bz/2);
            builder.add_term("CD",[j,j],-Bz/2);
            assert(abs(Bx)<1e-12);
        else:
            h1e[nloc*j+0,nloc*j+0] += Bz/2;
            h1e[nloc*j+1,nloc*j+1] += -Bz/2;
            h1e[nloc*j+0,nloc*j+1] += Bx/2;
            h1e[nloc*j+1,nloc*j+0] += Bx/2;

    if(block):
        mpo_from_builder = driver.get_mpo(builder.finalize());
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;

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
        print(E_fci_orig,"\n--->",E_fci);
    else:
        E_fci = np.array([E_fci]);
    # ci object
    CI_list = [];
    for fcii in range(len(E_fci)):
        CI_list.append( tdfci.CIObject(v_fci[fcii], len(h1e), nelec));
    return CI_list, E_fci, uhf_inst;

                        
##################################################################################
#### run code
if(__name__ == "__main__"):
    
    # top level
    verbose = 2; assert verbose in [1,2,3];
    np.set_printoptions(precision = 6, suppress = True);
    json_name = sys.argv[1];
    params = json.load(open(json_name)); print(">>> Params = ",params);
    is_block = True;
    if("tdfci" in params.keys()):
        if(params["tdfci"]==1): is_block=False;

    # total num electrons. For fci, should all be input as spin up
    myNsites, myNe = params["Nsites"], params["Ne"];
    nloc = 2; # spin dofs
    loop_start = time.time();

    # iter over U/t
    Uvals = np.linspace(0.0,params["U"],29);
    Evals = np.zeros((len(Uvals),),dtype=float);
    Evals_inf = np.zeros((len(Uvals),),dtype=float); # Lieb and Wu expression
    S2vals = np.zeros((len(Uvals),),dtype=float);
    S2vals_inf = np.zeros((len(Uvals),),dtype=float);
    for Uvali in range(len(Uvals)):
        # override json
        params_over = params.copy();
        params_over["U"] = Uvals[Uvali];
        
        # build H, get gd state
        if(is_block):
            gdstate_mps_inst = H_driver.get_random_mps(tag="gdstate",nroots=1,
                                 bond_dim=params["bdim_0"][0] )
            gdstate_E_dmrg = H_driver.dmrg(H_mpo_initial, gdstate_mps_inst,#tol=1e-24, # <------ !!!!!!
                bond_dims=params["bdim_0"], noises=params["noises"], n_sweeps=params["dmrg_sweeps"], 
                cutoff=params["cutoff"], iprint=2); # set to 2 to see Mmps
            eris_or_driver = H_driver;
            Evals[Uvali] = gdstate_E_dmrg/params_over["Nsites"];
            Sz2_mpo = tddmrg.get_sz2(eris_or_driver, 0, is_block);
            S2vals[Uvali] = 3*tdddmrg.compute_obs(gdstate_psi[0], Sz2_mpo, eris_or_driver);

        else:
            H_mpo_initial = None;
            H_1e, H_2e = H_builder(params_over, is_block, scratch_dir=json_name, verbose=verbose);
            if(Uvali==0): print("H_1e =\n", H_1e); 
            gdstate_psi, gdstate_E, gdstate_scf_inst = get_energy_fci(H_1e, H_2e,
                                    (myNe, 0), nroots=20, tol=1e6, verbose=0);
            H_eris = tdfci.ERIs(H_1e, H_2e, gdstate_scf_inst.mo_coeff);
            eris_or_driver = H_eris;
            Evals[Uvali] = gdstate_E[0]/params_over["Nsites"];
            Sz2_mpo = tddmrg.get_sz2(eris_or_driver, 0, is_block);
            S2vals[Uvali] = 3*tddmrg.compute_obs(gdstate_psi, Sz2_mpo, eris_or_driver);

        # exact soln
        # infinite chain closed form numerical soln
        Ja_func = scipy.special.jv
        omega_crossover, nomega = 10.0,2000
        omega_mesh = np.linspace(1e-12,omega_crossover,nomega//2)
        omega_mesh = np.append(omega_mesh, np.linspace(omega_crossover, 1e1*omega_crossover,nomega//2));
        integrand = Ja_func(0,omega_mesh)*Ja_func(1,omega_mesh)/(omega_mesh*(1+np.exp(2*omega_mesh*params_over["U"]/(4*params_over["tl"]))))
        scipy_integ = np.trapz
        Evals_inf[Uvali] = -4*scipy_integ(integrand, x=omega_mesh);
        #fig, ax = plt.subplots()
        #ax.plot(omega_mesh, integrand)
        #plt.show()
        print("N = {:.0f}, U = {:.4f}, U/2t = {:.4f}, E = {:.4f}, E_inf = {:.4f}".format(params_over["Nsites"], params_over["U"], params_over["U"]/(2*params_over["tl"]), Evals[Uvali], Evals_inf[Uvali]))

    # plot E
    fig, axes = plt.subplots(2, sharex=True);
    axes[0].plot(Uvals/(2*params_over["tl"]), Evals_inf, color="black", label = "Exact");
    axes[0].scatter(Uvals/(2*params_over["tl"]), Evals, marker='o',s=100, facecolors='none',edgecolors='purple', label = "DMRG");
    axes[0].set_ylabel("Energy/site");

    # plot S2
    S2vals_inf = 3/4 - (3/2)*np.gradient(Evals_inf,Uvals)
    axes[1].plot(Uvals/(2*params_over["tl"]), S2vals_inf, color="black", label="Exact");
    axes[1].scatter(Uvals/(2*params_over["tl"]), S2vals, marker='o',s=100, facecolors='none',edgecolors='purple', label = "DMRG");
    S2_lims = [3/8,3/4];
    for lim in S2_lims: axes[1].axhline(lim,color="grey",linestyle="dashed");
    axes[1].set_yticks(S2_lims);
    axes[1].set_ylabel("$\\langle S_p^2 \\rangle$")
    loop_end = time.time();
    print(">>> Loop compute time = "+str(loop_end-loop_start));

    # format
    axes[0].legend();
    axes[0].set_title("$N = N_e =${:.0f}, $\\nu_p = ${:.2f}".format(myNsites, 0.0));
    axes[-1].set_xlabel("$U/2t$");
    plt.show();
