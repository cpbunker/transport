'''
Christian Bunker
M^2QM at UF
October 2023

Use density matrix renormalization group (DMRG) code (block2) from Huanchen Zhai
(Chan group, Caltech) to study molecular spin qubit (MSQ) systems
'''

from transport import tdfci
from transport.tdfci import utils
from pyblock2.driver import core
from pyblock3.block2.io import MPSTools, MPOTools
import numpy as np
import time
    
##########################################################################################################
#### driver of time propagation

def kernel(params_dict, driver_inst, mpo_inst, psi, check_func, plot_func, save_name, verbose=0):
    '''
    '''
    #assert(params_dict["te_type"]=="tdvp");
    print("\n\nSTART TIME EVOLUTION (te_type = "+params_dict["te_type"]+")\n\n","*"*50,"\n\n")
    print("\t driver.mpi = ",driver_inst.mpi);
    print("\t global threads = {:.0f}".format(driver_inst.bw.b.Global.threading.n_threads_global));
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
        evol_time0 = time.time()

        # time evol
        krylov_subspace = 20; # default
        if(params_dict["te_type"] == "tdvp"): krylov_subspace = 40;
        tevol_mps_inst = driver_inst.td_dmrg(mpo_inst, tevol_mps_inst, 
                delta_t=complex(0,time_step), target_t=complex(0,time_update),
                bond_dims=params_dict["bdim_t"], cutoff=params_dict["cutoff"], te_type=params_dict["te_type"],
                krylov_subspace_size=krylov_subspace,final_mps_tag=str(int(100*total_time)), iprint=the_verbose);
        evol_time1 = time.time();
        print("\n>>>> Evol CPU time = {:.1f} min\n".format((evol_time1-evol_time0)/60));

        # observables
        check_time0 = time.time();
        check_func(params_dict,tevol_mps_inst,driver_inst,mpo_inst,total_time, True);

        # plot and/or save observables
        if(plot_func != None): plot_func(tevol_mps_inst, driver_inst, params_dict, save_name, total_time,True);
        
        check_time1 = time.time();
        print("\n>>>> Observables CPU time = {:.1f} min\n".format((check_time1-check_time0)/60));
        
    return;

################################################################################
#### observables

def compute_obs(psi, mpo_inst, driver, conj=False):
    '''
    Compute expectation value of observable repped by given operator from the wf
    The wf psi must be a matrix product state, and the operator an MPO
    '''

    impo = driver.get_identity_mpo();
    return driver.expectation(psi, mpo_inst, psi)/driver.expectation(psi, impo, psi);

def get_occ(eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing the occupancy of site whichsite
    '''
    if(block): builder = eris_or_driver.expr_builder()
    else:
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite],1.0);
        builder.add_term("CD",[whichsite,whichsite],1.0);
    else:
        h1e[nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        h1e[nloc*whichsite+1,nloc*whichsite+1] += 1.0;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_Om(m, is_impurity):
    '''
    Theory: White 2006 (https://doi.org/10.1016/j.chemphys.2005.10.018) Table II
    '''
    if(is_impurity):
        Om_dict = {
            6: (["MP"],[1]),
            7: (["M"],[1]),
            10:(["P"],[1]),
            11:(["PM"],[1]),
            };
    else:
        Om_dict = { # each m -> tuple of [expressions] [coefficients]
            1: (["", "cd", "CD", "cdCD"],[1,-1,-1,1]),
            2: (["D", "cdD"], [1,-1]),
            3: (["d", "CDd"], [1,-1]),
            4: (["Dd"],[-1]),
            5: (["C", "cdC"], [1,-1]),
            6: (["CD", "cdCD"], [1,-1]),
            7: (["Cd"], [1]),
            8: (["CDd"], [-1]),
            9: (["c", "CDc"], [1,-1]),
            10:(["Dc"],[-1]),
            11:(["cd", "cdCD"], [1,-1]),
            12:(["cdD"], [1]),
            13:(["Cc"], [1]),
            14:(["CDc"],[-1]),
            15:(["cdC"], [1]),
            16:(["cdCD"],[1])
            };

    return Om_dict[m];

def oneorb_entropies_wrapper(psi, eris_or_driver, whichsites, sites_are_imps, block):
    '''
    Compute the one-orbital reduced density matrix and extract the von Neumann entropy, for all *fermionic* orbitals

    Theory: White 2006 (https://doi.org/10.1016/j.chemphys.2005.10.018) Table II
    '''
    if(core.SymmetryTypes.SZ not in eris_or_driver.bw.symm_type): raise TypeError;

    # return value
    ents = np.full((eris_or_driver.n_sites,),np.inf,dtype=float);

    # identify the sites in whichsites and whether or not they are classified as singly-occupied 'impurity sites' rather than molecular orbitals
    site_mask = np.array([True if site in whichsites else False for site in range(eris_or_driver.n_sites)]);
    sites_are_imps_expanded = np.zeros_like(site_mask);
    for listindex in range(len(whichsites)):
        sites_are_imps_expanded[whichsites[listindex]] = sites_are_imps[listindex];

    # iter over sites
    for sitei in np.arange(eris_or_driver.n_sites)[site_mask]: 

        # one-orbital reduced density matrix for this site
        # O(m) operators chosen come from Reiher 2013 (https://doi.org/10.1021/ct400247p) Table 2
        if(sites_are_imps_expanded[sitei]):
            oneorb_rdm = np.zeros((2,2),dtype=float);
            oneorb_Oms = [6,11];
        else: # molecular orbitals
            oneorb_rdm = np.zeros((4,4),dtype=float);
            oneorb_Oms = [1,6,11,16]; 

        # one orbital reduced density matrix is *diagonal*
        for diagi in range(len(oneorb_rdm)):

            # diagonal one-orb RDM elements are expectation values of fermionic operators
            # to get MPO for these expectation values, need to know expression, site, coefficient
            # get_Om gives list of expressions and coeffcients
            expressions, coefs = get_Om(oneorb_Oms[diagi], sites_are_imps_expanded[sitei]);
            expect_builder = eris_or_driver.expr_builder();
            for termi in range(len(expressions)):
                expect_builder.add_term(expressions[termi], [sitei]*len(expressions[termi]), coefs[termi]);
            expect_mpo = eris_or_driver.get_mpo(expect_builder.finalize(adjust_order=True, fermionic_ops="cdCD"));
            expect_val = compute_obs(psi, expect_mpo, eris_or_driver);

            # clean up numerical instabilities 
            if(abs(np.imag(expect_val)) > 1e-10): raise ValueError; # diag values of a density matrix are always real!
            else: expect_val = np.real(expect_val);
            if(expect_val<0.0): # clean up small <0 diag values that lead log -> nan
                if(abs(expect_val)<1e-10): expect_val = abs(expect_val); # eliminate log problems
                else: print("rho[",diagi,diagi,"] = ", expect_val); raise ValueError;

            oneorb_rdm[diagi,diagi] = expect_val;

        # get non Neumann entropy from one orb rdm
        ents[sitei] = np.trace( -oneorb_rdm*np.log(oneorb_rdm));

    return ents;

def twoorb_entropies_wrapper(psi, eris_or_driver, whichsites, block):
    '''
    Compute the two-orbital reduced density matrix and extract the von Neumann entropy
    '''
    if(core.SymmetryTypes.SZ not in eris_or_driver.bw.symm_type): raise TypeError;

    # return value
    ents = np.full((eris_or_driver.n_sites,eris_or_driver.n_sites), np.inf, dtype=float); # 

    # O(m) operators chosen come from Reiher 2013 (https://doi.org/10.1021/ct400247p) Table 3
    twoorb_Oms = [[[ 1, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 1, 6],[ 2, 5],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 5, 2],[ 6, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 1,11],[ 3, 9],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 9, 3],[11, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]], 
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 6, 6],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 1,16],[ 2,15],[ 3,14],[ 4,13],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 5,12],[ 6,11],[ 7,10],[ 8, 9],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 9, 8],[10, 7],[11, 6],[12, 5],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[13, 4],[14, 3],[15, 2],[16, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[11,11],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 6,16],[ 8,14],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[14, 8],[16, 6],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[11,16],[12,15],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[15,12],[16,11],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[16,16]]];
    twoorb_Oms = np.array(twoorb_Oms, dtype=int);
    twoorb_Oms_flags = np.array([[True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,False,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,False,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,False,False,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,False,False,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,False,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,False,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ]]);

    # we get whichsites paired withh ALL OTHERS but no pairs where neither are whichsites
    site_mask = np.array([True if site in whichsites else False for site in range(eris_or_driver.n_sites)]);
    for sitei in range(eris_or_driver.n_sites):
        if(site_mask[sitei]): jsites_iter = np.append(np.arange(0,sitei), np.arange(sitei+1,eris_or_driver.n_sites));
        else: jsites_iter = []; # skip these: sitejs with mask=True will be recovered when we do ents[sitei, sitej] = ents[sitej, sitei]
        for sitej in jsites_iter:

            # two-orbital reduced density matrix for (sitei, sitej) pair
            twoorb_rdm = np.zeros((len(twoorb_Oms),len(twoorb_Oms)),dtype=complex); # reduced density matrix elements are in general complex

            # expressions from Reiher 2013 (https://doi.org/10.1021/ct400247p) Table 3
            for rowi in range(len(twoorb_Oms)):
                #for coli in range(rowi+1): # only do lower half triangle, then use symmetry
                for coli in range(len(twoorb_Oms)):
                    Om_On = twoorb_Oms[rowi, coli];
                    if(not np.any(Om_On)): pass; # this matrix element is always zero
                    else: # Om_On is a tuple of integers telling which Om, On operator expressions to use
                        if(twoorb_Oms_flags[rowi,coli] == True):
                            Om_exprs, Om_coefs = get_Om(Om_On[0]);
                            On_exprs, On_coefs = get_Om(Om_On[1]);
                        else: # instead of taking <O(m)O(n)> we use the complex conjugate of the transposed matrix element
                            Om_On = twoorb_Oms[coli,rowi]; #<--- Here is where we call for transposed matrix element
                            assert(twoorb_Oms_flags[coli,rowi]==True); # otherwise neither is correct!
                            Om_exprs, Om_coefs = get_Om(Om_On[0]);
                            On_exprs, On_coefs = get_Om(Om_On[1]);

                        # distribute two lists of expressions into one
                        combo_exprs = []
                        combo_coefs = [];
                        combo_sites = [];
                        for Omi in range(len(Om_exprs)):
                            for Oni in range(len(On_exprs)):
                                combo_exprs.append(Om_exprs[Omi] + On_exprs[Oni]);
                                combo_coefs.append(Om_coefs[Omi] * On_coefs[Oni]);
                                combo_sites.append( [sitei]*len(Om_exprs[Omi]) + [sitej]*len(On_exprs[Oni]));

                        # expectation value
                        expect_builder = eris_or_driver.expr_builder();
                        for termi in range(len(combo_exprs)):
                            #print(termi, combo_exprs[termi], combo_sites[termi], combo_coefs[termi]);
                            expect_builder.add_term(combo_exprs[termi], combo_sites[termi], combo_coefs[termi]);
                        expect_mpo = eris_or_driver.get_mpo(expect_builder.finalize(adjust_order=True, fermionic_ops="cdCD"));
                        expect_val = compute_obs(psi, expect_mpo, eris_or_driver);
                        if(twoorb_Oms_flags[rowi,coli] == False):
                            twoorb_rdm[rowi, coli] = np.conj(expect_val); #<--- Here is where we take conjugate of transposed matrix element
                        else:
                            twoorb_rdm[rowi, coli] = expect_val;

            # clean up numerical instabilities
            for rowi in range(len(twoorb_Oms)):
                for coli in range(len(twoorb_Oms)):
                    Om_On = twoorb_Oms[rowi, coli];
                    if(not np.any(Om_On)): pass; # this matrix element is always zero
                    else:
                        # diagonal elements must be real
                        if(rowi == coli):
                            if(abs(np.imag(twoorb_rdm[rowi, coli])) > 1e-10): 
                                print("\n########################### rho_ij diag not real! #############################\n");
                                print("<{:.0f}|rho_ij|{:.0f}> --> O({:.0f}) O({:.0f})".format(rowi, coli, *Om_On));
                                print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(rowi, coli, np.real(twoorb_rdm[rowi, coli]), np.imag(twoorb_rdm[rowi, coli]))); raise ValueError;
                        # upper half triangle elements must be complex conjugate of lower half
                        if(coli>rowi):
                            if(abs(np.real(twoorb_rdm[rowi,coli]) - np.real(twoorb_rdm[coli,rowi]))>1e-10 or abs(np.imag(twoorb_rdm[rowi,coli]) + np.imag(twoorb_rdm[coli,rowi]))>1e-10):
                                #if(abs(np.imag(twoorb_rdm[rowi, coli]))>1e-1):
                                print("\n########################### rho_ij not Hermitian! #############################\n");
                                print("<{:.0f}|rho_ij|{:.0f}> --> O({:.0f}) O({:.0f})".format(rowi, coli, *Om_On));
                                print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(rowi, coli, np.real(twoorb_rdm[rowi, coli]), np.imag(twoorb_rdm[rowi, coli]))); 
                                print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(coli, rowi, np.real(twoorb_rdm[coli,rowi]), np.imag(twoorb_rdm[coli, rowi]))); 
                                for eigval in np.linalg.eig(twoorb_rdm)[0]: print(eigval);
                                raise ValueError;

            # diagonalize two orb rdm
            #print("before diagonalization:\n",twoorb_rdm[6:10,6:10]);
            twoorb_eigvals, _ = np.linalg.eigh(twoorb_rdm); # these will all be real

            # clean up numerical instabilities in eigvals
            for eigi in range(len(twoorb_eigvals)): 
                if(twoorb_eigvals[eigi]<0):
                    if(abs(twoorb_eigvals[eigi])<1e-10): twoorb_eigvals[eigi] = abs(twoorb_eigvals[eigi]); # eliminate log(-x)=nan
                    else: print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(eigi,eigi, np.real(twoorb_eigvals[eigi]), np.imag(twoorb_eigvals[eigi]))); raise ValueError;
                elif(twoorb_eigvals[eigi]==0): twoorb_eigvals[eigi] = np.exp(-100); # replace log(0.0)=-inf with log(e^-100)=-100

                                                                    # <---- maybe change e^-100 above
            del twoorb_rdm;
            if False:
                print("after diagonalization:");
                for eigi in range(len(twoorb_eigvals)): print(twoorb_eigvals[eigi]);
                print("after log(rho):");
                for eigi in range(len(twoorb_eigvals)): print(np.log(twoorb_eigvals[eigi]));
                print("after multiplication:");
                for eigi in range(len(twoorb_eigvals)): print(-twoorb_eigvals[eigi]*np.log(twoorb_eigvals[eigi]));
            # finish cleaning up numerical instabilities

            # get non Neumann entropy
            ents[sitei, sitej] = np.dot( -twoorb_eigvals, np.log(twoorb_eigvals)); # = trace since 2orb_rdm is diag
            ents[sitej, sitei] = 1*ents[sitei, sitej]; # symmetrize
            print("ents2[{:.0f},{:.0f}] = {:.10f}".format(sitei, sitej, ents[sitei, sitej]));

    return ents;

def twoorb_entropies_impurity(psi, eris_or_driver, whichsites, are_all_impurities, block):
    '''
    Compute the two-orbital reduced density matrix and extract the von Neumann entropy

    SPECIAL CASE that only singly-occupied states are allowed in the basis ("impurity sites" rather than "molecular orbitals")
    whichsites tells us all such states
    '''
    if(core.SymmetryTypes.SZ not in eris_or_driver.bw.symm_type): raise TypeError;
    assert(are_all_impurities);

    # return value
    ents = np.full((eris_or_driver.n_sites,eris_or_driver.n_sites), np.inf, dtype=float); # 

    # O(m) operators chosen come from Reiher 2013 (https://doi.org/10.1021/ct400247p) Table 3
    twoorb_Oms = [[[ 1, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 1, 6],[ 2, 5],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 5, 2],[ 6, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 1,11],[ 3, 9],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 9, 3],[11, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]], 
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 6, 6],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 1,16],[ 2,15],[ 3,14],[ 4,13],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 5,12],[ 6,11],[ 7,10],[ 8, 9],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 9, 8],[10, 7],[11, 6],[12, 5],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[13, 4],[14, 3],[15, 2],[16, 1],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[11,11],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 6,16],[ 8,14],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[14, 8],[16, 6],[ 0, 0],[ 0, 0],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[11,16],[12,15],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[15,12],[16,11],[ 0, 0]],
                  [[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[16,16]]];
    twoorb_Oms = np.array(twoorb_Oms, dtype=int);
    twoorb_Oms_flags = np.array([[True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,False,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,False,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,False,False,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,False,False,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,False,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,False,True ,True ],
                                 [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ,True ]]);
    #TRUNCATE above to singly-occupied basis states only!
    singly_occupied_mask = np.array([False, False, False, False, False, True, False, True, True, False, True, False, False, False, False, False]);
    if(are_all_impurities):
        twoorb_Oms = twoorb_Oms[singly_occupied_mask][:,singly_occupied_mask];
        twoorb_Oms_flags = twoorb_Oms_flags[singly_occupied_mask][:,singly_occupied_mask];
    
    # only get entropy between sites both in whichsites
    site_mask = np.array([True if site in whichsites else False for site in range(eris_or_driver.n_sites)]);
    for sitei in range(eris_or_driver.n_sites):
        if(site_mask[sitei]): 
            jsites_iter = np.arange(0, eris_or_driver.n_sites)[site_mask];
            jsites_iter = jsites_iter[jsites_iter < sitei]; # just do j<i and use ents2[j,i] = ents2[i,j]
        else: jsites_iter = []; 
        for sitej in jsites_iter:

            # two-orbital reduced density matrix for (sitei, sitej) pair
            twoorb_rdm = np.zeros((len(twoorb_Oms),len(twoorb_Oms)),dtype=complex); # reduced density matrix elements are in general complex

            # expressions from Reiher 2013 (https://doi.org/10.1021/ct400247p) Table 3
            for rowi in range(len(twoorb_Oms)):
                for coli in range(rowi+1): # only do lower half triangle, then use symmetry
                    Om_On = twoorb_Oms[rowi, coli];
                    if(not np.any(Om_On)): pass; # this matrix element is always zero
                    else: # Om_On is a tuple of integers telling which Om, On operator expressions to use
                        if(twoorb_Oms_flags[rowi,coli] == True):
                            Om_exprs, Om_coefs = get_Om(Om_On[0], are_all_impurities);
                            On_exprs, On_coefs = get_Om(Om_On[1], are_all_impurities); # <-- impurity assumption

                        else: # instead of taking <O(m)O(n)> we use the complex conjugate of the transposed matrix element
                            assert(not are_all_impurities);
                            Om_On = twoorb_Oms[coli,rowi]; #<--- Here is where we call for transposed matrix element
                            assert(twoorb_Oms_flags[coli,rowi]==True); # otherwise neither is correct!
                            Om_exprs, Om_coefs = get_Om(Om_On[0], are_all_impurities);
                            On_exprs, On_coefs = get_Om(Om_On[1], are_all_impurities);

                        # distribute two lists of expressions into one
                        combo_exprs = []
                        combo_coefs = [];
                        combo_sites = [];
                        for Omi in range(len(Om_exprs)):
                            for Oni in range(len(On_exprs)):
                                combo_exprs.append(Om_exprs[Omi] + On_exprs[Oni]);
                                combo_coefs.append(Om_coefs[Omi] * On_coefs[Oni]);
                                combo_sites.append( [sitei]*len(Om_exprs[Omi]) + [sitej]*len(On_exprs[Oni]));

                        # expectation value
                        expect_builder = eris_or_driver.expr_builder();
                        for termi in range(len(combo_exprs)):
                            #print(termi, combo_exprs[termi], combo_sites[termi], combo_coefs[termi]);
                            expect_builder.add_term(combo_exprs[termi], combo_sites[termi], combo_coefs[termi]);
                        expect_mpo = eris_or_driver.get_mpo(expect_builder.finalize(adjust_order=True, fermionic_ops="cdCD"));
                        expect_val = compute_obs(psi, expect_mpo, eris_or_driver);
                        if(twoorb_Oms_flags[rowi,coli] == False):
                            assert(not are_all_impurities);
                            expect_val = np.conj(expect_val); #<--- Here is where we take conjugate of transposed matrix element
                        twoorb_rdm[rowi, coli] = expect_val;
                        twoorb_rdm[coli, rowi] = np.conj(expect_val); # fill in upper half triangle of rdm with complex conjugate of lower half

            # clean up numerical instabilities
            for rowi in range(len(twoorb_Oms)):
                for coli in range(len(twoorb_Oms)):
                    Om_On = twoorb_Oms[rowi, coli];
                    if(not np.any(Om_On)): pass; # this matrix element is always zero
                    else:
                        # diagonal elements must be real
                        if(rowi == coli):
                            if(abs(np.imag(twoorb_rdm[rowi, coli])) > 1e-10): 
                                print("\n########################### rho_ij diag not real! #############################\n");
                                print("<{:.0f}|rho_ij|{:.0f}> --> O({:.0f}) O({:.0f})".format(rowi, coli, *Om_On));
                                print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(rowi, coli, np.real(twoorb_rdm[rowi, coli]), np.imag(twoorb_rdm[rowi, coli]))); raise ValueError;
                        # upper half triangle elements must be complex conjugate of lower half
                        if(coli>rowi):
                            if(abs(np.real(twoorb_rdm[rowi,coli]) - np.real(twoorb_rdm[coli,rowi]))>1e-10 or abs(np.imag(twoorb_rdm[rowi,coli]) + np.imag(twoorb_rdm[coli,rowi]))>1e-10):
                                #if(abs(np.imag(twoorb_rdm[rowi, coli]))>1e-1):
                                print("\n########################### rho_ij not Hermitian! #############################\n");
                                print("<{:.0f}|rho_ij|{:.0f}> --> O({:.0f}) O({:.0f})".format(rowi, coli, *Om_On));
                                print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(rowi, coli, np.real(twoorb_rdm[rowi, coli]), np.imag(twoorb_rdm[rowi, coli]))); 
                                print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(coli, rowi, np.real(twoorb_rdm[coli,rowi]), np.imag(twoorb_rdm[coli, rowi]))); 
                                for eigval in np.linalg.eig(twoorb_rdm)[0]: print(eigval);
                                raise ValueError;

            # diagonalize two orb rdm
            #print("before diagonalization:\n",twoorb_rdm[6:10,6:10]);
            twoorb_eigvals, _ = np.linalg.eigh(twoorb_rdm); # these will all be real

            # clean up numerical instabilities in eigvals
            for eigi in range(len(twoorb_eigvals)): 
                if(twoorb_eigvals[eigi]<0):
                    if(abs(twoorb_eigvals[eigi])<1e-10): twoorb_eigvals[eigi] = abs(twoorb_eigvals[eigi]); # eliminate log(-x)=nan
                    else: print("<{:.0f}|rho_ij|{:.0f}> = {:.10f}+{:.10f}j".format(eigi,eigi, np.real(twoorb_eigvals[eigi]), np.imag(twoorb_eigvals[eigi]))); raise ValueError;
                elif(twoorb_eigvals[eigi]==0): twoorb_eigvals[eigi] = np.exp(-100); # replace log(0.0)=-inf with log(e^-100)=-100

            del twoorb_rdm;
            if False:
                print("after diagonalization:");
                for eigi in range(len(twoorb_eigvals)): print(twoorb_eigvals[eigi]);
                print("after log(rho):");
                for eigi in range(len(twoorb_eigvals)): print(np.log(twoorb_eigvals[eigi]));
                print("after multiplication:");
                for eigi in range(len(twoorb_eigvals)): print(-twoorb_eigvals[eigi]*np.log(twoorb_eigvals[eigi]));
            # finish cleaning up numerical instabilities

            # get non Neumann entropy
            ents[sitei, sitej] = np.dot( -twoorb_eigvals, np.log(twoorb_eigvals)); # = trace since 2orb_rdm is diag
            ents[sitej, sitei] = 1*ents[sitei, sitej]; # symmetrize
            print("ents2[{:.0f},{:.0f}] = {:.10f}".format(sitei, sitej, ents[sitei, sitej]));

    return ents;

def mutual_info_wrapper(psi, eris_or_driver, whichsites, are_all_impurities, block, verbose=0):
    '''
    Get the mutual information between two sites, from their shared two-orbital von Neumann entropy and individual one-orbital von Neumann entropies

    Like S2_wrapper, this is designed as a pairwise observable between two impurity sites, but can be used on two molecular orbitals also
    
    Mutual information is a measure of entanglement.
    For two impurity sites, max entanglement is MI=ln(2)
    For two molecular orbitals, max entanglement is MI=ln(4)
    In both cases, min entanglement is MI=0
    '''
    if(not block): raise NotImplementedError;
    if(len(whichsites) != 2): raise ValueError; # like (S1+S2)^2, for now stick to pairwise impurities
    assert(not(whichsites[0] == whichsites[1])); # sites have to be distinct
    if(not isinstance(are_all_impurities, bool)): raise TypeError;

    # whether to treat all sites as impurity sites or molecular orbitals
    # NB oneorb_entropies wrapper can treat separate sites as impurity/mol orb separately
    # but the rest of the code lacks this functionality for now
    if(are_all_impurities): sites_are_imps = np.ones_like(whichsites, dtype=int);
    else: sites_are_imps = np.zeros_like(whichsites, dtype=int);
    site_mask = [True if site in whichsites else False for site in range(eris_or_driver.n_sites)];
    
    # von Neumann entropies, 1 orbital and 2 orbitals
    ents1 = oneorb_entropies_wrapper(psi, eris_or_driver, whichsites, sites_are_imps, block);
    ents2 = twoorb_entropies_impurity(psi, eris_or_driver, whichsites, are_all_impurities, block);

    # mutual information
    minfo = 0.5 * (ents1[:, None] + ents1[None, :] - ents2) * (1 - np.identity(len(ents1))); # (-1) * 2013 Reiher Eq (3)
    minfo = minfo[site_mask][:,site_mask];
    assert(len(minfo)==2);
    return minfo[0,1]; # off diagonal

def get_sz(eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing <Sz> of site whichsite
    '''
    if(block): builder = eris_or_driver.expr_builder()
    else: 
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite], 0.5);
        builder.add_term("CD",[whichsite,whichsite],-0.5);
    else:
        h1e[nloc*whichsite+0,nloc*whichsite+0] += 0.5;
        h1e[nloc*whichsite+1,nloc*whichsite+1] +=-0.5;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_sz2(eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing <Sz * Sz> of site whichsite
    '''
    #return eris_or_driver.get_spin_square_mpo(); this method works for *entire* system not site
    if(block): builder = eris_or_driver.expr_builder()
    else: 
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cdcd",[whichsite,whichsite,whichsite,whichsite], 0.25);
        builder.add_term("cdCD",[whichsite,whichsite,whichsite,whichsite],-0.25);
        builder.add_term("CDcd",[whichsite,whichsite,whichsite,whichsite],-0.25);
        builder.add_term("CDCD",[whichsite,whichsite,whichsite,whichsite], 0.25);
    else: 
        # g_pqrs a_p^+ a_q a_r^+ a_s
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0] += 0.25;
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+1] += -0.25;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+0] += -0.25;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1] += 0.25;
        # switch particle labels
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0] += 0.25;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+0] += -0.25;
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+1] += -0.25;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1] += 0.25;

        # - delta_qr a_p^+ a_s
        h1e[nloc*whichsite+0,nloc*whichsite+0] += 0.25;
        h1e[nloc*whichsite+0,nloc*whichsite+1] += -0.25;
        h1e[nloc*whichsite+1,nloc*whichsite+0] += -0.25;
        h1e[nloc*whichsite+1,nloc*whichsite+1] += 0.25;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_sxy(eris_or_driver, whichsite, block, sigmax, squared, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing <Sx>,<Sx^2>,<Sy>, or <Sy^2> of site whichsite
    '''
    if(block): builder = eris_or_driver.expr_builder()
    else: 
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=complex), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=complex);

    # construct
    if(not squared):
        if(sigmax): coefs = np.array([0.5,0.5]);
        else: coefs = np.array([complex(0,-0.5), complex(0,0.5)]);
        if(block):
            raise NotImplementedError;
        else:
            h1e[nloc*whichsite+0,nloc*whichsite+1] += coefs[0];
            h1e[nloc*whichsite+0,nloc*whichsite+1] += coefs[1];
    else:
        if(sigmax): coefs = np.array([0.25,0.25,0.25,0.25]);
        else: coefs = (-1)*np.array([0.25,-0.25,-0.25,0.25]);
        if(block):
            builder.add_term("cDcD",[whichsite,whichsite,whichsite,whichsite],coefs[0]);
            builder.add_term("cDCd",[whichsite,whichsite,whichsite,whichsite],coefs[0]);
            builder.add_term("CdcD",[whichsite,whichsite,whichsite,whichsite],coefs[0]);
            builder.add_term("CdCd",[whichsite,whichsite,whichsite,whichsite],coefs[0]);
        else:
            g2e[nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+1] += coefs[0];
            g2e[nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+0] += coefs[1];
            g2e[nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+1] += coefs[2];
            g2e[nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+0] += coefs[3];
            # switch particle labels
            g2e[nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+1] += coefs[0];
            g2e[nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+1] += coefs[1];
            g2e[nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+0] += coefs[2];
            g2e[nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+0] += coefs[3];

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_Sd_mu(eris_or_driver, whichsite, block, component="z", verbose=0):
    '''
    MPO representing <Sz> of site impurity at site whichsite
    '''
    if(not block): raise NotImplementedError;
    builder = eris_or_driver.expr_builder();

    # construct
    if(component=="z"):
        builder.add_term("Z",[whichsite], 1.0);
    elif(component=="x01"):
        builder.add_term("P",[whichsite], 1.0);
    elif(component=="x10"):
        builder.add_term("M",[whichsite], 1.0);
    elif(component=="y01"):
        builder.add_term("M",[whichsite], complex(0,-1));
    elif(component=="y10"):
        builder.add_term("M",[whichsite], complex(0,1));
    else: raise NotImplementedError;

    return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    
def get_Sd_z2(eris_or_driver, whichsite, block, verbose=0):
    '''
    MPO representing <Sz^2> of site impurity at site whichsite
    '''
    if(not block): raise NotImplementedError;
    builder = eris_or_driver.expr_builder();

    # construct
    builder.add_term("ZZ",[whichsite,whichsite], 1.0);

    return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    
def S2_wrapper(psi, eris_or_driver, whichsites, is_impurity, block, verbose=0):
    '''
    Build MPO and get expect val for the pairwise observable (S1+S2)^2

    Typically the pair of spins will be impurity sites rather than molecular orbitals, but both options are supported as long as each spin is same category
    '''
    if(not block): raise NotImplementedError;
    if(len(whichsites) != 2): raise ValueError; # (S1+S2)^2 is a pairwise observable
    assert(not(whichsites[0] == whichsites[1])); # sites have to be distinct
    if(not isinstance(is_impurity, bool)): raise TypeError;
    builder = eris_or_driver.expr_builder();

    # construct
    which1, which2 = whichsites;
    if(not is_impurity): # between fermions on two molecular orbitals
        for jpair in [[which1,which1,which1,which1], [which1,which1,which2,which2], [which2,which2,which1,which1], [which2,which2,which2,which2]]:
            builder.add_term("cdcd", jpair, 0.25);
            builder.add_term("cdCD", jpair,-0.25);
            builder.add_term("CDcd", jpair,-0.25);
            builder.add_term("CDCD", jpair, 0.25);
            builder.add_term("cDCd", jpair, 0.5);
            builder.add_term("CdcD", jpair, 0.5);

    else: # between two impurities
        for jpair in [[which1,which1], [which1,which2], [which2,which1], [which2,which2]]:
            builder.add_term("ZZ",jpair,1.0);
            builder.add_term("PM",jpair,0.5);
            builder.add_term("MP",jpair,0.5);

    # return
    mpo = eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    ret = compute_obs(psi, mpo, eris_or_driver);
    if(abs(np.imag(ret)) > 1e-10): print(ret); raise ValueError;
    return np.real(ret);

def purity_wrapper(psi,eris_or_driver, whichsite, block):
    '''
    Need to combine ops for x,y,z components of Sd to get purity
    '''
    components = ["x01","x10","y01","y10","z"];
    sterms = [];
    for comp in components:
        op = get_Sd_mu(eris_or_driver, whichsite, block, component=comp);
        sterms.append( compute_obs(psi, op, eris_or_driver));
    purity_vec = np.array([sterms[0]+sterms[1], sterms[2]+sterms[3], sterms[4]]);    
    ret = np.sqrt( np.dot(np.conj(purity_vec), purity_vec));
    if(abs(np.imag(ret)) > 1e-10): print(ret); raise ValueError;
    return np.real(ret);

def get_chirality(eris_or_driver, whichsites, block, symm_block, verbose=0):
    '''
    MPO representing S1 \cdot (S2 \times S3)
    '''
    assert(block);
    builder = eris_or_driver.expr_builder();

    def sblock_from_string(st):
        sblock=0;
        for c in st:
            if(c in ["c","D"]): sblock += 1;
            elif(c in ["C","d"]): sblock += -1;
            else: raise Exception(c+" not in [c,d,C,D]");
        if(sblock not in [-4,0,4]): print(sblock); raise ValueError;
        return sblock//2;

    def string_from_pauli(pauli):
        if(pauli=="x"): st, coefs = ["cD","Cd"], [1,1];
        elif(pauli=="y"): st, coefs = ["cD","Cd"], [complex(0,-1),complex(0,1)];
        elif(pauli=="z"): st, coefs = ["cd","CD"], [1,-1];
        else: raise Exception(pauli+" not in [x,y,z]");
        return st, coefs;

    def term_from_pauli(pauli3, coef):
        if(len(pauli3) != 3): raise ValueError;
        s0vals, c0vals = string_from_pauli(pauli3[0]);
        s1vals, c1vals = string_from_pauli(pauli3[1]);
        s2vals, c2vals = string_from_pauli(pauli3[2]);
        for s0i in range(len(s0vals)):
            for s1i in range(len(s1vals)):
                for s2i in range(len(s2vals)):
                    s_full = s0vals[s0i]+s1vals[s1i]+s2vals[s2i];
                    c_full = coef*c0vals[s0i]*c1vals[s1i]*c2vals[s2i];
                    print(s_full, "{:.2f}+{:.2f}j".format(np.real(c_full),np.imag(c_full)), sblock_from_string(s_full));
        return s_full, c_full;

    # brute force strings
    terms_m2, terms_0, terms_p2 = {}, {}, {}
    # 1st set
    term_from_pauli("xyz",1);
    assert False;

    # separate terms by symm_block
    ima = complex(0,1);
    if(symm_block == 2):
        terms = {"cDcDcd":-1*ima,"cDcDCD":1*ima};
    elif(symm_block == 0):
        terms = {"cDCdcd":1*ima,"cDCdCD":-1*ima,"CdcDcd":-1*ima,"CdcDCD":1*ima};
    elif(symm_block == -2):
        terms = {"CdCdcd":1*ima,"CdCdCD":-1*ima};
    else:
        raise NotImplementedError;

    # construct
    jlist = [whichsites[0],whichsites[0],whichsites[1],whichsites[1],whichsites[2],whichsites[2]];

    return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);

def chirality_wrapper(psi,eris_or_driver, whichsites, block):
    '''
    Need to combine symmetry blocks of chirality op
    '''
    sblocks = [2,0,-2];
    sterms = [];
    for sblock in sblocks:
        op = get_chirality(eris_or_driver, whichsites, block, sblock);
        sterms.append( compute_obs(psi, op, eris_or_driver));
    ret = np.sum(sterms);
    if(abs(np.imag(ret)) > 1e-10): print(ret); raise ValueError;
    return np.real(ret);

def get_pcurrent(eris_or_driver, whichsite, sigma, block, verbose=0):
    '''
    MPO for particle current from whichsite-1 to whichsite
    positive particle current is rightward, associated with positive bias st left 
    lead chem potential > right lead chem potential
    
    Ultimately, what we calculate here feeds into two observables:
        (1) the current, from Eq (69) in Garnet's coupled cluster dynamics paper, JCP 2021:
        <J_j> =  e/\hbar * hopping * i * \sum_sigma 
        <c_j,\sigma^\dagger c_j-1,\sigma - c_j-1,\sigma^\dagger c_j,\sigma >

        (2) the conductance in units of the conductance quanta G0=e^2/\pi\hbar
        G/G0 = \pi <J>/(Vb/e), where Vb/e is a VOLTAGE
    
    HOWEVER here we just calculate i*<c_j,\sigma^\dagger c_j-1,\sigma - c_j-1,\sigma^\dagger c_j,\sigma>
    We always wait until after wrapper (plotting step) to apply the prefactors, which are either
    (1) e/\hbar * hopping
    (2) \pi*hopping/Vb

    Args:
    eris_or_driver, Block2 driver
    whichsite, int, the site index -> j, then we find current between j-1, j
    sigma, int 0 or 1, meaning up or down current
    '''
    if(sigma==0): sigmastr = "cd";
    elif(sigma==1): sigmastr = "CD";
    else: raise ValueError;
    
    # we get current between this site and the one to its left, with rightward particle flow positive
    whichsites = [whichsite-1, whichsite];

    if(block):# construct MPO
        builder = eris_or_driver.expr_builder();
        builder.add_term(sigmastr, whichsites[::-1], complex(0,1)); # c on right, d on left = positive particle current
        builder.add_term(sigmastr, whichsites, complex(0,-1)); # c on left, d on right = negative particle current
        return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: # construct ERIs
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=complex), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=complex);
        h1e[nloc*whichsites[1]+sigma,nloc*whichsites[0]+sigma] += complex(0, 1.0);
        h1e[nloc*whichsites[0]+sigma,nloc*whichsites[1]+sigma] += complex(0,-1.0);
        return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff, imag_cutoff = 1e-12);
        
def pcurrent_wrapper(psi, eris_or_driver, whichsite, block, verbose=0):
    '''
    Consider site whichsite. This wrapper sums the spin currents (see get_pcurrent)
    from whichsite-1 to whichsite (LEFT part)
    In plotting steps, we multiply this by e/\hbar * hopping to make it current
    '''
    if(block): compute_func = compute_obs;
    else: compute_func = tdfci.compute_obs;

    the_pcurrent = 0.0;
    for sigma in [0,1]:
        the_mpo = get_pcurrent(eris_or_driver, whichsite, sigma, block, verbose=verbose);
        the_pcurrent += compute_func(psi, the_mpo, eris_or_driver);

    # average
    ret = 1*(the_pcurrent); # must add e/\hbar * hopping factor later
    if(abs(np.imag(ret)) > 1e-10): print(ret); raise ValueError;
    return np.real(ret);

def conductance_wrapper(psi, eris_or_driver, whichsite, block, verbose=0):
    '''
    Consider site whichsite. This wrapper:
    1) sums the spin currents from whichsite-1 to whichsite (LEFT part)
    In plotting steps, we multiply this by  \pi*hopping/Vb to make it conductance/G0
    '''
    if(block): compute_func = compute_obs;
    else: compute_func = tdfci.compute_obs;

    # left part
    pcurrent_left = 0.0;
    for sigma in [0,1]:
        left_mpo = get_pcurrent(eris_or_driver, whichsite, sigma, block, verbose=verbose);
        left_val = compute_func(psi, left_mpo, eris_or_driver);
        pcurrent_left += left_val;

    # just return left part
    ret = 1*(pcurrent_left); # must add  \pi*hopping/Vb factor later
    if(abs(np.imag(ret)) > 1e-10): print(ret); raise ValueError;
    return np.real(ret);

def band_wrapper(psi, eris_or_driver, whichsite, params_dict, block, verbose=0):
    '''
    '''

    # load data from json
    v, w, Vg = params_dict["v"], params_dict["w"], params_dict["Vg"];
    Ntotal = params_dict["NL"]+params_dict["NFM"]+params_dict["NR"];

    # classify site indices (spin not included)
    RMdofs = 2;
    lleads = np.arange(params_dict["NL"]); # <-- blocks
    centrals = np.arange(params_dict["NL"],params_dict["NL"]+params_dict["NFM"])
    rleads = np.arange(params_dict["NL"]+params_dict["NFM"],Ntotal);
    alls = np.arange(Ntotal);
    Nmolorbs = RMdofs*Ntotal;

    # which lead to project onto
    whichblock = whichsite // 2; 
    if(whichblock in lleads): site_projector = np.arange(RMdofs*params_dict["NL"]); 
    elif(whichblock in centrals): site_projector = np.arange(RMdofs*params_dict["NL"],RMdofs*(params_dict["NL"]+params_dict["NFM"]));
    elif(whichblock in rleads): site_projector = np.arange(RMdofs*(params_dict["NL"]+params_dict["NFM"]),RMdofs*Ntotal);
    else: print(whichblock); raise NotImplementedError;
    # override - all sites
    site_projector = np.arange(RMdofs*Ntotal);

    # construct single-body Hamiltonian as matrix
    h1e_twhen, g2e_dummy = H_RM_builder(params_dict, block=False);
    h1e_twhen, g2e_dummy = H_RM_polarizer(params_dict, (h1e_twhen, g2e_dummy), block=False);
    h1e_twhen = h1e_twhen[::2,::2]; # <- make spinless !!
    print(h1e_twhen[:8,:8]);
    print(h1e_twhen[-8:,-8:]);

    # diagonalize single-body Hamiltonian -> energy eigenstates
    vals_twhen, vecs_twhen = np.linalg.eigh(h1e_twhen);
    vecs_twhen = vecs_twhen.T;

    # output for density of states
    print("\n\nH_RM_builder + H_RM_polarizer energies");
    print([val for val in vals_twhen]);
    assert(Nmolorbs == len(vals_twhen)); # 1 eigenval for each spinless orb

    # which band to project onto
    # valence band
    if(whichsite % 2 ==0): band_divider = np.arange(0,len(vals_twhen)//2);
    # conduction band
    elif(whichsite % 2 == 1): band_divider = np.arange(len(vals_twhen)//2,len(vals_twhen));
    print("in band_wrapper");
    print("whichblock = {:.0f}, whichsite = {:.0f}".format(whichblock, whichsite)+"->")
    print("band_divider = ", band_divider, "\n({:.0f} total k states)".format(len(vecs_twhen)));
    print("site_projector = ", site_projector, "\n({:.0f} total sites)".format(len(vecs_twhen[0])))
    #assert False;

    # observable for band occupancy
    if(block):
        builder = eris_or_driver.expr_builder();
    else:
        nloc = 2;
        Nspinorbs = nloc*RMdofs*Ntotal;
        h1e, g2e = np.zeros((Nspinorbs, Nspinorbs),dtype=float);
        g2e_zeros = np.zeros((Nspinorbs, Nspinorbs, Nspinorbs, Nspinorbs),dtype=float);

    # coefs for this observable come from energy eigenstates
    assert(len(vecs_twhen[0]) == Nmolorbs); # spatial extent of vecs = number of spinless orbs 
    for kmvali in band_divider: #iter over states in band
        for j in site_projector:
            for jp in site_projector:
                for sigma in [0,1]:
                    spinstr = ["cd","CD"][sigma];
                    alpha_alpha = vecs_twhen[kmvali,j]*np.conj(vecs_twhen[kmvali,jp]);
                    if(block):
                        builder.add_term(spinstr,[j,jp],alpha_alpha);
                    else:
                        h1e[nloc*j+sigma,nloc*jp+sigma] += alpha_alpha;

    # fci operator
    if(not block): op = tdfci.ERIs(h1e, g2e_zeros, eris_or_driver.mo_coeff);

    # matrix product operator
    band_mpo = eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    ret = compute_obs(psi, band_mpo, eris_or_driver);
    if(abs(np.imag(ret)) > 1e-10): print(ret); raise ValueError;
    return np.real(ret);


##########################################################################################################
#### hamiltonian constructors

def reblock(mat):
    '''
    reshape a 4d matrix which has shape (outer_dof,outer_dof,inner_dof,inner_dof)
    into shape (inner_dof,inner_dof,outer_dof,outer_dof)
    '''

    outer_dof, _, inner_dof, _ = np.shape(mat);
    new_mat = np.zeros((inner_dof,inner_dof,outer_dof,outer_dof),dtype=mat.dtype);
    for outi in range(outer_dof):
        for outj in range(outer_dof):
            for ini in range(inner_dof):
                for inj in range(inner_dof):
                    new_mat[ini,inj,outi,outj] = mat[outi,outj,ini,inj];
    return utils.mat_4d_to_2d(new_mat);

def H_fermion_builder(params_dict, block, scratch_dir="tmp",verbose=0):
    '''
    '''
    assert(params_dict["sys_type"]=="fermion");

    # load data from json
    tl, Jsd, Jz, Jx = params_dict["tl"], params_dict["Jsd"], params_dict["Jz"], params_dict["Jx"];
    NL, NFM, NR = params_dict["NL"], params_dict["NFM"], params_dict["NR"];

    # fermionic sites and spin
    Nsites = NL+2*NFM+NR; # ALL sites now fermion sites
    Ne_jsites, TwoSz = params_dict["Ne"], params_dict["TwoSz"]; #Ne param gives # of *j site fermions*
    Ne = Ne_jsites + NFM; # we assume NFM sites (d sites) are all singly filled

    # classify site indices (spin not included)
    llead_j = np.arange(NL);
    central_j = np.arange(NL,NL+2*NFM,2);
    central_d = central_j + 1;
    rlead_j = np.arange(NL+2*NFM,Nsites);
    all_j = np.append(llead_j,np.append(central_j, rlead_j));

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir, symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
        else: raise NotImplementedError;
        builder = driver.expr_builder();
        print("\n",40*"#","\nConstructed builder\n",40*"#","\n");
    else:       # <---------- change dtype to complex ?
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        h1e, g2e = np.zeros((Nspinorbs, Nspinorbs),dtype=float), np.zeros((Nspinorbs, Nspinorbs, Nspinorbs, Nspinorbs),dtype=float);
        
    # j <-> j+1 hopping everywhere
    for whichj in range(len(all_j[:-1])):
        if(block):
            builder.add_term("cd",[all_j[whichj],all_j[whichj+1]],-tl); 
            builder.add_term("CD",[all_j[whichj],all_j[whichj+1]],-tl);
            builder.add_term("cd",[all_j[whichj+1],all_j[whichj]],-tl);
            builder.add_term("CD",[all_j[whichj+1],all_j[whichj]],-tl);
        else:
            raise NotImplementedError;

    # Heisenberg exchange btwn adjacent central_d
    for whichd in range(len(central_d[:-1])):
        thisd, nextd = central_d[whichd], central_d[whichd+1];
        # z terms
        builder.add_term("cdcd",[thisd,thisd,nextd,nextd],-Jz/4);
        builder.add_term("cdCD",[thisd,thisd,nextd,nextd], Jz/4);
        builder.add_term("CDcd",[thisd,thisd,nextd,nextd], Jz/4);
        builder.add_term("CDCD",[thisd,thisd,nextd,nextd],-Jz/4);
        # plus minus terms
        builder.add_term("cDCd",[thisd,thisd,nextd,nextd],-Jx/2);
        builder.add_term("CdcD",[thisd,thisd,nextd,nextd],-Jx/2);

    # sd exchange btwn itinerant charge density on site j and singly occupied site d
    for whichj in range(len(central_j)):
        thisj, thisd = central_j[whichj], central_d[whichj];
        # z terms
        builder.add_term("cdcd",[thisj,thisj,thisd,thisd],-Jsd/4);
        builder.add_term("cdCD",[thisj,thisj,thisd,thisd], Jsd/4);
        builder.add_term("CDcd",[thisj,thisj,thisd,thisd], Jsd/4);
        builder.add_term("CDCD",[thisj,thisj,thisd,thisd],-Jsd/4);
        # plus minus terms
        builder.add_term("cDCd",[thisj,thisj,thisd,thisd],-Jsd/2);
        builder.add_term("CdcD",[thisj,thisj,thisd,thisd],-Jsd/2);

    if(block): return driver, builder;
    else: return h1e, g2e;

def H_fermion_polarizer(params_dict, to_add_to, block, verbose=0):
    '''
    '''
    assert(params_dict["sys_type"]=="fermion");

    # load data from json
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];

    # fermionic sites and spin
    Nsites = NL+2*NFM+NR; # ALL sites now fermion sites

    # classify site indices (spin not included)
    conf_j = np.arange(Nconf);
    llead_j = np.arange(NL);
    central_j = np.arange(NL,NL+2*NFM,2);
    central_d = central_j + 1;
    rlead_j = np.arange(NL+2*NFM,Nsites);
    all_j = np.append(llead_j,np.append(central_j, rlead_j));
    
    # unpack builder
    if(block):
        driver, builder = to_add_to;
        if(driver.n_sites != Nsites): raise ValueError;
    else:
        h1e, g2e = to_add_to;
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        if(len(h1e) != Nspinorbs): raise ValueError;

    # weak hopping between d sites nd j sites to prevent local minima
    if("weak_t" in params_dict.keys()):
        weak_t = params_dict["weak_t"];
        for whichj in range(len(central_j)):
            thisj, thisd = central_j[whichj], central_d[whichj];
            if(block):
                builder.add_term("cd", [thisj, thisd], -weak_t);
                builder.add_term("CD", [thisj, thisd], -weak_t);
            else:
                raise NotImplementedError;

    # ARTIFICIAL Vg and U ensure single occ on d sites
    Vg_art, U_art = params_dict["Vg_art"], params_dict["U_art"];
    for d in central_d:
        if(block):
            builder.add_term("cd",[d,d], Vg_art);
            builder.add_term("CD",[d,d], Vg_art);
            builder.add_term("cdCD",[d,d,d,d], U_art);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vg_art;
            h1e[nloc*j+1,nloc*j+1] += Vg_art;
            assert(U_art==0.0);

    # B field on the d sites ----------> ASSUMED IN THE Z
    BFM = params_dict["BFM"];
    for d in central_d:
        if(block):
            builder.add_term("cd",[d,d],-BFM/2);
            builder.add_term("CD",[d,d], BFM/2);
        else:
            raise NotImplementedError;

    # B field that targets 1st d site only
    if("BFM_first" in params_dict.keys()):
        BFM_first = params_dict["BFM_first"];
        d = central_d[0];
        if(block):
            builder.add_term("cd",[d,d], (-1/2)*(BFM_first-BFM));
            builder.add_term("CD",[d,d], ( 1/2)*(BFM_first-BFM));
        else:
            raise NotImplementedError;

    # t<0 term to entangle localized spins (in triplet if positive term)
    if("Bent" in params_dict.keys()): 
        assert(len(central_d)==2);
        Bent = params_dict["Bent"];
        for whichd in range(len(central_d[:-1])):
            thisd, nextd = central_d[whichd], central_d[whichd+1];
            if(block):
                # z z terms
                builder.add_term("cdcd",[thisd,thisd,nextd,nextd],-Bent/4);
                builder.add_term("cdCD",[thisd,thisd,nextd,nextd], Bent/4);
                builder.add_term("CDcd",[thisd,thisd,nextd,nextd], Bent/4);
                builder.add_term("CDCD",[thisd,thisd,nextd,nextd],-Bent/4);
                # plus minus terms
                builder.add_term("cDCd",[thisd,thisd,nextd,nextd],-Bent/2);
                builder.add_term("CdcD",[thisd,thisd,nextd,nextd],-Bent/2);
            else:
                raise NotImplementedError;

    # potential in the confined region
    if("Vconf" in params_dict.keys()):
        Vconf = params_dict["Vconf"];
        for j in conf_j:
            if(block):
                builder.add_term("cd",[j,j],-Vconf);
                builder.add_term("CD",[j,j],-Vconf);
            else:
                raise NotImplementedError;

    # B field in the confined region ----------> ASSUMED IN THE Z
    Be = params_dict["Be"];
    for j in conf_j:
        if(block):
            builder.add_term("cd",[j,j],-Be/2);
            builder.add_term("CD",[j,j], Be/2);
        else:
            raise NotImplementedError;

    # B field on the j that couples to the first localized spin
    if("Bsd" in params_dict.keys()): 
        Bsd = params_dict["Bsd"];
        j = central_j[0];
        if(block):
            builder.add_term("cd",[j,j],-Bsd/2);
            builder.add_term("CD",[j,j], Bsd/2);
        else:
            raise NotImplementedError;

    # return
    if(block):
        mpo_from_builder = driver.get_mpo(builder.finalize());
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;


def H_RM_builder(params_dict, block, scratch_dir="tmp",verbose=0):
    '''
    Builds SIAM Hamiltonian in RM model at all time
    The physical params are contained in a .json file. They are all in eV.
    They are:
    v, w (RM hoppings), Vg (gate voltage on impurity),
    Vb (bias between left and right leads.
    Vb>0 means that left lead is higher chem potential than right, leading to
    rightward/positive current).

    NL (number sites in left lead),  NR (number of sites in right lead).
    The total Sz of the electrons is always 0, so Ne_up=Ne_down=Ne//2
    NB this requires that Ne%2==0

    There is NO supersiting in this system

    Returns: a tuple of DMRGDriver, ExprBuilder objects
    '''

    # load data from json
    v, w, u, th, Vb = params_dict["v"], params_dict["w"], params_dict["u"], params_dict["th"], params_dict["Vb"];
    #assert(abs(w) == abs(params_dict["th"]));
    NL, NFM, NR = params_dict["NL"], params_dict["NFM"], params_dict["NR"];
    Ntotal = NL+NFM+NR;

    Ne = params_dict["Ne"];   
    assert(Ne%2 ==0); # need even number of electrons for TwoSz=0
    TwoSz = 0;        # <------ !!!!

    # classify site indices (spin not included)
    RMdofs = 2;
    lleads = np.arange(NL); # <-- blocks
    centrals = np.arange(NL,NL+NFM);
    rleads = np.arange(NL+NFM,Ntotal);
    alls = np.arange(Ntotal);
    Nmolorbs = RMdofs*Ntotal;

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir, symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            driver.initialize_system(n_sites=Nmolorbs, n_elec=Ne, spin=TwoSz);
            print(">>> driver(n_sites={:.0f}, n_elec={:.0f}, spin={:.0f})".format(Nmolorbs, Ne, TwoSz));
        else: raise NotImplementedError;
        
        # def custom states and operators
        if(params_dict["sys_type"] in ["SIETS_RM"]):
            qnumber_wrapper = driver.bw.SX # quantum number wrapper function
            custom_states, custom_ops = get_custom_states_ops(params_dict, qnumber_wrapper);
            # input custom site basis states and ops to driver, and build builder
            driver.ghamil = driver.get_custom_hamiltonian(custom_states, custom_ops)
        builder = driver.expr_builder();
        print("\n",40*"#","\nConstructed builder\n",40*"#","\n");     
    # end of if(block) code

    else:   
        nloc = 2;
        Nspinorbs = nloc*RMdofs*Ntotal;
        h1e, g2e = np.zeros((Nspinorbs, Nspinorbs),dtype=float), np.zeros((Nspinorbs, Nspinorbs, Nspinorbs, Nspinorbs),dtype=float);

    # j <-> j+1 hopping for fermions everywhere
    for sigma in [0,1]:
        spinstr = ["cd","CD"][sigma];
        for j in alls[:-1]:
            # j is block -> spin orb index
            muA, muB, muA_next = RMdofs*j, RMdofs*j+1, RMdofs*(j+1)
            if(block):
                builder.add_term(spinstr,[muA,muB],v); 
                builder.add_term(spinstr,[muB,muA],v);
                builder.add_term(spinstr,[muB, muA_next],w);
                builder.add_term(spinstr,[muA_next, muB],w);
                builder.add_term(spinstr,[muA,muA], u); # staggered intra-dimer **potential**
                builder.add_term(spinstr,[muB,muB],-u);
            else:
                h1e[nloc*muA+sigma,nloc*muB+sigma] += v;
                h1e[nloc*muB+sigma,nloc*muA+sigma] += v;
                h1e[nloc*muB+sigma,nloc*muA_next+sigma] += w;
                h1e[nloc*muA_next+sigma,nloc*muB+sigma] += w;
                h1e[nloc*muA+sigma,nloc*muA+sigma] +=  u; # staggered intra-dimer **potential**
                h1e[nloc*muB+sigma,nloc*muB+sigma] += -u;
        for j in [alls[-1]]: # last block needs intra-block only
            muA, muB = RMdofs*j, RMdofs*j+1;
            if(block):
                builder.add_term(spinstr,[muA,muB],v);
                builder.add_term(spinstr,[muB,muA],v);
                builder.add_term(spinstr,[muA,muA], u); # staggered intra-dimer **potential**
                builder.add_term(spinstr,[muB,muB],-u);
            else:
                h1e[nloc*muA+sigma,nloc*muB+sigma] += v;
                h1e[nloc*muB+sigma,nloc*muA+sigma] += v;
                h1e[nloc*muA+sigma,nloc*muA+sigma] +=  u; # staggered intra-dimer **potential**
                h1e[nloc*muB+sigma,nloc*muB+sigma] += -u;

    # scattering region
    assert(params_dict["U"] == 0.0); # no Coulomb
    for sigma in [0,1]:
        spinstr = ["cd","CD"][sigma];
        if(params_dict["sys_type"] in ["SIETS_RM"]): # sd exchange btwn impurities & charge density on their site
            Jsd = params_dict["Jsd"];
            for j in centrals:
                if(sigma==0): # only do once since spin independent (hacky code)
                    muA, muB = RMdofs*j, RMdofs*j+1;
                    for jmu in [muA, muB]: # since these terms are identical for A, B orbitals
                        if(block):
                            # z terms
                            builder.add_term("cdZ",[jmu,jmu,jmu],-Jsd/2);
                            builder.add_term("CDZ",[jmu,jmu,jmu], Jsd/2);
                            # plus minus terms
                            builder.add_term("cDM",[jmu,jmu,jmu],-Jsd/2);
                            builder.add_term("CdP",[jmu,jmu,jmu],-Jsd/2);
                        else: # Jsd not supported for td-fci
                            assert(Jsd==0.0);
        if("Vg" in params_dict.keys()): # apply gate voltage
            Vg = params_dict["Vg"];
            for j in centrals:
                muA, muB = RMdofs*j, RMdofs*j+1;
                if(block):
                    builder.add_term(spinstr,[muA, muA], Vg);
                    builder.add_term(spinstr,[muB, muB], Vg);
                else:
                    h1e[nloc*muA+sigma,nloc*muA+sigma] += Vg;
                    h1e[nloc*muB+sigma,nloc*muB+sigma] += Vg;
        # hybridization btwn scattering region and leads
        for j_hyb in [centrals[0]-1,centrals[-1]]: # should be interblock only, i.e. muB <-> muA_next
            muA, muB, muA_next = RMdofs*j_hyb, RMdofs*j_hyb+1, RMdofs*(j_hyb+1);
            print("hybridization: j={:.0f} -> muB={:.0f}, muA_next={:.0f}".format(j_hyb, muB, muA_next));
            assert(th > 0.0); # to keep track of different sign conventions
            if(block):     
                builder.add_term(spinstr,[muB, muA_next], -w -th); # REMOVE `w` already there
                builder.add_term(spinstr,[muA_next, muB], -w -th);
            else:
                h1e[nloc*muB+sigma,nloc*muA_next+sigma] += -w -th; # REMOVE `w` already there
                h1e[nloc*muA_next+sigma,nloc*muB+sigma] += -w -th;     
    
    # bias (NB this will be REMOVED by polarizer so that it is ABSENT for t<0
    # and PRESENT at t>0 (opposite to B fields in STT, but still "added"
    # by the polarizer
    for sigma in [0,1]:
        spinstr = ["cd","CD"][sigma];
        for j in lleads:
            muA, muB = RMdofs*j, RMdofs*j+1;
            if(block):
                builder.add_term(spinstr,[muA,muA], Vb/2); 
                builder.add_term(spinstr,[muB,muB], Vb/2); 
            else:
                h1e[nloc*muA+sigma,nloc*muA+sigma] += Vb/2;
                h1e[nloc*muB+sigma,nloc*muB+sigma] += Vb/2;
        for j in rleads:
            muA, muB = RMdofs*j, RMdofs*j+1;
            if(block):
                builder.add_term(spinstr,[muA,muA], -Vb/2);
                builder.add_term(spinstr,[muB,muB], -Vb/2); 
            else:
                h1e[nloc*muA+sigma,nloc*muA+sigma] += -Vb/2;
                h1e[nloc*muB+sigma,nloc*muB+sigma] += -Vb/2;

    if(block): return driver, builder;
    else: return h1e, g2e;

def H_RM_polarizer(params_dict, to_add_to, block, verbose=0):
    '''
    Adds terms specific to the t<0 SIAM Hamiltonian in RM model
    (REMOVES Vb)

    There is NO supersiting in this system

    Args:
    Params_dict: dict containing physical param values, these are defined in Hsys_base
    to_add_to, tuple of objects to add terms to:
        if block is True: these will be DMRGDriver, ExprBuilder objects
        else: these will be 1-body and 2-body parts of the second quantized
        Hamiltonian

    Returns: a tuple of DMRGDriver, MPO
    '''

    # load data from json
    Vb = params_dict["Vb"];
    NL, NFM, NR = params_dict["NL"], params_dict["NFM"], params_dict["NR"];
    Ntotal = NL+NFM+NR;

    # classify site indices (spin not included)
    RMdofs = 2;
    lleads = np.arange(NL); # <-- blocks
    centrals = np.arange(NL,NL+NFM);
    rleads = np.arange(NL+NFM,Ntotal);
    alls = np.arange(Ntotal);

    # unpack ExprBuilder
    if(block):
        driver, builder = to_add_to;
        if(driver.n_sites != RMdofs*Ntotal): raise ValueError; # 2 * number of blocks
    else:
        h1e, g2e = to_add_to;
        nloc = 2;
        Nspinorbs = nloc*RMdofs*Ntotal;
        if(len(h1e) != Nspinorbs): raise ValueError;

    # REMOVE bias
    for sigma in [0,1]:
        spinstr = ["cd","CD"][sigma];
        for j in lleads:
            muA, muB = RMdofs*j, RMdofs*j+1;
            if(block):
                builder.add_term(spinstr,[muA,muA], -Vb/2);
                builder.add_term(spinstr,[muB,muB], -Vb/2);
            else:
                h1e[nloc*muA+sigma,nloc*muA+sigma] += -Vb/2;
                h1e[nloc*muB+sigma,nloc*muB+sigma] += -Vb/2;
        for j in rleads:
            muA, muB = RMdofs*j, RMdofs*j+1;
            if(block):
                builder.add_term(spinstr,[muA,muA], Vb/2);
                builder.add_term(spinstr,[muB,muB], Vb/2);
            else:
                h1e[nloc*muA+sigma,nloc*muA+sigma] += Vb/2;
                h1e[nloc*muB+sigma,nloc*muB+sigma] += Vb/2;

    # B field on the loc spins
    if(params_dict["sys_type"] in ["SIETS_RM"]):
        BFM = params_dict["BFM"];
        for j in centrals:
            muA, muB = RMdofs*j, RMdofs*j+1;
            for jmu in [muA,muB]:
                if(block):
                    builder.add_term("Z",[jmu],-BFM);
                else: raise NotImplementedError;
        
    # special case initialization 
    if("BFM_first" in params_dict.keys() ): # B field that targets 1st loc spin only
        BFM_first = params_dict["BFM_first"];
        j = centrals[0];
        muA = RMdofs*j;
        if(block):
            builder.add_term("Z",[muA], -BFM_first+BFM); # first orb of first block
        else: raise NotImplementedError;                
    if("Bent" in params_dict.keys() and len(centrals)==1): # B field that entangles 2 loc spins
        Bent = params_dict["Bent"];
        if("MSQ_spacer" in params_dict.keys()): # MSQs at each end of NFM only
            raise NotImplementedError;
        else: # NFM full of MSQs
            sitepairs = [];
            for j in centrals[:-1]: 
                muA, muB, muA_next = RMdofs*j, RMdofs*j+1, RMdofs*(j+1)
                sitepairs.append([muA, muB]); # inter-block pair
                sitepairs.append([muB, muA_next]); # intra-block pair
            for j in [centrals[-1]]:
                muA, muB = RMdofs*j, RMdofs*j+1;
                sitepairs.append([muA, muB]); # inter-block pair only
        print("entangled pairs = ",sitepairs)
        for sitepair in sitepairs: # sitepair is a list of two sites (not blocks!) to entangle
            if(not ("triplet_flag" in params_dict.keys())):
                print("no triplet flag");           # sometimes we need to skip ZZ term to get |T0>
                builder.add_term("ZZ",sitepair,-Bent); # rather than |T+>. Can ask for this with triplet_flag
            else: print("triplet flag");
            builder.add_term("PM",sitepair,-Bent/2);
            builder.add_term("MP",sitepair,-Bent/2);

    # return
    if(block):
        print("Finalizing builder")
        mpo_from_builder = driver.get_mpo(builder.finalize());
        print("Finalized builder");
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;

def H_SIAM_builder(params_dict, block, scratch_dir="tmp",verbose=0):
    '''
    Builds the parts of the SIAM Hamiltonian which apply at all t
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
    assert(params_dict["sys_type"]=="SIAM");

    # load data from json
    tl, th, Vg, U, Vb = params_dict["tl"], params_dict["th"], params_dict["Vg"], params_dict["U"], params_dict["Vb"];
    NL, NR = params_dict["NL"], params_dict["NR"];
    Nsites = NL+1+NR;
    assert("Ne" not in params_dict.keys());
    if("Ne_override" in params_dict.keys()):
        Ne = params_dict["Ne_override"];
    else:
        Ne = 1*Nsites;
    
    assert(Ne%2 ==0); # need even number of electrons for TwoSz=0
    TwoSz = 0;        # <------ !!!!

    # classify site indices (spin not included)
    llead_sites = np.arange(NL);
    central_sites = np.arange(NL,NL+1);
    rlead_sites = np.arange(NL+1,Nsites);
    all_sites = np.arange(Nsites);

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir, symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
        else: raise NotImplementedError;
        builder = driver.expr_builder();
        print("\n",40*"#","\nConstructed builder\n",40*"#","\n");
    else:       # <---------- change dtype to complex ?
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        h1e, g2e = np.zeros((Nspinorbs, Nspinorbs),dtype=float), np.zeros((Nspinorbs, Nspinorbs, Nspinorbs, Nspinorbs),dtype=float);


    # LEAD j <-> j+1 hopping for fermions
    for lead_sites in [llead_sites, rlead_sites]:
        for j in lead_sites[:-1]:
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

    # lead coupling to impurity
    jpairs = [(llead_sites[-1], central_sites[0]), (rlead_sites[0], central_sites[-1])];
    for jpair in jpairs:
        jlead, jimp = jpair;
        if(block):
            builder.add_term("cd",[jlead,jimp],-th);
            builder.add_term("CD",[jlead,jimp],-th);
            builder.add_term("cd",[jimp,jlead],-th);
            builder.add_term("CD",[jimp,jlead],-th);
        else:
            h1e[nloc*jlead+0,nloc*jimp+0] += -th;
            h1e[nloc*jimp+0,nloc*jlead+0] += -th;
            h1e[nloc*jlead+1,nloc*jimp+1] += -th;
            h1e[nloc*jimp+1,nloc*jlead+1] += -th;

    # Vg and U on impurity
    for j in central_sites:
        if(block):
            builder.add_term("cd",[j,j], Vg);
            builder.add_term("CD",[j,j], Vg);
            builder.add_term("cdCD",[j,j,j,j], U);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vg;
            h1e[nloc*j+1,nloc*j+1] += Vg;
            assert(U==0.0);

    # bias (NB this will be REMOVED by polarizer so that it is ABSENT for t<0
    # and PRESENT at t>0 (opposite to B fields in STT, but still "added"
    # by the polarizer
    for j in llead_sites:
        if(block):
            builder.add_term("cd",[j,j], Vb/2); 
            builder.add_term("CD",[j,j], Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vb/2;
            h1e[nloc*j+1,nloc*j+1] += Vb/2;
    for j in rlead_sites:
        if(block):
            builder.add_term("cd",[j,j],-Vb/2); 
            builder.add_term("CD",[j,j],-Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += -Vb/2;
            h1e[nloc*j+1,nloc*j+1] += -Vb/2;

    if(block): return driver, builder;
    else: return h1e, g2e;

def H_SIAM_polarizer(params_dict, to_add_to, block, verbose=0):
    '''
    Adds terms specific to the t<0 SIAM Hamiltonian (REMOVES Vb)

    There is NO supersiting in this system

    Args:
    Params_dict: dict containing physical param values, these are defined in Hsys_base
    to_add_to, tuple of objects to add terms to:
        if block is True: these will be DMRGDriver, ExprBuilder objects
        else: these will be 1-body and 2-body parts of the second quantized
        Hamiltonian

    Returns: a tuple of DMRGDriver, MPO
    '''
    assert(params_dict["sys_type"]=="SIAM");
    
    # load data from json
    tl, th, Vg, U, Vb = params_dict["tl"], params_dict["th"], params_dict["Vg"], params_dict["U"], params_dict["Vb"];
    NL, NR = params_dict["NL"], params_dict["NR"];
    Nsites = NL+1+NR;

    # classify site indices (spin not included)
    llead_sites = np.arange(NL);
    rlead_sites = np.arange(NL+1,Nsites);

    # unpack ExprBuilder
    if(block):
        driver, builder = to_add_to;
        if(driver.n_sites != Nsites): raise ValueError;
    else:
        h1e, g2e = to_add_to;
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        if(len(h1e) != Nspinorbs): raise ValueError;

    # REMOVE bias
    for j in llead_sites:
        if(block):
            builder.add_term("cd",[j,j],-Vb/2);
            builder.add_term("CD",[j,j],-Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += -Vb/2;
            h1e[nloc*j+1,nloc*j+1] += -Vb/2;
    for j in rlead_sites:
        if(block):
            builder.add_term("cd",[j,j], Vb/2);
            builder.add_term("CD",[j,j], Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vb/2;
            h1e[nloc*j+1,nloc*j+1] += Vb/2;

    # return
    if(block):
        mpo_from_builder = driver.get_mpo(builder.finalize());
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;

def H_SIETS_builder(params_dict, block, scratch_dir="tmp", verbose=0):
    '''
    Builds the parts of the spin IETS Hamiltonian which apply at all t
    The physical params are contained in a .json file. They are all in eV.
    They are:
    tl (lead hopping), th (lead-impurity hopping),Jz (z component of loc spins
    XXZ exchange), Jx (x comp of loc spins XXZ exchange), Jsd (deloc e's -
    loc spins exchange), Delta (energy of Sdz up - energy of Sdz down)
    Vb (bias between left and right leads. Vb>0 means
    that left lead is higher chem potential than right, leading to
    rightward/positive current), BFM (field to polarize loc spins).

    NL (number sites in left lead), NFM (number of sites in central region
    = number of loc spins), NR (number of sites in right lead)

    NB this system is assumed half-filled, so Ne=Nsites.
    The total Sz of the electrons is always 0, so Ne_up=Ne_down=Ne//2

    NB this builds in terms of supersited dofs, rather than fermionic dofs

    Returns: a tuple of DMRGDriver, ExprBuilder objects
    '''
    assert(params_dict["sys_type"]=="SIETS");

    # load data from json
    tl, th, Jz, Jx, Jsd, Delta, Vb = params_dict["tl"], params_dict["th"], params_dict["Jz"], params_dict["Jx"], params_dict["Jsd"], params_dict["Delta"], params_dict["Vb"];
    NL, NFM, NR = params_dict["NL"], params_dict["NFM"], params_dict["NR"];

    # fermionic sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    Ne=1*Nsites;
    TwoSz = 0; assert(Ne%2 ==0); # need even number of electrons for TwoSz=0
    TwoSz += np.sign(int(params_dict["BFM"]))*NFM; # add imp spin
    if("BFM_first" in params_dict.keys() ): # 1st imp has diff spin
        TwoSz += np.sign(int(params_dict["BFM_first"])) - np.sign(int(params_dict["BFM"])); # add new, remove old

    # impurity spin
    #TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    #TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    #n_fer_dof = 4;
    #n_imp_dof = len(TwoSdz_ladder);
    #assert(TwoSd == 1); # for now, to get degeneracies right

    # classify site indices (spin not included)
    llead_sites = np.arange(NL);
    central_sites = np.arange(NL,NL+NFM);
    rlead_sites = np.arange(NL+NFM,Nsites);
    all_sites = np.arange(Nsites);

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir, symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            # using complex symmetry type, as above, seems linked to
            # Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM warnings
            # but only when TwoSz is input correctly
            # in latter case, we get a floating point exception even when complex sym is turned off!
            driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
            print(">>> driver(n_sites={:.0f}, n_elec={:.0f}, spin={:.0f})".format(Nsites, Ne, TwoSz));
        else: raise NotImplementedError;
        
        # def custom states and operators
        qnumber_wrapper = driver.bw.SX # quantum number wrapper function
        custom_states, custom_ops = get_custom_states_ops(params_dict, qnumber_wrapper);
        # input custom site basis states and ops to driver, and build builder
        driver.ghamil = driver.get_custom_hamiltonian(custom_states, custom_ops)
        builder = driver.expr_builder();
        print("\n",40*"#","\nConstructed builder\n",40*"#","\n");
        
    # end of if(block) code
    else:
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);
    
    # LEAD j <-> j+1 hopping for fermions
    for lead_sites in [llead_sites, rlead_sites]:
        for j in lead_sites[:-1]:
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
                
        
    # lead coupling to impurity 
    if(len(central_sites)==2):
        jpairs = [(llead_sites[-1], central_sites[0]), (central_sites[0], central_sites[1]), (central_sites[-1], rlead_sites[0])];
    elif(len(central_sites)==1):
        jpairs = [(llead_sites[-1], central_sites[0]), (central_sites[-1], rlead_sites[0])];
    else: raise NotImplementedError;
    for jpair in jpairs:
        jlead, jimp = jpair;
        if(block):
            builder.add_term("cd",[jlead,jimp],-th);
            builder.add_term("CD",[jlead,jimp],-th);
            builder.add_term("cd",[jimp,jlead],-th);
            builder.add_term("CD",[jimp,jlead],-th);
        else:
            h1e[nloc*jlead+0,nloc*jimp+0] += -th;
            h1e[nloc*jimp+0,nloc*jlead+0] += -th;
            h1e[nloc*jlead+1,nloc*jimp+1] += -th;
            h1e[nloc*jimp+1,nloc*jlead+1] += -th;            

    # XXZ exchange between neighboring impurities
    for j in central_sites[:-1]:
        if(block):
            builder.add_term("ZZ",[j,j+1],-Jz);
            builder.add_term("PM",[j,j+1],-Jx/2);
            builder.add_term("MP",[j,j+1],-Jx/2);
        else:
            assert(Jz==0.0);
            assert(Jx==0.0);

    # sd exchange between impurities and charge density on their site
    for j in central_sites:
        if(block):
            # z terms
            builder.add_term("cdZ",[j,j,j],-Jsd/2);
            builder.add_term("CDZ",[j,j,j], Jsd/2);
            # plus minus terms
            builder.add_term("cDM",[j,j,j],-Jsd/2);
            builder.add_term("CdP",[j,j,j],-Jsd/2);
        else:
            assert(Jsd==0.0);

    # energy splitting of impurity spin
    for j in central_sites: # Delta and BFM must be same sign so t=0 is gd state
        if(block):
            builder.add_term("Z",[j],-Delta);
        else:
            assert(Delta==0.0);

    # bias (NB this will be REMOVED by polarizer so that it is ABSENT for t<0
    # and PRESENT at t>0) ie opposite to B fields in STT, but still "added"
    # by the polarizer
    for j in llead_sites:
        if(block):
            builder.add_term("cd",[j,j], Vb/2); 
            builder.add_term("CD",[j,j], Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vb/2;
            h1e[nloc*j+1,nloc*j+1] += Vb/2;
    for j in rlead_sites:
        if(block):
            builder.add_term("cd",[j,j],-Vb/2); 
            builder.add_term("CD",[j,j],-Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += -Vb/2;
            h1e[nloc*j+1,nloc*j+1] += -Vb/2;
        
    # special case terms
    if("Vg" in params_dict.keys()):
        Vg = params_dict["Vg"];
        if(block):
            for j in central_sites:
                builder.add_term("cd",[j,j], Vg);
                builder.add_term("CD",[j,j], Vg);
        else:
            for j in central_sites:
                h1e[nloc*j+0,nloc*j+0] += Vg;
                h1e[nloc*j+1,nloc*j+1] += Vg;

    if(block): return driver, builder;
    else: return h1e, g2e;

def H_SIETS_polarizer(params_dict, to_add_to, block, verbose=0):
    '''
    Adds terms specific to the t<0 spin IETS Hamiltonian in which the impurity
    spins are polarized and the bias is removed

    NB this builds in terms of supersited dofs, rather than fermionic dofs

    Args:
    Params_dict: dict containing physical param values, these are defined in Hsys_base
    to_add_to, tuple of objects to add terms to:
        if block is True: these will be DMRGDriver, ExprBuilder objects
        else: these will be 1-body and 2-body parts of the second quantized
        Hamiltonian
        
    Returns: a tuple of DMRGDriver, MPO
    '''
    assert(params_dict["sys_type"]=="SIETS");

    # load data from json
    Jsd, BFM, Vb, th = params_dict["Jsd"], params_dict["BFM"], params_dict["Vb"], params_dict["th"];
    NL, NFM, NR = params_dict["NL"], params_dict["NFM"], params_dict["NR"];

    # fermionic sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain

    # classify site indices (spin not included)
    llead_sites = np.arange(NL);
    central_sites = np.arange(NL,NL+NFM);
    rlead_sites = np.arange(NL+NFM,Nsites);

    # unpack ExprBuilder
    if(block):
        driver, builder = to_add_to;
        if(driver.n_sites != Nsites): raise ValueError;
    else:
        h1e, g2e = to_add_to;
        nloc = 2;
        Nspinorbs = nloc*Nsites;
        if(len(h1e) != Nspinorbs): raise ValueError;

    # REMOVE bias
    for j in llead_sites:
        if(block):
            builder.add_term("cd",[j,j],-Vb/2);
            builder.add_term("CD",[j,j],-Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += -Vb/2;
            h1e[nloc*j+1,nloc*j+1] += -Vb/2;
    for j in rlead_sites:
        if(block):
            builder.add_term("cd",[j,j], Vb/2);
            builder.add_term("CD",[j,j], Vb/2);
        else:
            h1e[nloc*j+0,nloc*j+0] += Vb/2;
            h1e[nloc*j+1,nloc*j+1] += Vb/2;

    # B field on the loc spins
    if(block):
        for j in central_sites:
            builder.add_term("Z",[j],-BFM);
        
    # special case initialization
    if("thquench" in params_dict.keys() and (params_dict["thquench"]>0)):
        # REMOVE lead coupling to impurity at time 0
        thquench = params_dict["thquench"];
        if(len(central_sites)==2):
            jpairs = [(llead_sites[-1], central_sites[0]), (central_sites[0], central_sites[1]), (central_sites[-1], rlead_sites[0])];
        elif(len(central_sites)==1):
            jpairs = [(llead_sites[-1], central_sites[0]), (central_sites[-1], rlead_sites[0])];
        else: raise NotImplementedError;
        for jpair in jpairs:
            jlead, jimp = jpair;
            if(block):
                builder.add_term("cd",[jlead,jimp], th-thquench);
                builder.add_term("CD",[jlead,jimp], th-thquench);
                builder.add_term("cd",[jimp,jlead], th-thquench);
                builder.add_term("CD",[jimp,jlead], th-thquench);
            else:
                h1e[nloc*jlead+0,nloc*jimp+0] +=  th-thquench;
                h1e[nloc*jimp+0,nloc*jlead+0] +=  th-thquench;
                h1e[nloc*jlead+1,nloc*jimp+1] +=  th-thquench;
                h1e[nloc*jimp+1,nloc*jlead+1] +=  th-thquench; 
    if("BFM_first" in params_dict.keys() and len(central_sites)>0): # B field that targets 1st loc spin only
        BFM_first = params_dict["BFM_first"];
        j = central_sites[0];
        if(block):
            builder.add_term("Z",[j], -BFM_first+BFM);
        else: raise NotImplementedError;
    if("B_Heis" in params_dict.keys() and len(central_sites)>0): # prep singlet
        B_Heis = params_dict["B_Heis"];
        if(block):
            for j in central_sites[:-1]:
                builder.add_term("ZZ",[j,j+1],-B_Heis);
                builder.add_term("PM",[j,j+1],-B_Heis/2);
                builder.add_term("MP",[j,j+1],-B_Heis/2);
        else:
            assert(B_Heis==0.0);
                    
    # return
    if(block):
        mpo_from_builder = driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"));
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;

def H_STT_builder(params_dict, block, scratch_dir="tmp", verbose=0):
    '''
    Builds the parts of the STT Hamiltonian which apply at all t
    The physical params are contained in a .json file. They are all in eV.
    They are:
    tl (lead hopping), Vconf (confining voltage depth), Be (field to polarize
    deloc es), BFM (field to polarize loc spins), Jz (z component of exchange
    for loc spins XXZ model), Jx (x component of exchange for loc spins XXZ
    model), Jsd (deloc e's - loc spins exchange)

    NL (number sites in left lead), NFM (number of sites in central region
    = number of loc spins), NR (number of sites in right lead), Nconf (width
    of confining region), Ne (number of electrons), TwoSz (Twice the total Sz
    of the system)

    NB this builds in terms of supersited dofs, rather than fermionic dofs

    Returns: a tuple of DMRGDriver, ExprBuilder objects
    '''
    if(not block): raise NotImplementedError;
    assert(params_dict["sys_type"]=="STT");

    # load data from json
    tl, Jsd = params_dict["tl"], params_dict["Jsd"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];
    Nbuffer = 0;
    if("Nbuffer" in params_dict.keys()): Nbuffer = params_dict["Nbuffer"];

    # fermionic sites and spin
    Nsites = Nbuffer+NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

    # impurity spin
    #TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    #TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    #n_fer_dof = 4;
    #n_imp_dof = len(TwoSdz_ladder);
    #assert(TwoSd == 1); # for now, to get degeneracies right

    # classify site indices (spin not included)
    llead_sites = np.arange(Nbuffer,Nbuffer+NL);
    if("MSQ_spacer" in params_dict.keys()): # MSQs on either end of NFM only
        print("MSQ spacer");
        central_sites = np.array([Nbuffer+NL,Nbuffer+NL+NFM-1]);
    else: # NFM full of MSQs
        central_sites = np.arange(Nbuffer+NL,Nbuffer+NL+NFM);
    rlead_sites = np.arange(Nbuffer+NL+NFM,Nbuffer+Nsites);
    all_sites = np.arange(Nsites);

    # construct ExprBuilder
    if(params_dict["symmetry"] == "Sz"):
        driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir, symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4) #None, mpi=True);
        # using complex symmetry type, as above, seems linked to
        # Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM warnings
        # but only when TwoSz is input correctly
        # in latter case, we get a floating point exception even when complex sym is turned off!
        #driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ, n_threads=4)
        driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
    else: raise NotImplementedError;

    # ###############
    print("\t driver.mpi = ",driver.mpi);
    print("\t global threads = {:.0f}".format(driver.bw.b.Global.threading.n_threads_global));

    # def custom states and operators
    qnumber_wrapper = driver.bw.SX # quantum number wrapper function
    custom_states, custom_ops = get_custom_states_ops(params_dict, qnumber_wrapper);
    # input custom site basis states and ops to driver, and build builder
    driver.ghamil = driver.get_custom_hamiltonian(custom_states, custom_ops)
    builder = driver.expr_builder();
    print("\n",40*"#","\nConstructed builder\n",40*"#","\n");

    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        print("j={:.0f} -> tl={:.2f}".format(j,tl));
        builder.add_term("cd",[j,j+1],-tl); 
        builder.add_term("CD",[j,j+1],-tl);
        builder.add_term("cd",[j+1,j],-tl);
        builder.add_term("CD",[j+1,j],-tl);

    # XXZ exchange between neighboring impurities
    if("Jz" in params_dict.keys() and "Jx" in params_dict.keys()):
        Jz, Jx = params_dict["Jz"], params_dict["Jx"];
        if("MSQ_spacer" in params_dict.keys()):
            raise NotImplementedError("code assumes MSQs are neighbors");
        else:
            for j in central_sites[:-1]:
                builder.add_term("ZZ",[j,j+1],-Jz);
                builder.add_term("PM",[j,j+1],-Jx/2);
                builder.add_term("MP",[j,j+1],-Jx/2);

    # sd exchange between impurities and charge density on their site
    for j in central_sites:
        # z terms
        builder.add_term("cdZ",[j,j,j],-Jsd/2);
        builder.add_term("CDZ",[j,j,j], Jsd/2);
        # plus minus terms
        builder.add_term("cDM",[j,j,j],-Jsd/2);
        builder.add_term("CdP",[j,j,j],-Jsd/2);
        #pass;
        
    # special case terms
    if("tunnel" in params_dict.keys()):
        raise NotImplementedError;
        # tunnel barrier on last few sites of left lead
        tunnel = params_dict["tunnel"];
        tunnel_size = 3; # last tunnel_size sites
        assert(NL-Nconf>tunnel_size);
        for j in llead_sites[-tunnel_size:]:
            builder.add_term("cd",[j,j],tunnel);
            builder.add_term("CD",[j,j],tunnel);
    if("Vdelta" in params_dict.keys()):
        # voltage gradient from site to site, ie constant applied Electric field
        Vdelta = params_dict["Vdelta"];
        for j in all_sites:
            builder.add_term("cd",[j,j],-Vdelta*j);
            builder.add_term("CD",[j,j],-Vdelta*j);
    if("tp" in params_dict.keys()):
        # different hopping (t') in confinement region 
        tp = params_dict["tp"];
        conf_sites = np.arange(Nconf);
        tp_symmetry = 1;
        if("tp_symmetry" in params_dict.keys()): tp_symmetry = params_dict["tp_symmetry"];
        assert(tp_symmetry in [1,0]);
        if(tp_symmetry==1): # different hopping for first AND last Nconf sites (preserves left-right symmetry)
            assert(NL+NFM+NR - Nconf > Nconf);
            anticonf_sites = np.arange(NL+NFM+NR - Nconf, NL+NFM+NR);
        else:
            anticonf_sites = np.array([]);
        for j in conf_sites:
            print("j={:.0f} -> tp={:.2f}".format(j,tp));
            builder.add_term("cd",[j,j+1],tl-tp); # <-- replace tl with tp
            builder.add_term("CD",[j,j+1],tl-tp);
            builder.add_term("cd",[j+1,j],tl-tp);
            builder.add_term("CD",[j+1,j],tl-tp);
        for j in (anticonf_sites-1):
            print("j={:.0f} -> tp={:.2f}".format(j,tp));
            builder.add_term("cd",[j,j+1],tl-tp); # <-- replace tl with tp
            builder.add_term("CD",[j,j+1],tl-tp);
            builder.add_term("cd",[j+1,j],tl-tp);
            builder.add_term("CD",[j+1,j],tl-tp);     

    return driver, builder;

def H_STT_polarizer(params_dict, to_add_to, block, verbose=0):
    '''
    Adds terms specific to the t<0 STT Hamiltonian in which the deloc e's, loc spins are
    confined and polarized by application of external fields Be, BFM

    NB this builds in terms of supersited dofs, rather than fermionic dofs

    Args:
    Params_dict: dict containing physical param values, these are defined in Hsys_base
    to_add_to, tuple of objects to add terms to:
        if block is True: these will be DMRGDriver, ExprBuilder objects
        else: these will be 1-body and 2-body parts of the second quantized
        Hamiltonian
        
    Returns: a tuple of DMRGDriver, MPO
    '''
    if(not block): raise NotImplementedError;
    assert(params_dict["sys_type"]=="STT");

    # load data from json
    tl, Vconf, Be, BFM = params_dict["tl"], params_dict["Vconf"], params_dict["Be"], params_dict["BFM"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];
    Nbuffer = 0;
    if("Nbuffer" in params_dict.keys()): Nbuffer = params_dict["Nbuffer"];

    # fermionic sites and spin
    Nsites = Nbuffer+NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

    # classify site indices (spin not included)
    llead_sites = np.arange(Nbuffer,Nbuffer+NL);
    if("MSQ_spacer" in params_dict.keys()): # MSQs only at either end of NFM
        central_sites = np.array([Nbuffer+NL,Nbuffer+NL+NFM-1]);
    else: # NFM full of MSQs
        central_sites = np.arange(Nbuffer+NL,Nbuffer+NL+NFM);
    rlead_sites = np.arange(Nbuffer+NL+NFM,Nbuffer+Nsites);
    all_sites = np.arange(Nsites);
    conf_sites = np.arange(Nbuffer,Nbuffer+Nconf);

    # unpack ExprBuilder
    driver, builder = to_add_to;
    if(driver.n_sites != Nsites): raise ValueError;
    
    # confining potential in left lead
    for j in conf_sites:
        builder.add_term("cd",[j,j],-Vconf); 
        builder.add_term("CD",[j,j],-Vconf);

    # eigenstates of the confined, spinless, t<0 system
    if("Bstate" in params_dict.keys()):
        Bstate = params_dict["Bstate"];
        h1e_t0 = np.zeros((Nsites,Nsites),dtype=float);
        nloc = 1; # spinless
        # j <-> j+1 hopping for fermions
        for j in all_sites[:-1]:
            h1e_t0[nloc*j+0,nloc*(j+1)+0] += -tl; # spinless
            h1e_t0[nloc*(j+1)+0,nloc*j+0] += -tl;
        # confinement
        for j in conf_sites:
            h1e_t0[nloc*j+0,nloc*j+0] += -Vconf; # spinless!
        # t<0 eigenstates (|k_m> states)
        vals_t0, vecs_t0 = np.linalg.eigh(h1e_t0);
        vecs_t0 = vecs_t0.T;
        # now we have the eigenstates that span the confined well at t<0
        # use them to apply B field *directly to these states*
        how_many_states = params_dict["Bstate_num"]; # we block only lowest (this number) of states
        for kmvali in range(how_many_states): #iter over states
            for j in all_sites:
                for jp in all_sites:
                    builder.add_term("cd",[j,jp], Bstate*vecs_t0[kmvali,j]*np.conj(vecs_t0[kmvali,jp]));
                    builder.add_term("CD",[j,jp], Bstate*vecs_t0[kmvali,j]*np.conj(vecs_t0[kmvali,jp]));

    # B field in the confined region ----------> ASSUMED IN THE Z
    # only within the region of confining potential
    for j in conf_sites:
        builder.add_term("cd",[j,j],-Be/2);
        builder.add_term("CD",[j,j], Be/2);

    # B field on the loc spins
    for j in central_sites:
        builder.add_term("Z",[j],-BFM);

    # special case initialization
    if("BFM_first" in params_dict.keys() and len(central_sites)>0): # B field that targets 1st loc spin only
        BFM_first = params_dict["BFM_first"];
        j = central_sites[0];
        builder.add_term("Z",[j], -BFM_first+BFM);
    if("Bsd" in params_dict.keys() and len(central_sites)>0): # B field on the j that couples to the first loc spin
        Bsd = params_dict["Bsd"];
        j = central_sites[0];
        builder.add_term("cd",[j,j],-Bsd/2);
        builder.add_term("CD",[j,j], Bsd/2);
    if("Bent" in params_dict.keys() and len(central_sites)==2): # B field that entangles 2 loc spins
        Bent = params_dict["Bent"];
        if("MSQ_spacer" in params_dict.keys()): # MSQs at each end of NFM only
            jpairs = [central_sites];
        else: # NFM full of MSQs
            jpairs = [];
            for j in central_sites[:-1]: jpairs.append([j,j+1]);
        for jpair in jpairs: # jpair is a list of two sites to entangle
            if(not ("triplet_flag" in params_dict.keys())):
                print("no triplet flag")
                builder.add_term("ZZ",jpair,-Bent);
            else: print("triplet flag");
            builder.add_term("PM",jpair,-Bent/2);
            builder.add_term("MP",jpair,-Bent/2);
    if("DSz2" in params_dict.keys()):
        # hard z-axis for cases where Bent is applied
        raise NotImplementedError("pointless for s-1/2 systems");
        DSz2 = params_dict["DSz2"];
        for j in central_sites:
            builder.add_term("ZZ",[j,j],DSz2);
    if("Bx" in params_dict.keys()): # B in the x direction, w/in the confining region
        Bx = params_dict["Bx"];
        for j in conf_sites:
            builder.add_term("cD",[j,j],-Bx/2);
            #builder.add_term("Cd",[j,j],-Bx/2);
    if("Vdelta" in params_dict.keys()):
        # REMOVE the voltage gradient for time < 0, so it doesn't affect initialization
        Vdelta = params_dict["Vdelta"];
        for j in all_sites:
            builder.add_term("cd",[j,j],Vdelta*j);
            builder.add_term("CD",[j,j],Vdelta*j);

    # return
    mpo_from_builder = driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"));
    return driver, mpo_from_builder;
    
def get_custom_states_ops(params_dict, qnumber):
    '''
    Args:
    params_dict, json of all physical and numerical parameters
    qnumber, quantum number wrapper function
    returns: tuple of
    site_states--a list of quantum states, defined by good quantum numbers Ne and TwoSz
    site_ops--a list of 2nd quantized ops which act on the states
    '''
    
    # load data from json
    NL, NFM, NR = params_dict["NL"], params_dict["NFM"], params_dict["NR"];
    Nsites = NL+NFM+NR; 

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);
    assert(TwoSd == 1); # for now, to get degeneracies right

    # classify site indices (spin not included)
    if(params_dict["sys_type"] in ["SIAM_RM", "SIETS_RM"]): block2site = 2;
    else: block2site = 1;
    llead_sites = np.arange(NL*block2site);
    if("MSQ_spacer" in params_dict.keys()): # MSQs on either end of NFM only
        print("MSQ spacer");
        central_sites = np.array([NL*block2site,(NL+NFM-1)*block2site]);
    else: # NFM full of MSQs
        central_sites = np.arange(NL*block2site,(NL+NFM)*block2site);
    rlead_sites = np.arange((NL+NFM)*block2site,Nsites*block2site);
    all_sites = np.arange(Nsites*block2site);
    
    
    # Szd blocks for fermion-impurity operators
    # squares are diagonal blocks and triangles are one off diagonal
    squar_I = np.eye(n_fer_dof); # identity - for basis see states below
    squar_c = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]); # c_up^\dagger
    squar_d = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]); # c_up
    squar_C = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0,-1, 0, 0]]); # c_down^\dagger
    squar_D = np.array([[0, 0, 1, 0], [0, 0, 0,-1], [0, 0, 0, 0], [0, 0, 0, 0]]); # c_down

    # construct 4d ops from blocks
    # fermion ops 
    fourd_base = np.zeros((n_imp_dof,n_imp_dof,n_fer_dof,n_fer_dof),dtype=float);
    fourd_c = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof): fourd_c[Sdz_index,Sdz_index] = np.copy(squar_c);
    fourd_d = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof): fourd_d[Sdz_index,Sdz_index] = np.copy(squar_d);
    fourd_C = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof): fourd_C[Sdz_index,Sdz_index] = np.copy(squar_C);
    fourd_D = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof): fourd_D[Sdz_index,Sdz_index] = np.copy(squar_D);
    # Sd ops 
    fourd_Sdz = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof): fourd_Sdz[Sdz_index,Sdz_index] = (TwoSdz_ladder[Sdz_index]/2)*np.eye(n_fer_dof);
    print("TwoSdz_ladder =\n",TwoSdz_ladder);
    print("four_Sdz = \n",reblock(fourd_Sdz))
    fourd_Sdminus = np.copy(fourd_base);
    fourd_Sdplus = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof-1): 
        fourd_Sdminus[Sdz_index+1,Sdz_index] = np.sqrt(0.5*TwoSd*(0.5*TwoSd+1)-0.5*TwoSdz_ladder[Sdz_index]*(0.5*TwoSdz_ladder[Sdz_index]-1))*np.eye(n_fer_dof);
        fourd_Sdplus[Sdz_index,Sdz_index+1] = np.sqrt(0.5*TwoSd*(0.5*TwoSd+1)-0.5*TwoSdz_ladder[Sdz_index+1]*(0.5*TwoSdz_ladder[Sdz_index+1]+1))*np.eye(n_fer_dof);
    print("four_Sdminus = \n",reblock(fourd_Sdminus))
    print("four_Sdplus = \n",reblock(fourd_Sdplus))

    # def custom states and operators
    site_states, site_ops = [], [];
    # quantum numbers here: nelec, TwoSz, TwoSdz
    # Sdz is z projection of impurity spin: ladder from +s to -s
    for sitei in all_sites:
        if(sitei not in central_sites): # regular fermion dofs
            states = [(qnumber(0, 0,0),1), # |> # (always obey n_elec and TwoSz symmetry)
                      (qnumber(1, 1,0),1), # |up> #<--
                      (qnumber(1,-1,0),1), # |down>
                      (qnumber(2, 0,0),1)];# |up down>
            ops = { "":np.copy(squar_I), # identity
                   "c":np.copy(squar_c), # c_up^\dagger 
                   "d":np.copy(squar_d), # c_up
                   "C":np.copy(squar_C), # c_down^\dagger
                   "D":np.copy(squar_D)} # c_down
        elif(sitei in central_sites): # has fermion AND impurity dofs
            states = [];
            nelec_dofs, spin_dofs = [0,1,1,2], [0,1,-1,0];
            qnumber_degens = {};
            for fer_dofi in range(len(nelec_dofs)):
                for TwoSdz in TwoSdz_ladder:
                    pass; # TODO: create qnumber_degens here
            qnumber_degens = {(0, 1,0):1,
                              (0,-1,0):1,
                              (1, 2,0):1,
                              (1, 0,0):2,
                              (1,-2,0):1,
                              (2, 1,0):1,
                              (2,-1,0):1};
            qnumbers_added = {};
            for fer_dofi in range(len(nelec_dofs)):
                for TwoSdz in TwoSdz_ladder:
                    qnumber_tup = (nelec_dofs[fer_dofi],spin_dofs[fer_dofi]+TwoSdz,0);
                    if(qnumber_tup in qnumber_degens and qnumber_tup not in qnumbers_added):
                        print(">>>",qnumber_tup)
                        states.append((qnumber(*qnumber_tup),qnumber_degens[qnumber_tup]));         
                        qnumbers_added[qnumber_tup] = 1;
            # ops dictionary
            ops = { "":np.eye(n_fer_dof*n_imp_dof), # identity
                   "c":reblock(fourd_c), # c_up^\dagger
                   "d":reblock(fourd_d), # c_up
                   "C":reblock(fourd_C), # c_down^\dagger
                   "D":reblock(fourd_D), # c_down
                   "Z":reblock(fourd_Sdz)    # Sz of impurity
                   ,"P":reblock(fourd_Sdplus) # S+ on impurity
                   ,"M":reblock(fourd_Sdminus) # S- on impurity
                    }
        else:
            raise Exception("Site i = ",sitei," never caught");
        site_states.append(states);
        site_ops.append(ops);
        
    print(">>> Custom operators at ",central_sites);
    return site_states, site_ops;

