'''
Christian Bunker
M^2QM at UF
June 2021

tddmrg/wrappers.py

use dmrg for time evol of model ham systems
- single impurity anderson model

pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr

'''

import numpy as np
import time

#################################################
#### get current data

def SiamData(nleads, nelecs, ndots, timestop, deltat, phys_params, bond_dims, noises, 
spinstate = "", prefix = "data/", namevar = "Vg", verbose = 0) -> str:
    '''
    Walks thru all the steps for plotting current thru a SIAM, using DMRG for equil state
    and td-DMRG for nonequilibirum dynamics. Impurity is a quantum dot w/ gate voltage, hubbard U
    - construct the eq hamiltonian, 1e and 2e parts, as np arrays
    - store eq ham in FCIDUMP object which allows us to access it w/ pyblock3
    - from FCIDUMP create a pyblock3.hamiltonian.Hamiltonian object\
    - use this to build a Matrix Product Operator (MPO) and initial guess MPS
    - use these to construct Matrix Product Expectation (MPE) which calls dmrg() to get gd state
    - construct noneq ham (thyb = 1e-5 -> 0.4 default) and repeat to get MPE (in td_dmrg module)
    - then MPE.tddmrg() method updates wf in time and we can get observables (in td_dmrg module)
    	NB tddmrg uses max bonddim of dmrg as of now

    Args:
    nleads, tuple of ints of left lead sites, right lead sites
    nelecs, tuple of num up e's, 0 due to ASU formalism
    ndots, int, number of dots in impurity
    timestop, float, how long to run for
    deltat, float, time step increment
    bond_dims, list of increasing bond dim over dmrg sweeps, optional
    noises, list of decreasing noises over dmrg sweeps, optional
    physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta, phi
    	if None, gives defaults vals for all (see below)
    prefix: assigns prefix (eg folder) to default output file name

    Returns:
    name of observables vs t data file
    '''
    from transport import tddmrg, fci_mod
    from transport.fci_mod import ops, ops_dmrg
    from pyblock3 import fcidump, hamiltonian
    from pyblock3.algebra.mpe import MPE

    # check inputs
    if(not isinstance(nleads, tuple) ): raise TypeError;
    if(not isinstance(nelecs, tuple) ): raise TypeError;
    if(not isinstance(ndots, int) ): raise TypeError;
    if(not bond_dims[0] <= bond_dims[-1]): raise ValueError; # bdims must have increasing behavior 
    if(not noises[0] >= noises[-1] ): raise ValueError; # noises must have decreasing behavior 

    # set up the hamiltonian
    imp_i = [nleads[0]*2, nleads[0]*2 + 2*ndots - 1 ]; # imp sites, inclusive
    norbs = 2*(nleads[0]+nleads[1]+ndots); # num spin orbs
    # nelecs left as tunable
    t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, B, theta = phys_params;


    # get h1e and h2e for siam, h_imp = h_dot
    if(verbose): print("1. Construct hamiltonian")
    ham_params = t_leads, 1e-5, t_dots, 0.0, mu, V_gate, U, B, theta; # thyb, Vbias turned off, mag field in theta direction
    h1e, g2e, input_str = fci_mod.ops.dot_hams(nleads, ndots, ham_params, spinstate, verbose = verbose);

    # store physics in fci dump object
    hdump = fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); # twos = 2S tells spin    

    # instead of np array, dmrg wants ham as a matrix product operator (MPO)
    h_obj = hamiltonian.Hamiltonian(hdump,flat=True);
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff=1E-15); # compressing saves memory
    if verbose: print("- Built H as compressed MPO: ", h_mpo.show_bond_dims() );

    # initial ansatz for wf, in matrix product state (MPS) form
    psi_mps = h_obj.build_mps(bond_dims[0]);
    E_mps_init = ops_dmrg.compute_obs(h_mpo, psi_mps);
    if verbose: print("- Initial gd energy = ", E_mps_init);

    # ground-state DMRG
    # runs thru an MPE (matrix product expectation) class built from mpo, mps
    if(verbose): print("2. DMRG solution");
    MPE_obj = MPE(psi_mps, h_mpo, psi_mps);

    # solve system by doing dmrg sweeps
    # MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
    # can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
    # noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
    dmrg_obj = MPE_obj.dmrg(bdims=bond_dims, noises = noises, tol = 1e-8, iprint=0);
    E_dmrg = dmrg_obj.energies;
    if verbose: print("- Final gd energy = ", E_dmrg[-1]);

    # nonequil hamiltonian (as MPO)
    if(verbose > 2 ): print("- Add nonequilibrium terms");
    ham_params_neq = t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, 0.0, 0.0; # thyb and Vbias on, no zeeman splitting
    h1e_neq, g2e_neq, input_str_neq = fci_mod.ops.dot_hams(nleads, ndots, ham_params_neq, "", verbose = verbose);
    hdump_neq = fcidump.FCIDUMP(h1e=h1e_neq,g2e=g2e_neq,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); 
    h_obj_neq = hamiltonian.Hamiltonian(hdump_neq, flat=True);
    h_mpo_neq = h_obj_neq.build_qc_mpo(); # got mpo
    h_mpo_neq, _ = h_mpo_neq.compress(cutoff=1E-15); # compression saves memory

    try:
        from pyblock3.algebra import flat
        assert( isinstance(h_obj_neq.FT, flat.FlatFermionTensor) );
        assert( isinstance(h_mpo_neq, flat.FlatFermionTensor) );
    except:
        pass;

    # time propagate the noneq state
    # td dmrg uses highest bond dim
    if(verbose): print("3. Time propagation");
    observables = tddmrg.kernel(h1e, g2e, h1e_neq, nelecs, bond_dims, timestop, deltat, verbose = verbose);

    # write results to external file
    if namevar == "Vg":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vg"+str(V_gate)+".npy";
    elif namevar == "U":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_U"+str(U)+".npy";
    elif namevar == "Vb":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vb"+str(V_bias)+".npy";
    elif namevar == "th":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_th"+str(t_hyb)+".npy";
    else: assert(False); # invalid option
    hstring = time.asctime(); # header has lots of important info: phys params, bond dims, etc
    hstring += "\ntf = "+str(timestop)+"\ndt = "+str(deltat);
    hstring += "\nASU formalism, t_hyb noneq. term, td-DMRG,\nbdims = "+str(bond_dims)+"\n noises = "+str(noises); 
    hstring += "\nEquilibrium"+input_str; # write input vals to txt
    hstring += "\nNonequlibrium"+input_str_neq;
    np.savetxt(fname[:-4]+".txt", observables[0], header = hstring); # saves info to txt
    np.save(fname, observables);
    if(verbose): print("4. Saved data to "+fname);
    
    return fname;

def get_h1e(n_mols,s_mols,spatial_orbs,mytm, myB_mm, myB_elec, myJH, myJK, my_chiral, verbose = 0):
    '''
    make the 1body and 2body parts of the 2nd qu'd ham
    The Ham is Kumar 2017 Eqs (1)-(3)
    We only have 1 electron so all the interactions are 1body
    Electron has n_loc_dof channels which account for mol spin dofs as well
    Args:
    -mytm, hopping between mols
    -myB_mm, zeeman strength for the mol spins. Contains g*\mu_B !
    -myB_elec, zeeman strength for the electron. Contains g*\mu_B !
    -myJH, Heisenberg exchange between mol spins
    -myJK, Kondo exchange between elec and mol spins
    -my_chiral, strength of chiral operator acting on mol spins,
       should benonzero to lift chiral degeneracy, which avoids numerical issues
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    
    # return var
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    n_loc_dof = 2*len(mol_states);
    h1e = np.zeros((spatial_orbs, spatial_orbs, n_loc_dof, n_loc_dof),dtype=complex);

    #### fermionic terms

    # hacky code to break fermion spatial symetry
    for loci in range(n_loc_dof):
        h1e[1,1,loci,loci] += -0.0;

    # spin-independent hopping between n.n. sys orbs
    for sysi in range(spatial_orbs-1):
        # iter over local dofs (up, down, etc)
        for loci in range(n_loc_dof):
            h1e[sysi,sysi+1,loci,loci] += -mytm; 
            h1e[sysi+1,sysi,loci,loci] += -mytm;
    if(spatial_orbs > 2): # last to first hopping
        for loci in range(n_loc_dof): 
            h1e[0,-1,loci,loci] += -mytm; 
            h1e[-1,0,loci,loci] += -mytm;

    # fermionic Zeeman
    for sysi in range(spatial_orbs):
        h1e += myB_elec*get_sigz(n_loc_dof, spatial_orbs, sysi);

    #### spin terms

    # chiral breaking
    h1e += my_chiral*get_chiral_op(n_mols,s_mols,spatial_orbs);

    # spin Zeeman terms
    if(verbose): print("Zeeman"); 
    for sysi in range(spatial_orbs):
        # have to iter over local dofs paticle-by-particle
        # iter over all (2s+1)^n_mols many-body mol spin states
        for mol_statei in range(len(mol_states)):
            Sztot = sum(mol_states[mol_statei]);
            # iter over electron spin 
            for sigma in range(2):
                loci = 2*mol_statei+sigma;
                h1e[sysi,sysi,loci,loci] += myB_mm*Sztot;
            if(verbose>1 and sysi==0): print("->",2*mol_statei,mol_states[mol_statei],'->',Sztot);

    # Heisenberg - regardless of elec location, couples mol spins
    if(verbose): print("Heisenberg"); 
    for mola in range(n_mols):
        for molb in range(n_mols):
            if(molb-mola==1 or (mola==n_mols-1 and molb==0)): # nn only
                h1e += myJH*get_SaSb(n_mols, s_mols, spatial_orbs, mola, molb, verbose=verbose);

    # Kondo exchange - couples elec to molecule it is on
    if(n_mols != spatial_orbs): assert(myJK == 0); raise Exception; return h1e;
    if(verbose): print("Kondo");
    for mola in range(n_mols):
        h1e += myJK*get_SaSigb(n_mols, s_mols, spatial_orbs, mola, mola, verbose=verbose);

    # return
    return h1e;
 

def get_SaSb(n_mols,s_mols,spatial_orbs,aindex,bindex, verbose = 0):
    '''
    Get the operator mol spin S_a dotted into mol spin S_b
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    
    # construct as 4d in the spatial orbs, mol_states basis
    SaSb = np.zeros((spatial_orbs,spatial_orbs,2*len(mol_states),2*len(mol_states)));

    # iter over many-body mol spin states twice
    for mol_statei in range(len(mol_states)):
        for mol_statej in range(len(mol_states)):
            # difference between states
            n_different = np.count_nonzero(mol_states[mol_statei]-mol_states[mol_statej]);
            if(n_different in [0,2]):
                               
                # quantum numbers
                Szi_a = mol_states[mol_statei][aindex];
                Szi_b = mol_states[mol_statei][bindex];
                Szj_a = mol_states[mol_statej][aindex];
                Szj_b = mol_states[mol_statej][bindex];

                # S^z_a S^z_b - couples state to itself
                if(mol_statei == mol_statej):
                    # add term to diag of all spatial blocks
                    for spacei in range(spatial_orbs):
                        # for both elec spins
                        for sigma in [0,1]:
                            SaSb[spacei,spacei,2*mol_statei+sigma,2*mol_statej+sigma] += Szi_a*Szi_b;

                # S^+_a S^-_b couples spin flipped states
                if(Szi_a - Szj_a==1 and Szi_b-Szj_b==-1):
                    # add term to diag of all spatial blocks
                    for spacei in range(spatial_orbs):
                        if(verbose>1 and spacei==0): print("-> S_"+str(aindex)+" S_"+str(bindex)); print("->",2*mol_statei,mol_states[mol_statei],"->",2*mol_statej,mol_states[mol_statej],'->',1/2); 
                        # for both elec spins
                        for sigma in [0,1]:
                            SaSb[spacei,spacei,2*mol_statei+sigma,2*mol_statej+sigma] += (1/2);
                            # hc
                            SaSb[spacei,spacei,2*mol_statej+sigma,2*mol_statei+sigma] += (1/2);

    # return                       
    return SaSb;

def get_SaSigb(n_mols, s_mols, spatial_orbs, aindex, bindex, verbose=0):
    '''
    Get the operator mol spin S_a dotted into elec spin sigma on site b
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    
    # construct as 4d in the spatial orbs, mol_states basis
    SaSigb = np.zeros((spatial_orbs,spatial_orbs,2*len(mol_states),2*len(mol_states)));

    # iter over many-body mol spin states twice
    for mol_statei in range(len(mol_states)):
        for mol_statej in range(len(mol_states)):
            # difference between states
            n_different = np.count_nonzero(mol_states[mol_statei]-mol_states[mol_statej]);
            if(n_different in [0,1]):
                
                # S^z - couples state to itself
                if(mol_statei == mol_statej):
                    for sigma in range(2):
                        loci = 2*mol_statei+sigma;
                        SaSigb[aindex,bindex,loci,loci] += mol_states[mol_statei][aindex]*(1/2-sigma);
                        if(verbose>1 and aindex==0 and bindex==0): print("-> S_"+str(aindex)+" \sigma_"+str(bindex)); print("->",loci,mol_states[mol_statei],(1/2-sigma),'->',mol_states[mol_statei][aindex]*(1/2-sigma));

                # S^+ - couples statei to statej with moli flipped up by one
                if(mol_states[mol_statei][aindex]+1 == mol_states[mol_statej][aindex]):
                    # all other have to be the same
                    if(n_different == 1):
                        # couple statei with elec up to statej wth elec down
                        SaSigb[aindex,bindex,2*mol_statei,2*mol_statej+1] += (1/2);
                        # hc
                        SaSigb[aindex,bindex,2*mol_statej+1,2*mol_statei] += (1/2);
                        if(verbose>1 and aindex==0 and bindex==0): print("-> S_"+str(aindex)+" \sigma_"+str(bindex)); print("->",2*mol_statei+0,mol_states[mol_statei],0.5,"->",2*mol_statej+1,mol_states[mol_statej],-0.5,'->',1/2);                   

    # return
    return SaSigb;

def get_chiral_op(n_mols,s_mols, spatial_orbs):
    '''
    get the operator S_1 \cdot (S_2 \times S_3)
    '''
    assert n_mols == 3;
    assert s_mols == 1/2;
    hilbert_space = int((2*s_mols+1)**n_mols);
    chiral_op = np.zeros((hilbert_space,hilbert_space),dtype=complex);

    # define spin ops
    Sx = (1/2)*np.array([[0,1],[1,0]],dtype=complex);
    Sy = (1/2)*np.array([[0,-complex(0,1)],[complex(0,1),0]],dtype=complex);
    Sz = (1/2)*np.array([[1,0],[0,-1]],dtype=complex);

    # add in tensor products
    S2yS3z = fci_mod.mat_4d_to_2d(np.tensordot(Sy,Sz,axes=0));
    S2zS3y = fci_mod.mat_4d_to_2d(np.tensordot(Sz,Sy,axes=0));
    cross_x = fci_mod.mat_4d_to_2d(np.tensordot(Sx,S2yS3z-S2zS3y,axes=0));
    S2xS3z = fci_mod.mat_4d_to_2d(np.tensordot(Sx,Sz,axes=0));
    S2zS3x = fci_mod.mat_4d_to_2d(np.tensordot(Sz,Sx,axes=0));
    cross_y = fci_mod.mat_4d_to_2d(np.tensordot(Sy,S2xS3z-S2zS3x,axes=0));
    S2xS3y = fci_mod.mat_4d_to_2d(np.tensordot(Sx,Sy,axes=0));
    S2yS3x = fci_mod.mat_4d_to_2d(np.tensordot(Sy,Sx,axes=0));
    cross_z = fci_mod.mat_4d_to_2d(np.tensordot(Sz,S2xS3y-S2yS3x,axes=0));
    chiral_op += cross_x - cross_y + cross_z;

    # convert from (2*s_mols+1)^n_mols dimensional to full dimensionality
    chiral_op_sigma = np.zeros((spatial_orbs,spatial_orbs,2*hilbert_space,2*hilbert_space),dtype=complex);
    # iter over spatial
    for spacei in range(spatial_orbs):
        # iter over mol dimensionality
        for oldi in range(hilbert_space):
            for oldj in range(hilbert_space):
                # add in spin block
                for sigma in [0,1]:
                    chiral_op_sigma[spacei,spacei,2*oldi+sigma,2*oldj+sigma] += chiral_op[oldi,oldj];

    # return
    return chiral_op_sigma;

def get_Stot2(n_mols,s_mols, spatial_orbs):
    '''
    Get the total spin operator squared
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    
    # construct as 4d in the spatial orbs, mol_states basis
    Stot2 = np.zeros((spatial_orbs,spatial_orbs,2*len(mol_states),2*len(mol_states)));

    # add mol spins squared
    S1 = get_SaSb(s_mols, n_mols, spatial_orbs, 0, 0)
    Stot2 += np.matmul(S1,S1);
    S2 = get_SaSb(s_mols, n_mols, spatial_orbs, 1, 1)
    Stot2 += np.matmul(S2,S2);
    S3 = get_SaSb(s_mols, n_mols, spatial_orbs, 2, 2)
    Stot2 += np.matmul(S3,S3);

    # add in unique spin-spin correlation
    for mola in range(n_mols):
        for molb in range(n_mols):
            if(mola < molb):
                Stot2 += get_SaSb(n_mols,s_mols,spatial_orbs,mola,molb);

    

   
#################################################
#### exec code

if(__name__ == "__main__"):

    pass;


