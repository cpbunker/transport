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
import numpy as np

    
##########################################################################################################
#### driver of time propagation

def kernel(h1e, g2e, h1e_neq, nelecs, bdims, tf, dt, verbose = 0):
    '''
    Drive time prop for dmrg
    Use real time time dependent dmrg method outlined here:
    https://pyblock3.readthedocs.io/en/latest/Documentation/rttddmrg.html

    Args:
    -h1e, ndarray, 1body 2nd qu'd ham
    -g2e, ndarray, 2body 2nd qu'd ham
    -h1e_neq, ndarray, 1body ham that drives time evol, e.g. turns on hopping
    -nelecs, tuple, number of up, down electrons
    -bdims, ndarray, bond dimension of the DMRG solver
    -tf, float, the time to end the time evolution at
    -dt, float, the time step of the time evolution
    '''
    if(not isinstance(h1e, np.ndarray)): raise TypeError;
    if(not isinstance(g2e, np.ndarray)): raise TypeError;
    if(not isinstance(nelecs, tuple)): raise TypeError;
    if(not isinstance(bdims, list)): raise TypeError;
    if(not bdims[0] <= bdims[-1]): raise ValueError; # bdims must have increasing behavior 

    # unpack
    if(verbose): print("1. Hamiltonian\n-h1e = \n",h1e);
    norbs = len(h1e); # n fermion orbs
    nsteps = 1+int(tf/dt+1e-6); # n time steps
    sites = np.array(range(norbs)).reshape(norbs//2,2); #sites[i] gives a list of fermion orb indices spanning spin space
    
    # convert everything to matrix product form
    if(verbose): print("2. DMRG solution");
    h_obj, h_mpo, psi_init = fci_mod.arr_to_mpo(h1e, g2e, nelecs, bdims[0]);
    if verbose: print("- built H as compressed MPO: ", h_mpo.show_bond_dims() );
    E_init = ops_dmrg.compute_obs(h_mpo, psi_init);
    if verbose: print("- guessed gd energy = ", E_init);

    # solve ham with DMRG
    dmrg_mpe = MPE(psi_init, h_mpo, psi_init);
    # MPE.dmrg method controls bdims,noises, n_sweeps,conv tol (tol),verbose (iprint)
    # noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
    dmrg_obj = dmrg_mpe.dmrg(bdims=bdims, tol = 1e-8, iprint=0);
    if verbose: print("- variational gd energy = ", dmrg_obj.energies[-1]);

    # return vals
    obs_gen = 2; # time, E
    obs_per_site = 2; # occ, Sz
    observables = np.empty((nsteps,obs_gen+obs_per_site*len(sites)), dtype=complex); 

    # mpos for observables
    obs_mpos = [];
    for site in sites: # site specific observables
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.occ(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sz(site, norbs) ) );

    # time evol
    _, h_mpo_neq, _ = fci_mod.arr_to_mpo(h1e_neq, g2e, nelecs, bdims[0]);
    dmrg_mpe_neq = MPE(psi_init, h_mpo_neq, psi_init); # must be built with initial state!
    if(verbose): print("3. Time evolution\n-h1e_neq = \n",h1e_neq);
    for ti in range(nsteps):
        if(verbose>2): print("-time: ", ti*dt);

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        E_t = dmrg_mpe_neq.tddmrg(bdims,-np.complex(0,dt), n_sweeps = 1, iprint=0, cutoff = 0).energies
        psi_t = dmrg_mpe_neq.ket; # update wf

        # compute observables
        observables[ti,0] = ti*dt; # time
        observables[ti,1] = E_t[-1]; # energy
        for mi in range(len(obs_mpos)): # iter over mpos
            observables[ti,obs_gen+mi] = ops_dmrg.compute_obs(obs_mpos[mi], psi_t);
        

    # site specific observables at t=0 in array where rows are sites
    initobs = np.real(np.reshape(observables[0,obs_gen:],(len(sites), obs_per_site)));
    print("-init observables:\n",initobs);
    
    # return observables as arrays vs time
    return observables;

################################################################################
#### observables

def compute_obs(psi, mpo_inst, driver, conj=False):
    '''
    Compute expectation value of observable repped by given operator from the wf
    The wf psi must be a matrix product state, and the operator an MPO
    '''

    impo = driver.get_identity_mpo();
    return driver.expectation(psi, mpo_inst, psi)/driver.expectation(psi, impo, psi);

def get_occ(eris_or_driver, whichsite, block=True, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing the occupancy of site whichsite
    '''
    if(block): builder = eris_or_driver.expr_builder()
    else:
        Nspinorbs = len(eris_or_driver.h1e[0]);
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite],1.0);
        builder.add_term("CD",[whichsite,whichsite],1.0);
    else:
        h1e[nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        h1e[nloc*whichsite+1,nloc*whichsite+1] += 1.0;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_sz(eris_or_driver, whichsite, block=True, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sz> of site whichsite
    '''
    if(block): builder = eris_or_driver.expr_builder()
    else: 
        Nspinorbs = len(eris_or_driver.h1e[0]);
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite], 0.5);
        builder.add_term("CD",[whichsite,whichsite],-0.5);
    else:
        h1e[nloc*whichsite+0,nloc*whichsite+0] += 0.5;
        h1e[nloc*whichsite+1,nloc*whichsite+1] +=-0.5;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_Sd_mu(eris_or_driver, whichsite, component="z", verbose=0):
    '''
    Constructs an MPO representing <Sz> of site impurity at site whichsite
    '''
    builder = eris_or_driver.expr_builder()

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

    return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);

def purity_wrapper(psi,eris_or_driver, whichsite):
    '''
    Need to combine ops for x,y,z components of Sd to get purity
    '''
    components = ["x01","x10","y01","y10","z"];
    sterms = [];
    for comp in components:
        op = get_Sd_mu(eris_or_driver, whichsite, component=comp);
        sterms.append( compute_obs(psi, op, eris_or_driver));
    purity_vec = np.array([sterms[0]+sterms[1], sterms[2]+sterms[3], sterms[4]]);    
    ret = np.sqrt( np.dot(np.conj(purity_vec), purity_vec));
    if(abs(np.imag(ret)) > 1e-12): print(ret); assert False;
    return np.real(ret);

def get_concurrence(eris_or_driver, whichsites, symm_block, verbose=0):
    '''
    '''
    builder = eris_or_driver.expr_builder()

    # construct
    which1, which2 = whichsites;
    if(symm_block == 2):
        builder.add_term("PP",[which1,which2],-1.0);
    elif(symm_block == 0):
        builder.add_term("PM",[which1,which2], 1.0);
        builder.add_term("MP",[which1,which2], 1.0);
    elif(symm_block ==-2):
        builder.add_term("MM",[which1,which2],-1.0);
    else: raise NotImplementedError;

    # return
    return eris_or_driver.get_mpo(builder.finalize(),add_ident=False, iprint=verbose);

def concurrence_wrapper(psi,eris_or_driver, whichsites):
    '''
    Need to combine operators from TwoSz=+2, 0, -2 symmetry blocks
    to get concurrence
    '''

    # block3 MPS
    from pyblock3.block2.io import MPSTools, MPOTools
    psi_b3 = MPSTools.from_block2(psi); #  block 3 mps
    psi_star = psi_b3.conj(); # so now we can do this operation

    # exp vals across symmetry blocks
    sblocks = [-2,0,2];
    sterms = [];
    for sblock in sblocks:
        concur_mpo = get_concurrence(eris_or_driver, whichsites, sblock);
        concur_mpo_b3 = MPOTools.from_block2(concur_mpo);
        sterms.append( np.dot(psi_b3.conj(), concur_mpo_b3 @ psi_star)/np.dot(psi_b3.conj(),psi_b3) );
    concur_norm = np.sum(sterms);
    ret = np.sqrt(np.conj(np.sum(sterms))*np.sum(sterms));
    if(abs(np.imag(ret)) > 1e-12): print(ret); assert False;
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

def Hsys_builder(params_dict, block, scratch_dir="tmp", verbose=0):
    '''
    Builds the parts of the Hamiltonian which apply at all t
    The physical params are contained in a .json file. They are all in eV.
    They are:
    tl (lead hopping), Vconf (confining voltage depth), Be (field to polarize
    deloc es), BFM (field to polarize loc spins), Jz (z component of exchange
    for loc spins XXY model), Jx (x component of exchange for loc spins XXY
    model), Jsd (deloc e's - loc spins exchange)

    NL (number sites in left lead), NFM (number of sites in central region
    = number of loc spins), NR (number of sites in right lead), Nconf (width
    of confining region), Ne (number of electrons), TwoSz (Twice the total Sz
    of the system)

    NB this builds in terms of supersited dofs, rather than fermionic dofs

    Returns:
        if block is True: a tuple of DMRGDriver, ExprBuilder objects
        else: a tuple of 1-body, 2-body 2nd quantized Hamiltonian arrays
    '''

    # load data from json
    tl, Jz, Jx, Jsd = params_dict["tl"], params_dict["Jz"], params_dict["Jx"], params_dict["Jsd"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];

    # fermionic sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);
    assert(TwoSd == 1); # for now, to get degeneracies right

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(NL)]);
    central_sites = np.array([j for j in range(NL,NL+NFM) ]);
    rlead_sites = np.array([j for j in range(NL+NFM,Nsites)]);
    all_sites = np.array([j for j in range(Nsites)]);

    # construct ExprBuilder
    if(params_dict["symmetry"] == "Sz"):
        driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
        # using complex symmetry type, as above, seems linked to
        # Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM warnings
        # but only when TwoSz is input correctly
        # in latter case, we get a floating point exception even when complex sym is turned off!
        #driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ, n_threads=4)
        driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
    else: raise NotImplementedError

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
    qnumber = driver.bw.SX # quantum number wrapper
    # quantum numbers here: nelec, TwoSz, TwoSdz
    # Sdz is z projection of impurity spin: ladder from +s to -s
    for sitei in all_sites:
        if(sitei in llead_sites or sitei in rlead_sites): # regular fermion dofs
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
            #assert False
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

    # input custom site basis states and ops to driver
    driver.ghamil = driver.get_custom_hamiltonian(site_states, site_ops)
    builder = driver.expr_builder();
    print("\n",40*"#","\nConstructed builder\n",40*"#","\n");

    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        builder.add_term("cd",[j,j+1],-tl); 
        builder.add_term("CD",[j,j+1],-tl);
        builder.add_term("cd",[j+1,j],-tl);
        builder.add_term("CD",[j+1,j],-tl);

    # XXZ exchange between neighboring impurities
    for j in central_sites[:-1]:
        builder.add_term("ZZ",[j,j+1],-Jz);
        builder.add_term("PM",[j,j+1],-Jx/2);
        builder.add_term("MP",[j,j+1],-Jx/2);

    # sd exchange between impurities and charge density on their site
    for j in central_sites:
        # z terms
        builder.add_term("cdZ",[j,j,j],-Jsd);
        builder.add_term("CDZ",[j,j,j], Jsd);
        # plus minus terms
        builder.add_term("cDM",[j,j,j],-Jsd/2);
        builder.add_term("CdP",[j,j,j],-Jsd/2);

    return driver, builder;

def Hsys_polarizer(params_dict, to_add_to, verbose=0):
    '''
    Adds terms specific to the t<0 Hamiltonian in which the deloc e's, loc spins are
    confined and polarized by application of external fields Be, BFM

    NB this builds in terms of supersited dofs, rather than fermionic dofs

    Args:
    Params_dict: dict containing physical param values, these are defined in Hsys_base
    to_add_to, tuple of objects to add terms to:
        if block is True: these will be DMRGDriver, ExprBuilder objects
        else: these will be 1-body and 2-body parts of the second quantized
        Hamiltonian
        
    Returns:
        if block is True: a tuple of DMRGDriver, MPO
        else: return a tuple of 1-body and 2-body Hamiltonian arrays
    '''

    # load data from json
    Vconf, Be, BFM = params_dict["Vconf"], params_dict["Be"], params_dict["BFM"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];

    # fermionic sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(NL)]);
    central_sites = np.array([j for j in range(NL,NL+NFM) ]);
    rlead_sites = np.array([j for j in range(NL+NFM,Nsites)]);
    all_sites = np.array([j for j in range(Nsites)]);
    conf_sites = np.array([j for j in range(Nconf)]);

    # construct ExprBuilder
    driver, builder = to_add_to;
    if(driver.n_sites != Nsites): raise ValueError;
    
    # confining potential in left lead
    for j in conf_sites:
        builder.add_term("cd",[j,j],-Vconf); 
        builder.add_term("CD",[j,j],-Vconf);

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

    # return
    mpo_from_builder = driver.get_mpo(builder.finalize());
    return driver, mpo_from_builder;
 
    # special case initialization
    if("Bsd_x" in params_dict.keys()): # B in the x on the j that couples to 1st loc spin
        Bsd_x = params_dict["Bsd_x"];
        s = central_sites[0];
        builder.add_term("cD",[s,s],-Bsd_x/2);
    if("Bcentral" in params_dict.keys()): # B field on all js coupled to loc spins
        Bcentral = params_dict["Bcentral"];
        for s in central_sites:
            builder.add_term(spin_strs[0],[s,s],-Bcentral/2);
            builder.add_term(spin_strs[1],[s,s], Bcentral/2);

