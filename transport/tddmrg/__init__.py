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

    
##########################################################################################################
#### driver of time propagation

def kernel(params_dict, driver_inst, mpo_inst, psi, check_func, save_name, verbose=0):
    '''
    '''
    print("\n\nSTART TIME EVOLUTION (te_type = "+params_dict["te_type"]+")\n\n","*"*50,"\n\n")
    evol_start = time.time();
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

        # time evol
        krylov_subspace = 20; # default
        if(params["te_type"] == "tdvp"): krylov_subspace = 40;
        tevol_mps_inst = driver_inst.td_dmrg(mpo_inst, tevol_mps_inst, 
                delta_t=complex(0,time_step), target_t=complex(0,time_update),
                bond_dims=params_dict["bdim_t"], cutoff=params_dict["cutoff"], te_type=params["te_type"],krylov_subspace_size=krylov_subspace,
                final_mps_tag=str(int(100*total_time)), iprint=the_verbose);

        # observables
        check_func(params_dict,tevol_mps_inst,driver_inst,mpo_inst,total_time);
        plot.snapshot_bench(tevol_mps_inst, driver_inst, params_dict, save_name, time=total_time);

    evol_end = time.time();
    print(">>> Time evol compute time = {:.2f}".format(evol_end-evol_start));

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
    Constructs an operator (either MPO or ERIs) representing <Sz> of site whichsite
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
    MPO representing <Sz> of site impurity at site whichsite
    '''
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
    MPO for concurrence
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
    Sums ops made by get_concurrence from TwoSz=+2, 0, -2 symmetry blocks to find concurrence

    NB since get_concurrence ops are def'd in terms of P,M,Z operators, cannot get
    concurrence between two fermionic spins or between fermionic spin and imp spin

    NB also for s>1/2 we will need to define which levels are the qubits
    '''
    if(whichsites[0] == whichsites[1]): return np.nan;

    # block3 MPS
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

def get_pcurrent(eris_or_driver, whichsites, spin, verbose=0):
    '''
    MPO for particle current. Positive is rightward current
    Args:
    eris_or_driver, Block2 driver
    whichsites, list of site indices. must be orered, so that
    add_term( "cd", whichsites ) represents NEGATIVE current
    spin, int 0 or 1, meaning up or down current
    '''
    if(whichsites[1]-whichsites[0]!=1): raise ValueError;
    if(spin=0): spinstr = "cd";
    elif(spin=1): spinstr = "CD";
    else: raise ValueError;

    # construct MPO
    builder = eris_or_driver.expr_builder();
    builder.add_term(spinstr, whichsites, -1); # c on left, d on right = negative current
    builder.add_term(spinstr, whichsites[::-1], 1); # c on right, d on left = positive current
    return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);

def pcurrent_wrapper(psi, eris_or_driver, whichsite, verbose=0):
    '''
    Consider site whichsite. This wrapper:
    1) sums the spin currents from whichsite-1 to whcihsite (LEFT part)
    2) sums the spin currents from whichsite to whichsite+1 (RIGHT part)
    3) averages over the results of 1 and 2 to find the current "through" whichsite
    '''

    # left part
    pcurrent_left = 0.0;
    for spin in [0,1]:
        left_mpo += get_pcurrent(eris_or_driver, [whichsite-1, whichsite], spin, verbose=verbose);
        pcurrent_left += compute_obs(psi, left_mpo, eris_or_driver);

    # right part
    pcurrent_right = 0.0;
    for spin in [0,1]:
        right_mpo += get_pcurrent(eris_or_driver, [whichsite, whichsite+1], spin, verbose=verbose);
        pcurrent_right += compute_obs(psi, right_mpo, eris_or_driver);

    # average
    return (pcurrent_left + pcurrent_right)/2;
    

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

def H_wrapper(params_dict, sys_type, time, scratch_dir, verbose=0):
    '''
    Wrapper that allows calling builder/polarizer Ham constructor for MULTIPLE system types,
    eg, STT, SIAM, etc.

    Args:
    params_dict, a dictionary with all the physical params. Its correspondence with sys_type is
        automatically checked
    sys_type, a string telling what kind of 1D system we are choosing
    time, int in 0 or 1, whether to include initial state prep Ham ("polarizing"
    Ham at time<0) or not (time>0)
    scratch_dir, path to where to save MPS info
    '''

    if(sys_type=="STT"):
        needed_keys = ["Jsd","Jx","Jz"];
        H_constructor = H_STT_builder;
        H_add = H_STT_polarizer;
    elif(sys_type=="SIAM"):
        needed_keys = ["U","Vg","Vb"];
        H_constructor = H_SIAM_builder;
        H_add = H_SIAM_polarizer;
    else:
        raise Exception("System type = "+sys_type+" not supported");

    # check compatibility
    for key in needed_keys:
        if(key not in params_dict.keys()):
            raise KeyError("params_dict missing "+key);

    # construct
    H_t = H_constructor(params_dict, scratch_dir=scratch_dir, verbose=verbose); # all times
    if(time==0): H_t = H_add(params_dict, H_t, verbose=verbose); # time<=0 only;
    return H_t;

def H_SIAM_builder(params_dict, scrath_dir="tmp",verbose=0):
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
    NB this requires that Ne//2==0

    There is NO supersiting in this system

    Returns: a tuple of DMRGDriver, ExprBuilder objects
    '''

    # load data from json
    tl, th, Vg, U, Vb = params_dict["tl"], params_dict["th"], params_dict["Vg"], params_dict["U"], params_dict["Vb"];
    NL, NR = params_dict["NL"], params_dict["NR"];
    Nsites = NL+1+NR;
    Ne=Nsites;
    assert(Ne//2 ==0); # need even number of electrons for TwoSz=0
    TwoSz = 0;

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(NL)]);
    central_sites = np.array([j for j in range(NL,NL+1) ]);
    rlead_sites = np.array([j for j in range(NL+1,Nsites)]);
    all_sites = np.array([j for j in range(Nsites)]);

    # construct ExprBuilder
    if(params_dict["symmetry"] == "Sz"):
        driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
        driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
    else: raise NotImplementedError;
    builder = driver.expr_builder();
    print("\n",40*"#","\nConstructed builder\n",40*"#","\n");

    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        builder.add_term("cd",[j,j+1],-tl); 
        builder.add_term("CD",[j,j+1],-tl);
        builder.add_term("cd",[j+1,j],-tl);
        builder.add_term("CD",[j+1,j],-tl);

    # Vg and U on impurity
    for j in central_sites:
        builder.add_term("cd",[j,j], Vg);
        builder.add_term("CD",[j,j], Vg);
        builder.add_term("cdCD",[j,j,j,j], U);

    # bias
    # NB this will be REMOVED by polarizer so that it is ABSENT for t<0
    # and PRESENT at t>0 (opposite to B fields in STT, but still "added"
    # by the polarizer
    for j in llead_sites:
        builder.add_term("cd",[j,j], Vb/2); 
        builder.add_term("CD",[j,j], Vb/2);
    for j in rlead_sites:
        builder.add_term("cd",[j,j],-Vb/2); 
        builder.add_term("CD",[j,j],-Vb/2);

    return driver, builder;

def H_SIAM_polarizer(params_dict, to_add_to, verbose=0):
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
    
    # load data from json
    tl, th, Vg, U, Vb = params_dict["tl"], params_dict["th"], params_dict["Vg"], params_dict["U"], params_dict["Vb"];
    NL, NR = params_dict["NL"], params_dict["NR"];
    Nsites = NL+1+NR;

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(NL)]);
    rlead_sites = np.array([j for j in range(NL+1,Nsites)]);

    # unpack ExprBuilder
    driver, builder = to_add_to;
    if(driver.n_sites != Nsites): raise ValueError;

    # REMOVE bias
    for j in llead_sites:
        builder.add_term("cd",[j,j],-Vb/2); 
        builder.add_term("CD",[j,j],-Vb/2);
    for j in rlead_sites:
        builder.add_term("cd",[j,j], Vb/2); 
        builder.add_term("CD",[j,j], Vb/2);

    # return
    mpo_from_builder = driver.get_mpo(builder.finalize());
    return driver, mpo_from_builder;

def H_STT_builder(params_dict, scratch_dir="tmp", verbose=0):
    '''
    Builds the parts of the STT Hamiltonian which apply at all t
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

    Returns: a tuple of DMRGDriver, ExprBuilder objects
    '''

    # load data from json
    tl, Jz, Jx, Jsd = params_dict["tl"], params_dict["Jz"], params_dict["Jx"], params_dict["Jsd"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];
    Nbuffer = 0;
    if("Nbuffer" in params_dict.keys()): Nbuffer = params_dict["Nbuffer"];

    # fermionic sites and spin
    Nsites = Nbuffer+NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);
    assert(TwoSd == 1); # for now, to get degeneracies right

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(Nbuffer,Nbuffer+NL)]);
    central_sites = np.array([j for j in range(Nbuffer+NL,Nbuffer+NL+NFM) ]);
    rlead_sites = np.array([j for j in range(Nbuffer+NL+NFM,Nbuffer+Nsites)]);
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
    else: raise NotImplementedError;

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

    # input custom site basis states and ops to driver, and build builder
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
        builder.add_term("cdZ",[j,j,j],-Jsd/2);
        builder.add_term("CDZ",[j,j,j], Jsd/2);
        # plus minus terms
        builder.add_term("cDM",[j,j,j],-Jsd/2);
        builder.add_term("CdP",[j,j,j],-Jsd/2);

    return driver, builder;

def H_STT_polarizer(params_dict, to_add_to, verbose=0):
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

    # load data from json
    Vconf, Be, BFM = params_dict["Vconf"], params_dict["Be"], params_dict["BFM"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];
    Nbuffer = 0;
    if("Nbuffer" in params_dict.keys()): Nbuffer = params_dict["Nbuffer"];

    # fermionic sites and spin
    Nsites = Nbuffer+NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(Nbuffer,Nbuffer+NL)]);
    central_sites = np.array([j for j in range(Nbuffer+NL,Nbuffer+NL+NFM) ]);
    rlead_sites = np.array([j for j in range(Nbuffer+NL+NFM,Nbuffer+Nsites)]);
    all_sites = np.array([j for j in range(Nsites)]);
    conf_sites = np.array([j for j in range(Nbuffer,Nbuffer+Nconf)]);

    # unpack ExprBuilder
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

