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
    if(conj): psi_bra = None;
    return driver.expectation(psi, mpo_inst, psi)/driver.expectation(psi, impo, psi);

def get_occ(N, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing the occupancy of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((N,N),dtype=float), np.zeros((N,N,N,N),dtype=float);

    # construct
    for spin in spin_inds:
        if(block):
            builder.add_term(spin_strs[spin],[whichsite,whichsite],1.0);
        else:
            h1e[nloc*whichsite+spin,nloc*whichsite+spin] += 1.0;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_sz(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sz> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite], 0.5);
        builder.add_term("CD",[whichsite,whichsite],-0.5);
    else:
        h1e[nloc*whichsite+spin_inds[0],nloc*whichsite+spin_inds[0]] += 0.5;
        h1e[nloc*whichsite+spin_inds[1],nloc*whichsite+spin_inds[1]] +=-0.5;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);

def get_sx01(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sx> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cD",[whichsite,whichsite], 0.5);
    else:
        h1e[nloc*whichsite+spin_inds[0],nloc*whichsite+spin_inds[1]] += 0.5;
        
    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff, imag_cutoff = 1e3);

def get_sx10(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sx> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("Cd",[whichsite,whichsite],0.5);
    else:
        h1e[nloc*whichsite+spin_inds[1],nloc*whichsite+spin_inds[0]] += 0.5;
        
    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff, imag_cutoff = 1e3);

def get_sy01(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sy> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cD",[whichsite,whichsite], complex(0,-0.5));
    else:
        h1e[nloc*whichsite+spin_inds[0],nloc*whichsite+spin_inds[1]] += complex(0,-0.5);
        
    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff, imag_cutoff = 1e3);

def get_sy10(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sy> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("Cd",[whichsite,whichsite],complex(0,0.5));
    else:
        h1e[nloc*whichsite+spin_inds[1],nloc*whichsite+spin_inds[0]] += complex(0,0.5);
        
    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff, imag_cutoff = 1e3);

def get_concurrence(Nspinorbs, eris_or_driver, whichsites, block, symm_block, verbose=0):
    '''
    '''
    spin_inds = [0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): builder = eris_or_driver.expr_builder()
    else: h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    which1, which2 = whichsites;
    if(block):
        if(symm_block == 2):
            builder.add_term("cDcD",[which1,which1,which2,which2],-1.0);
        elif(symm_block == 0):
            builder.add_term("cDCd",[which1,which1,which2,which2], 1.0);
            builder.add_term("CdcD",[which1,which1,which2,which2], 1.0);
        elif(symm_block ==-2):
            builder.add_term("CdCd",[which1,which1,which2,which2],-1.0);
        else: raise NotImplementedError;
    else:
        if(symm_block == 2):
            g2e[nloc*which1+spin_inds[0],nloc*which1+spin_inds[1],nloc*which2+spin_inds[0],nloc*which2+spin_inds[1]] += -1.0;
            #
            g2e[nloc*which2+spin_inds[0],nloc*which2+spin_inds[1],nloc*which1+spin_inds[0],nloc*which1+spin_inds[1]] += -1.0;
        elif(symm_block == 0):
            g2e[nloc*which1+spin_inds[0],nloc*which1+spin_inds[1],nloc*which2+spin_inds[1],nloc*which2+spin_inds[0]] += 1.0;
            g2e[nloc*which1+spin_inds[1],nloc*which1+spin_inds[0],nloc*which2+spin_inds[0],nloc*which2+spin_inds[1]] += 1.0;
            #
            g2e[nloc*which2+spin_inds[1],nloc*which2+spin_inds[0],nloc*which1+spin_inds[0],nloc*which1+spin_inds[1]] += 1.0;
            g2e[nloc*which2+spin_inds[0],nloc*which2+spin_inds[1],nloc*which1+spin_inds[1],nloc*which1+spin_inds[0]] += 1.0;
        elif(symm_block ==-2):
            g2e[nloc*which1+spin_inds[1],nloc*which1+spin_inds[0],nloc*which2+spin_inds[1],nloc*which2+spin_inds[0]] += -1.0;
            #
            g2e[nloc*which2+spin_inds[1],nloc*which2+spin_inds[0],nloc*which1+spin_inds[1],nloc*which1+spin_inds[0]] += -1.0;
        else: raise NotImplementedError;

    # return
    if(block):
        mpo_from_builder = eris_or_driver.get_mpo(builder.finalize(),add_ident=False, iprint=verbose);
        return mpo_from_builder;
    else:
        occ_eri = tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);
        return occ_eri;

def concurrence_wrapper(psi,eris_or_driver, whichsites, block):
    '''
    Need to combine operators from TwoSz=+2, 0, -2 symmetry blocks
    to get concurrence
    '''
    # unpack
    spin_inds = [0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);
    if(block): Nspinorbs = eris_or_driver.n_sites*2;
    else: Nspinorbs = len(eris_or_driver.h1e[0]);
    which1, which2 = whichsites;

    # not implemented for FCI
    if(not block): return np.nan;

    # block3 MPS
    from pyblock3.block2.io import MPSTools, MPOTools
    psi_b3 = MPSTools.from_block2(psi); #  block 3 mps
    psi_star = psi_b3.conj(); # so now we can do this operation

    # exp vals across symmetry blocks
    sblocks = [-2,0,2];
    sterms = [];
    for sblock in sblocks:
        concur_mpo = get_concurrence(Nspinorbs, eris_or_driver, whichsites, block, sblock);
        concur_mpo_b3 = MPOTools.from_block2(concur_mpo);
        sterms.append( np.dot(psi_b3.conj(), concur_mpo_b3 @ psi_star)/np.dot(psi_b3.conj(),psi_b3) );
    concur_norm = np.sum(sterms);
    ret = np.sqrt(np.conj(np.sum(sterms))*np.sum(sterms));
    if(abs(np.imag(ret)) > 1e-12): print(ret); assert False;
    return np.real(ret);

def purity_wrapper(psi,eris_or_driver, whichsites, block):
    '''
    Need to combine ops for Sx, Sy, Sz to get purity
    '''
    # unpack
    spin_inds = [0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);
    if(block): Nspinorbs = eris_or_driver.n_sites*2;
    else: Nspinorbs = len(eris_or_driver.h1e[0]);
    whichsite = whichsites[0];

    # not implemented for FCI
    if(not block): return np.nan;

    # exp vals across symmetry blocks
    sblocks = [get_sx01, get_sx10, get_sy01, get_sy10, get_sz];
    sterms = [];
    for sblock in sblocks:
        op = sblock(Nspinorbs, eris_or_driver, whichsite, block);
        sterms.append( compute_obs(psi, op, eris_or_driver));
    purity_vec = np.array([sterms[0]+sterms[1], sterms[2]+sterms[3], sterms[4]]);    ret = np.sqrt( np.dot(np.conj(purity_vec), purity_vec));
    if(abs(np.imag(ret)) > 1e-12): print(ret); assert False;
    return np.real(ret);

##########################################################################################################
#### hamiltonian constructors

def Hsys_builder(params_dict, block, scratch_dir="tmp", verbose=0):
    '''
    Builds the parts of the Hamiltonian which apply at all t
    NB this contains one-body terms, which are spin-independent, and
    two-body terms, which are spin-dependent

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

    Returns:
        if block is True: a tuple of DMRGDriver, ExprBuilder objects
        else: a tuple of 1-body, 2-body 2nd quantized Hamiltonian arrays
    '''

    # load data from json
    tl, Jz, Jx, Jsd = params_dict["tl"], params_dict["Jz"], params_dict["Jx"], params_dict["Jsd"];
    NL, NFM, NR, Nconf, Ne, TwoSz = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"], params_dict["Ne"], params_dict["TwoSz"];

    # sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    Nspinorbs = 2*2*Nsites; # number of fermionic states w/ occupancy 0 or 1
                # factor of 2 for spin and 2 for off-chain loc spin (d sites)
    spin_strs = np.array(params_dict["spin_strs"]); # operator strings for each spin
    spin_inds, nloc = np.array(range(len(spin_strs))), len(spin_strs);  # for summing over spin
    assert(NL>0 and NR>0); # leads must exist

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(2*NL) if j%2==0]);
    central_sites = np.array([j for j in range(2*NL,2*(NL+NFM) ) if j%2==0]);
    loc_spins = np.array([d for d in range(2*NL,2*(NL+NFM))  if d%2==1]);
    rlead_sites = np.array([j for j in range(2*(NL+NFM), 2*Nsites) if j%2==0]);
    j_sites = np.array([j for j in range(2*Nsites) if j%2==0]);

    # return objects
    if(block): # construct ExprBuilder
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            #driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ, n_threads=4);
            driver.initialize_system(n_sites=2*Nsites, n_elec=Ne+NFM, spin=TwoSz);
        else:
            raise NotImplementedError;
        builder = driver.expr_builder();
    else:       # <---------- change dtype to complex ?
      h1e, g2e = np.zeros((Nspinorbs, Nspinorbs),dtype=float), np.zeros((Nspinorbs, Nspinorbs, Nspinorbs, Nspinorbs),dtype=float);

    # j-j hopping everywhere
    for jindex in range(len(j_sites)-1):
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[j_sites[jindex],j_sites[jindex+1]],-tl);
                builder.add_term(spin_strs[spin],[j_sites[jindex+1],j_sites[jindex]],-tl);
            else:
                h1e[nloc*j_sites[jindex]+spin,nloc*j_sites[jindex+1]+spin] += -tl;
                h1e[nloc*j_sites[jindex+1]+spin,nloc*j_sites[jindex]+spin] += -tl;


    # sd exchange between loc spins and adjacent central sites
    # central sites are indexed j, loc spin sites are indexed d
    sdpairs = [(central_sites[index], loc_spins[index]) for index in range(len(loc_spins))];
    if(verbose): print("j - d site pairs = ",sdpairs);
    # form of this interaction is
    # \sum_{\mu=x,y,z} \sum_{\sigma \sigma' \tau \tau'}
    #            c_j\sigma^\dagger c_j\sigma' c_d\tau^\dagger c_d\tau'
    #            (J \sigma^\mu_{\sigma\sigma'} \sigma^\mu_{\tau\tau'}
    # where \sigma^\mu denotes a single Pauli matrix, the mu^th compoent of the Pauli vector
    for (j,d) in sdpairs:
        if(block):
            # z component terms
            builder.add_term("cdcd",[j,j,d,d],-Jsd/4);
            builder.add_term("cdCD",[j,j,d,d], Jsd/4);
            builder.add_term("CDcd",[j,j,d,d], Jsd/4);
            builder.add_term("CDCD",[j,j,d,d],-Jsd/4);
            # x+y component -> +- terms
            builder.add_term("cDCd",[j,j,d,d],-Jsd/2);
            builder.add_term("CdcD",[j,j,d,d],-Jsd/2);
        else:
            # z component terms
            g2e[nloc*j+spin_inds[0],nloc*j+spin_inds[0],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] += -Jsd/4;
            g2e[nloc*j+spin_inds[0],nloc*j+spin_inds[0],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] +=  Jsd/4;
            g2e[nloc*j+spin_inds[1],nloc*j+spin_inds[1],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] +=  Jsd/4;
            g2e[nloc*j+spin_inds[1],nloc*j+spin_inds[1],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] += -Jsd/4;
            # x+y component -> +- terms
            g2e[nloc*j+spin_inds[0],nloc*j+spin_inds[1],nloc*(d)+spin_inds[1],nloc*(d)+spin_inds[0]] += -Jsd/2;
            g2e[nloc*j+spin_inds[1],nloc*j+spin_inds[0],nloc*(d)+spin_inds[0],nloc*(d)+spin_inds[1]] += -Jsd/2;
            # repeat above with switched particle labels (pq|rs) = (rs|pq)
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*j+spin_inds[0],nloc*j+spin_inds[0]] += -Jsd/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*j+spin_inds[0],nloc*j+spin_inds[0]] +=  Jsd/4;
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*j+spin_inds[1],nloc*j+spin_inds[1]] +=  Jsd/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*j+spin_inds[1],nloc*j+spin_inds[1]] += -Jsd/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[0],nloc*j+spin_inds[0],nloc*j+spin_inds[1]] += -Jsd/2;
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[1],nloc*j+spin_inds[1],nloc*j+spin_inds[0]] += -Jsd/2;


    # XXZ for loc spins
    for loci in range(len(loc_spins)-1): # nearest neighbor only
        d, dp1 = loc_spins[loci], loc_spins[loci+1];
        if(block):
            # z component termse
            builder.add_term("cdcd",[d,d,dp1,dp1],-Jz/4);
            builder.add_term("cdCD",[d,d,dp1,dp1], Jz/4);
            builder.add_term("CDcd",[d,d,dp1,dp1], Jz/4);
            builder.add_term("CDCD",[d,d,dp1,dp1],-Jz/4);
            # x+y component -> +- terms
            builder.add_term("cDCd",[d,d,dp1,dp1],-Jx/2);
            builder.add_term("CdcD",[d,d,dp1,dp1],-Jx/2);
        else:
            # z component terms
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0]] += -Jz/4;
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1]] +=  Jz/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0]] +=  Jz/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1]] += -Jz/4;
            # x+y component -> +- terms
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[0]] += -Jx/2;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[1]] += -Jx/2;
            # repeat above with switched particle labels (pq|rs) = (rs|pq)
            g2e[nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] += -Jz/4;
            g2e[nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] +=  Jz/4;
            g2e[nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] +=  Jz/4;
            g2e[nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] += -Jz/4;
            g2e[nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[0],nloc*d+spin_inds[0],nloc*d+spin_inds[1]] += -Jx/2;
            g2e[nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[1],nloc*d+spin_inds[1],nloc*d+spin_inds[0]] += -Jx/2;

    # return
    if(block):
        return driver, builder;
    else:
        return h1e, g2e;

def Hsys_polarizer(params_dict, block, to_add_to, verbose=0):
    '''
    Adds terms specific to the t<0 Hamiltonian in which the deloc e's, loc spins are
    confined and polarized by application of external fields Be, BFM
    NB this contains one-body terms only, and these terms differ by spin

    Args:
    Params_dict: dict containing physical param values, these are defined in Hsys_base
    block: bool, tells how to construct the object that will hold the Hamiltonian:
      if True: construct a block2 ExprBuilder object
      else: construct a tuple of 1body 2nd quantized hams
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
    NL, NFM, NR, Nconf, Ne, TwoSz = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"], params_dict["Ne"], params_dict["TwoSz"];

    # sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    Nspinorbs = 2*2*Nsites; # number of fermionic states w/ occupancy 0 or 1
                # factor of 2 for spin and 2 for off-chain loc spin (d sites)
    spin_strs = np.array(params_dict["spin_strs"]); # operator strings for each spin
    spin_inds, nloc = np.array(range(len(spin_strs))), len(spin_strs);  # for summing over spin
    assert(Nconf <= NL); # must be able to confine within left lead
    assert(Nconf >= Ne); # must be enough room for all deloc es

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(2*NL) if j%2==0]);
    central_sites = np.array([j for j in range(2*NL,2*(NL+NFM) ) if j%2==0]);
    loc_spins = np.array([d for d in range(2*NL,2*(NL+NFM))  if d%2==1]);
    rlead_sites = np.array([j for j in range(2*(NL+NFM), 2*Nsites) if j%2==0]);
    j_sites = np.array([j for j in range(2*Nsites) if j%2==0]);
    d_sites = np.array([d for d in range(2*Nsites) if d%2==1]);
    conf_sites = np.array([j for j in range(2*Nconf) if j%2==0]);

    # return objects
    if(block): # construct ExprBuilder
        driver, builder = to_add_to;
        if(driver.n_sites != 2*Nsites): raise ValueError;
    else:
        h1e, g2e = to_add_to;
        if(len(h1e) != Nspinorbs): raise ValueError;

    # confining potential in left lead
    for j in conf_sites:
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[j,j],-Vconf); 
            else:
                h1e[nloc*j+spin,nloc*j+spin] += -Vconf;

    # B field in the confined region ----------> ASSUMED IN THE Z
    # only within the region of confining potential
    for j in conf_sites:
        if(block):
            builder.add_term(spin_strs[0],[j,j],-Be/2);
            builder.add_term(spin_strs[1],[j,j], Be/2);
        else:
            h1e[nloc*j+spin_inds[0],nloc*j+spin_inds[0]] += -Be/2; # if Be>0, spin up should be favored
            h1e[nloc*j+spin_inds[1],nloc*j+spin_inds[1]] +=  Be/2;

    # confining potential on loc spins
    if("noFM" in params_dict.keys()):
        multiplier = -0.5; # special case: conf region is filled before loc spins
    else: multiplier = -2; # default: loc spins are filled, one each, first
    for d in loc_spins:
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[d,d],multiplier*Vconf);
            else:
                h1e[nloc*d+spin,nloc*d+spin] += multiplier*Vconf;

    # B field on the loc spins
    for d in loc_spins:
        if(block):
            builder.add_term(spin_strs[0],[d,d],-BFM/2);
            builder.add_term(spin_strs[1],[d,d], BFM/2);
        else:
            h1e[nloc*d+spin_inds[0],nloc*d+spin_inds[0]] += -BFM/2;
            h1e[nloc*d+spin_inds[1],nloc*d+spin_inds[1]] +=  BFM/2; # typically BFM<0, spin down should be favored

    # special case initialization
    if("BFM_first" in params_dict.keys()): # B field that targets 1st loc spin only
        BFM_first = params_dict["BFM_first"];
        d = loc_spins[0];
        if(block):
            builder.add_term(spin_strs[0],[d,d],-BFM_first/2 + BFM/2);
            builder.add_term(spin_strs[1],[d,d], BFM_first/2 - BFM/2);
        else:
            h1e[nloc*d+spin_inds[0],nloc*d+spin_inds[0]] += -BFM_first/2 + BFM/2;
            h1e[nloc*d+spin_inds[1],nloc*d+spin_inds[1]] +=  BFM_first/2 - BFM/2; 
    if("Bsd" in params_dict.keys()): # B field on the j that couples to the first loc spin
        Bsd = params_dict["Bsd"];
        s = central_sites[0];
        if(block):
            builder.add_term(spin_strs[0],[s,s],-Bsd/2);
            builder.add_term(spin_strs[1],[s,s], Bsd/2);
        else:
            h1e[nloc*s+spin_inds[0],nloc*s+spin_inds[0]] += -Bsd/2;
            h1e[nloc*s+spin_inds[1],nloc*s+spin_inds[1]] +=  Bsd/2;
    if("Bsd_x" in params_dict.keys()): # B in the x on the j that couples to 1st loc spin
        Bsd_x = params_dict["Bsd_x"];
        s = central_sites[0];
        if block:
            builder.add_term("cD",[s,s],-Bsd_x/2);
        else:
            h1e[nloc*s+spin_inds[0],nloc*s+spin_inds[1]] += -Bsd_x/2;
    if("Bcentral" in params_dict.keys()): # B field on all js coupled to loc spins
        Bcentral = params_dict["Bcentral"];
        for s in central_sites:
            if(block):
                builder.add_term(spin_strs[0],[s,s],-Bcentral/2);
                builder.add_term(spin_strs[1],[s,s], Bcentral/2);
            else:
                h1e[nloc*s+spin_inds[0],nloc*s+spin_inds[0]] += -Bcentral/2;
                h1e[nloc*s+spin_inds[1],nloc*s+spin_inds[1]] +=  Bcentral/2;

    #### noise terms
                
    # d-d hopping noise
    jdnoise = params_dict["jdnoise"];
    for dindex in range(len(d_sites)-1):
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[d_sites[dindex],d_sites[dindex+1]],-jdnoise);
                builder.add_term(spin_strs[spin],[d_sites[dindex+1],d_sites[dindex]],-jdnoise);
            else:
                h1e[nloc*d_sites[dindex]+spin,nloc*d_sites[dindex+1]+spin] += -jdnoise;
                h1e[nloc*d_sites[dindex+1]+spin,nloc*d_sites[dindex]+spin] += -jdnoise;

    # j-d mixing noise
    jdnoise = params_dict["jdnoise"];
    for jord in range(2*Nsites-1):
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[jord,jord+1],-jdnoise);
                builder.add_term(spin_strs[spin],[jord+1,jord],-jdnoise);
            else:
                h1e[nloc*jord+spin,nloc*(jord+1)+spin] += -jdnoise;
                h1e[nloc*(jord+1)+spin,nloc*jord+spin] += -jdnoise;

    # return
    if(block):
        mpo_from_builder = driver.get_mpo(builder.finalize(), iprint=verbose);
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;

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

def Hsuper_builder(params_dict, block, scratch_dir="tmp", verbose=0):
    '''
    Builds the parts of the Hamiltonian which apply at all t
    ( see Hsys_builder above )

    However, this builds in terms of supersited dofs, rather than fermionic dofs

    Returns:
        if block is True: a tuple of DMRGDriver, ExprBuilder objects
        else: a tuple of 1-body, 2-body 2nd quantized Hamiltonian arrays
    '''

    # load data from json
    tl, Jz, Jx, Jsd = params_dict["tl"], params_dict["Jz"], params_dict["Jx"], params_dict["Jsd"];
    NL, NFM, NR, Nconf, Ne = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"], params_dict["Ne"];

    # fermionic sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    TwoSz = params_dict["TwoSz"]; # total fermion spin in the z
    #assert(NL>0 and NR>0); # leads must exist
    spin_strs = np.array(params_dict["spin_strs"]); # operator strings for each fermion spin
    spin_inds, nloc = np.array(range(len(spin_strs))), len(spin_strs);  # for summing over fermion spin

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd);
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);
    #assert(TwoSd == 1); # for now

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(NL)]);
    central_sites = np.array([j for j in range(NL,NL+NFM) ]);
    rlead_sites = np.array([j for j in range(NL+NFM,Nsites)]);
    all_sites = np.array([j for j in range(Nsites)]);

    # return object
    if(block): # construct ExprBuilder
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            # using complex symmetry type, as above, seems linked to
            # Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM warnings
            # but only when TwoSz is input correctly
            # in latter case, we get a floating point exception even when complex sym is turned off!
            #driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ, n_threads=4)
            driver.initialize_system(n_sites=Nsites, n_elec=Ne, spin=TwoSz);
        else:
            raise NotImplementedError;

    # def custom states and operators
    site_states, site_ops = [], [];
    qnumber = driver.bw.SX # quantum number wrapper
    # quantum numbers here: nelec, TwoSz, TwoSdz
    # Sdz is z projection of impurity spin: ladder from +s to -s

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
    print("four_Sdz = \n",utils.mat_4d_to_2d(fourd_Sdz))
    fourd_Sdminus = np.copy(fourd_base);
    fourd_Sdplus = np.copy(fourd_base);
    for Sdz_index in range(n_imp_dof-1): 
        fourd_Sdminus[Sdz_index,Sdz_index+1] = np.sqrt(0.5*TwoSd*(0.5*TwoSd+1)-0.5*TwoSdz_ladder[Sdz_index+1]*(0.5*TwoSdz_ladder[Sdz_index+1]-1))*np.eye(n_fer_dof);
        fourd_Sdplus[Sdz_index+1,Sdz_index] = np.sqrt(0.5*TwoSd*(0.5*TwoSd+1)-0.5*TwoSdz_ladder[Sdz_index]*(0.5*TwoSdz_ladder[Sdz_index]+1))*np.eye(n_fer_dof);
    print("four_Sdminus = \n",utils.mat_4d_to_2d(fourd_Sdminus))
    print("four_Sdplus = \n",utils.mat_4d_to_2d(fourd_Sdplus))
    A, B, C, D = 11,12,13,14
    pre = np.array([ [[[1,0],[0,A]],[[0,0],[B,0]] ],
                     [[[0,C],[0,0]],[[D,0],[0,1]]] ])
    #print(utils.mat_4d_to_2d(pre))
    #print(reblock(pre))
    #assert False

    # define site dependent basis
    for sitei in all_sites:
        if(not (sitei in central_sites)): # just has fermionic dofs
            states = [(qnumber(0, 0,0),1), # |> # <-- 2nd thing in tuple is degeneracy
                      (qnumber(1, 1,0),1), # |up> #<-- 
                      (qnumber(1,-1,0),1), # |down>
                      (qnumber(2, 0,0),1)];# |up down>
            ops = { "":np.copy(squar_I), # identity
                   "c":np.copy(squar_c), # c_up^\dagger
                   "d":np.copy(squar_d), # c_up
                   "C":np.copy(squar_C), # c_down^\dagger
                   "D":np.copy(squar_D)} # c_down
                    
        else: # has fermion AND impurity dofs
            states = [];
            for dummy in [1]: #TwoSdz in TwoSdz_ladder:
                point_ovld = 0;
                states.append((qnumber(0, 0,point_ovld),n_imp_dof)); # | > x {|+Sd>...|-Sd>}
                states.append((qnumber(1, 1,point_ovld),n_imp_dof)); # |up> x {|+Sd>...|-Sd>}
                states.append((qnumber(1,-1,point_ovld),n_imp_dof)); # |down> x {|+Sd>...|-Sd>}
                states.append((qnumber(2, 0,point_ovld),n_imp_dof)); # |up down> x {|+Sd>...|-Sd>}
            
            # ops dictionary
            ops = { "":np.eye(n_fer_dof*n_imp_dof), # identity
                   "c":reblock(fourd_c), # c_up^\dagger
                   "d":reblock(fourd_d), # c_up
                   "C":reblock(fourd_C), # c_down^\dagger
                   "D":reblock(fourd_D), # c_down
                   "Z":reblock(fourd_Sdz),    # Sz of impurity
                   "P":reblock(fourd_Sdplus), # S+ on impurity
                   "M":reblock(fourd_Sdminus) # S- on impurity
                    }
            print(len(states)*n_imp_dof)
            print("old shape = ",np.shape(fourd_c))
            print("new shape = ",np.shape(ops["c"]))
        site_states.append(states);
        site_ops.append(ops);

    # return objects
    if(block): # input custom site basis states and ops to driver
        driver.ghamil = driver.get_custom_hamiltonian(site_states, site_ops)
        builder = driver.expr_builder();
    else:
        raise NotImplementedError;

    # j <-> j+1 hopping everywhere
    for j in all_sites[:-1]:
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[j,j+1],-tl);
                builder.add_term(spin_strs[spin],[j+1,j],-tl);

    # XXZ exchange between neighboring impurities
    for j in central_sites[:-1]:
        if(block):
            #pass;
            builder.add_term("ZZ",[j,j+1],-Jz);
            builder.add_term("PM",[j,j+1],-Jx/2);
            builder.add_term("MP",[j,j+1],-Jx/2);

    # sd exchange between impurities and charge density on their site
    for j in central_sites:
        if(block):
            # z terms
            builder.add_term("cdZ",[j,j,j],-Jsd);
            builder.add_term("CDZ",[j,j,j], Jsd);
            # plus minus terms
            #builder.add_term("cDM",[j,j,j],-Jsd/2);
            #builder.add_term("CdP",[j,j,j],-Jsd/2);

    # return
    if(block):
        return driver, builder;
    else:
        return h1e, g2e;

    # sd exchange between loc spins and adjacent central sites
    # central sites are indexed j, loc spin sites are indexed d
    sdpairs = [(central_sites[index], loc_spins[index]) for index in range(len(loc_spins))];
    if(verbose): print("j - d site pairs = ",sdpairs);
    # form of this interaction is
    # \sum_{\mu=x,y,z} \sum_{\sigma \sigma' \tau \tau'}
    #            c_j\sigma^\dagger c_j\sigma' c_d\tau^\dagger c_d\tau'
    #            (J \sigma^\mu_{\sigma\sigma'} \sigma^\mu_{\tau\tau'}
    # where \sigma^\mu denotes a single Pauli matrix, the mu^th compoent of the Pauli vector
    for (j,d) in sdpairs:
        if(block):
            # z component terms
            builder.add_term("cdcd",[j,j,d,d],-Jsd/4);
            builder.add_term("cdCD",[j,j,d,d], Jsd/4);
            builder.add_term("CDcd",[j,j,d,d], Jsd/4);
            builder.add_term("CDCD",[j,j,d,d],-Jsd/4);
            # x+y component -> +- terms
            builder.add_term("cDCd",[j,j,d,d],-Jsd/2);
            builder.add_term("CdcD",[j,j,d,d],-Jsd/2);
        else:
            # z component terms
            g2e[nloc*j+spin_inds[0],nloc*j+spin_inds[0],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] += -Jsd/4;
            g2e[nloc*j+spin_inds[0],nloc*j+spin_inds[0],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] +=  Jsd/4;
            g2e[nloc*j+spin_inds[1],nloc*j+spin_inds[1],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] +=  Jsd/4;
            g2e[nloc*j+spin_inds[1],nloc*j+spin_inds[1],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] += -Jsd/4;
            # x+y component -> +- terms
            g2e[nloc*j+spin_inds[0],nloc*j+spin_inds[1],nloc*(d)+spin_inds[1],nloc*(d)+spin_inds[0]] += -Jsd/2;
            g2e[nloc*j+spin_inds[1],nloc*j+spin_inds[0],nloc*(d)+spin_inds[0],nloc*(d)+spin_inds[1]] += -Jsd/2;
            # repeat above with switched particle labels (pq|rs) = (rs|pq)
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*j+spin_inds[0],nloc*j+spin_inds[0]] += -Jsd/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*j+spin_inds[0],nloc*j+spin_inds[0]] +=  Jsd/4;
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*j+spin_inds[1],nloc*j+spin_inds[1]] +=  Jsd/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*j+spin_inds[1],nloc*j+spin_inds[1]] += -Jsd/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[0],nloc*j+spin_inds[0],nloc*j+spin_inds[1]] += -Jsd/2;
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[1],nloc*j+spin_inds[1],nloc*j+spin_inds[0]] += -Jsd/2;


    # XXZ for loc spins
    for loci in range(len(loc_spins)-1): # nearest neighbor only
        d, dp1 = loc_spins[loci], loc_spins[loci+1];
        if(block):
            # z component termse
            builder.add_term("cdcd",[d,d,dp1,dp1],-Jz/4);
            builder.add_term("cdCD",[d,d,dp1,dp1], Jz/4);
            builder.add_term("CDcd",[d,d,dp1,dp1], Jz/4);
            builder.add_term("CDCD",[d,d,dp1,dp1],-Jz/4);
            # x+y component -> +- terms
            builder.add_term("cDCd",[d,d,dp1,dp1],-Jx/2);
            builder.add_term("CdcD",[d,d,dp1,dp1],-Jx/2);
        else:
            # z component terms
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0]] += -Jz/4;
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[0],nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1]] +=  Jz/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0]] +=  Jz/4;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1]] += -Jz/4;
            # x+y component -> +- terms
            g2e[nloc*d+spin_inds[0],nloc*d+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[0]] += -Jx/2;
            g2e[nloc*d+spin_inds[1],nloc*d+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[1]] += -Jx/2;
            # repeat above with switched particle labels (pq|rs) = (rs|pq)
            g2e[nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] += -Jz/4;
            g2e[nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*d+spin_inds[0],nloc*d+spin_inds[0]] +=  Jz/4;
            g2e[nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[0],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] +=  Jz/4;
            g2e[nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[1],nloc*d+spin_inds[1],nloc*d+spin_inds[1]] += -Jz/4;
            g2e[nloc*(dp1)+spin_inds[1],nloc*(dp1)+spin_inds[0],nloc*d+spin_inds[0],nloc*d+spin_inds[1]] += -Jx/2;
            g2e[nloc*(dp1)+spin_inds[0],nloc*(dp1)+spin_inds[1],nloc*d+spin_inds[1],nloc*d+spin_inds[0]] += -Jx/2;

    # return
    if(block):
        return driver, builder;
    else:
        return h1e, g2e;

def Hsuper_polarizer(params_dict, block, to_add_to, verbose=0):
    '''
    Adds terms specific to the t<0 Hamiltonian in which the deloc e's, loc spins are
    confined and polarized by application of external fields Be, BFM

    However, this builds in terms of supersited dofs, rather than fermionic dofs


    Returns:
        if block is True: a tuple of DMRGDriver, MPO
        else: return a tuple of 1-body and 2-body Hamiltonian arrays
    '''

    # load data from json
    Vconf, Be, BFM = params_dict["Vconf"], params_dict["Be"], params_dict["BFM"];
    NL, NFM, NR, Nconf, Ne = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"], params_dict["Ne"];

    # fermionic sites and spin
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    TwoSz = params_dict["TwoSz"]; # total fermion spin in the z
    spin_strs = np.array(params_dict["spin_strs"]); # operator strings for each fermion spin
    spin_inds, nloc = np.array(range(len(spin_strs))), len(spin_strs);  # for summing over fermion spin

    # impurity spin
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd);
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);

    # classify site indices (spin not included)
    llead_sites = np.array([j for j in range(NL)]);
    conf_sites = np.array([j for j in range(Nconf)]);
    central_sites = np.array([j for j in range(NL,NL+NFM) ]);
    rlead_sites = np.array([j for j in range(NL+NFM,Nsites)]);
    all_sites = np.array([j for j in range(Nsites)]);

    # return objects
    if(block): # construct ExprBuilder
        driver, builder = to_add_to;
        if(driver.n_sites != Nsites): raise ValueError;
    else:
        raise NotImplementedError;

    # confining potential in left lead
    for j in conf_sites:
        for spin in spin_inds:
            if(block):
                builder.add_term(spin_strs[spin],[j,j],-Vconf); 

    # B field in the confined region ----------> ASSUMED IN THE Z
    # only within the region of confining potential
    for j in conf_sites:
        if(block):
            builder.add_term(spin_strs[0],[j,j],-Be/2);
            builder.add_term(spin_strs[1],[j,j], Be/2);

    # B field on the loc spins
    for j in central_sites:
        if(block):
            builder.add_term("Z",[j],BFM);

    # special case initialization
    if("BFM_first" in params_dict.keys()): # B field that targets 1st loc spin only
        BFM_first = params_dict["BFM_first"];
        j = central_sites[0];
        if(block):
            builder.add_term("Z",[j], BFM_first - BFM);
    if("Bsd" in params_dict.keys()): # B field on the j that couples to the first loc spin
        Bsd = params_dict["Bsd"];
        j = central_sites[0];
        if(block):
            builder.add_term(spin_strs[0],[j,j],-Bsd/2);
            builder.add_term(spin_strs[1],[j,j], Bsd/2);

    # return
    if(block):
        from pyblock2.driver.core import MPOAlgorithmTypes
        #mpo_from_builder = driver.get_mpo(builder.finalize(adjust_order=True,fermionic_ops="cdCD"),  algo_type=MPOAlgorithmTypes.FastBipartite);
        mpo_from_builder = driver.get_mpo(builder.finalize(),  algo_type=MPOAlgorithmTypes.FastBipartite);
        # in Huanchen's example, he uses passes
        # adjust_order=True, fermionic_ops="cdCD" to finalize
        # but I need to update block2 before I can do this
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;
 
    # special case initialization
    if("Bsd_x" in params_dict.keys()): # B in the x on the j that couples to 1st loc spin
        Bsd_x = params_dict["Bsd_x"];
        s = central_sites[0];
        if block:
            builder.add_term("cD",[s,s],-Bsd_x/2);
        else:
            h1e[nloc*s+spin_inds[0],nloc*s+spin_inds[1]] += -Bsd_x/2;
    if("Bcentral" in params_dict.keys()): # B field on all js coupled to loc spins
        Bcentral = params_dict["Bcentral"];
        for s in central_sites:
            if(block):
                builder.add_term(spin_strs[0],[s,s],-Bcentral/2);
                builder.add_term(spin_strs[1],[s,s], Bcentral/2);
            else:
                h1e[nloc*s+spin_inds[0],nloc*s+spin_inds[0]] += -Bcentral/2;
                h1e[nloc*s+spin_inds[1],nloc*s+spin_inds[1]] +=  Bcentral/2;

    # return
    if(block):
        mpo_from_builder = driver.get_mpo(builder.finalize(), iprint=verbose);
        return driver, mpo_from_builder;
    else:
        return h1e, g2e;
