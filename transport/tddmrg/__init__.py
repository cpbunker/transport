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
    assert(params_dict["te_type"]=="tdvp");
    print("\n\nSTART TIME EVOLUTION (te_type = "+params_dict["te_type"]+")\n\n","*"*50,"\n\n")
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
        if(params_dict["te_type"] == "tdvp"): krylov_subspace = 40;
        tevol_mps_inst = driver_inst.td_dmrg(mpo_inst, tevol_mps_inst, 
                delta_t=complex(0,time_step), target_t=complex(0,time_update),
                bond_dims=params_dict["bdim_t"], cutoff=params_dict["cutoff"], te_type=params_dict["te_type"],krylov_subspace_size=krylov_subspace,
                final_mps_tag=str(int(100*total_time)), iprint=the_verbose);

        # observables
        check_func(params_dict,tevol_mps_inst,driver_inst,mpo_inst,total_time, True);
        plot.snapshot_bench(tevol_mps_inst, driver_inst, params_dict, save_name, total_time, True);

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

def get_onehop(eris_or_driver, whichsite, block, verbose=0):
    '''
    '''
    if(block): builder = eris_or_driver.expr_builder()
    else:
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite+1],1.0);
        builder.add_term("cd",[whichsite+1,whichsite],1.0);
        builder.add_term("CD",[whichsite,whichsite+1],1.0);
        builder.add_term("CD",[whichsite+1,whichsite],1.0);
    else:
        h1e[nloc*(whichsite)+0,nloc*(whichsite+1)+0] += 1.0;
        h1e[nloc*(whichsite+1)+0,nloc*(whichsite)+0] += 1.0;
        h1e[nloc*(whichsite)+1,nloc*(whichsite+1)+1] += 1.0;
        h1e[nloc*(whichsite+1)+1,nloc*(whichsite)+1] += 1.0;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);


def get_occ2(eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing n^2
    '''
    raise NotImplementedError
    if(block): builder = eris_or_driver.expr_builder()
    else:
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cdcd",[whichsite,whichsite],1.0);
        builder.add_term("cdCD",[whichsite,whichsite],1.0);
        builder.add_term("CDcd",[whichsite,whichsite],1.0);
        builder.add_term("CDCD",[whichsite,whichsite],1.0);
    else:
        # g_pqrs a_p^+ a_q a_r^+ a_s
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+1] += 1.0;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1] += 1.0;
        # switch particle labels
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        g2e[nloc*whichsite+0,nloc*whichsite+0,nloc*whichsite+1,nloc*whichsite+1] += 1.0;
        g2e[nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1,nloc*whichsite+1] += 1.0;

        # - delta_qr a_p^+ a_s
        h1e[nloc*whichsite+0,nloc*whichsite+0] += 1.0;
        h1e[nloc*whichsite+0,nloc*whichsite+1] += 1.0;
        h1e[nloc*whichsite+1,nloc*whichsite+0] += 1.0;
        h1e[nloc*whichsite+1,nloc*whichsite+1] += 1.0;

    # return
    if(block): return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);   

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
    Constructs an operator (either MPO or ERIs) representing <Sz> of site whichsite
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
    assert(block);
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
    assert(block);
    builder = eris_or_driver.expr_builder();

    # construct
    builder.add_term("ZZ",[whichsite,whichsite], 1.0);

    return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    
def get_S2(eris_or_driver, whichsites, block, verbose=0):
    '''
    '''
    assert(block);
    builder = eris_or_driver.expr_builder();

    # construct
    which1, which2 = whichsites;
    for jpair in [[which1,which1], [which1,which2], [which2,which1], [which2,which2]]:
        builder.add_term("ZZ",jpair,1.0);
        builder.add_term("PM",jpair,0.5);
        builder.add_term("MP",jpair,0.5);

    # return
    return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    
def S2_wrapper(psi, eris_or_driver, whichsites, block):
    '''
    '''
    if(whichsites[0] == whichsites[1]): return np.nan;# sites have to be distinct
    
    op = get_S2(eris_or_driver, whichsites, block);
    ret = compute_obs(psi, op, eris_or_driver);
    if(abs(np.imag(ret)) > 1e-12): print(ret); raise ValueError;
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
    if(abs(np.imag(ret)) > 1e-12): print(ret); raise ValueError;
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
    if(abs(np.imag(ret)) > 1e-12): print(ret); raise ValueError;
    return np.real(ret);

def get_concurrence(eris_or_driver, whichsites, symm_block, add_ident, block, verbose=0):
    '''
    MPO corresponding to the action of the operator
    \sigma_y^1 \otimes \sigma_y^2 
    where 1, 2 denote the impurity spins, which in this case must be spin-1/2
    This operator is needed to compute concurrence and pseudoconcurrence, below
    
    Since the full form of this operator does not conserve total TwoSz, we break it up
    into blocks which do, and only return one of such "symmetry blocks" at a time
    '''
    assert(block);
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
    #print("\n"*20)
    from pyblock2.driver.core import MPOAlgorithmTypes
    ret = eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"),add_ident=add_ident, iprint=0)# cutoff=0.0, algo_type=MPOAlgorithmTypes.SVD, iprint=2);
    #print(type(ret))
    #print(type(ret.prim_mpo))
    #assert False
    return ret

def concurrence_wrapper(psi,eris_or_driver, whichsites, block, use_b3 = False):
    '''
    Concurrence is defined by Eqs 4 and 7 of https://doi.org/10.1103/PhysRevLett.80.2245
    It involves taking the complex conjugate of the coefficients of the ket |\psi> (this 
    is NOT the same as taking the bra, which complex conjugates the coefficients AND takes
    the basis states from kets to bras). 
    C=1 is maximal entanglement
    
    Sums expectation values across TwoSz=+2, 0, -2 symmetry blocks to find concurrence

    NB also for s>1/2 we will need to define which levels are the qubits
    '''
    if(whichsites[0] == whichsites[1]): return np.nan;# sites have to be distinct
    raise NotImplementedError; # not trusted

    # exp vals across symmetry blocks
    sblocks = [-2,0,2];
    sterms = [];
    if(use_b3): # use pyblock3 to calculate
        psi_b3 = MPSTools.from_block2(psi); #  convert the MPS to pyblock3
        psi_star = psi_b3.conj(); # complex conjugates coefs of ket
        for sblock in sblocks:
            mpo = get_concurrence(eris_or_driver, whichsites, sblock, False, block);
            mpo_b3 = MPOTools.from_block2(mpo); # convert the MPO to pyblock3
            # pyblock3 construction for determining the expectation value of an MPO:
            norm_b3 = np.dot(psi_b3.conj(),psi_b3);
            sterms.append( np.dot(psi_b3.conj(), mpo_b3 @ psi_star)/norm_b3 );

    else: # use block2 to calculate
        for sblock in sblocks:
            psi_star = eris_or_driver.copy_mps(psi, tag="PSI-STAR");
            psi_star.conjugate(); # in-place
            mpo = get_concurrence(eris_or_driver, whichsites, sblock, True, block);
            # use normal block2 construction for determining the expectation value of an MPO
            norm = eris_or_driver.expectation(psi, eris_or_driver.get_identity_mpo(), psi);
            sterms.append(eris_or_driver.expectation(psi, mpo, psi_star)/norm);
       
    # return
    ret = np.sqrt(np.conj(np.sum(sterms))*np.sum(sterms));
    if(abs(np.imag(ret)) > 1e-12): print(ret); raise ValueError;
    return np.real(ret);
    
def pseudoconcurrence_wrapper(psi,eris_or_driver, whichsites, block):
    '''
    I call this the "pseudo-concurrence", meaning we construct the same \sigma_y^1 \otimes \sigma_y^2 
    MPO as in the concurrence above, but we do not take the complex conjugate of the ket coefficients
    anymore (thus we do not have to use pyblock3 at all!)
    In other words this is just a normal expectation value of the operator produced by get_concurrence
    Pseudoconcurrence only measures entanglement when all the ket coefficients are real valued!
    '''
    if(whichsites[0] == whichsites[1]): return np.nan;# sites have to be distinct
    raise NotImplementedError; # not physically meaningful
    
    # exp vals across symmetry blocks
    sblocks = [-2,0,2];
    sterms = [];
    for sblock in sblocks:
        mpo = get_concurrence(eris_or_driver, whichsites, sblock, True, block);
        # since complex conjugation of the state is not required, we can use
        # the normal block2 construction for determining the expectation value of an MPO
        sterms.append( compute_obs(psi, mpo, eris_or_driver)); # already/norm

    # return
    ret = np.sqrt(np.conj(np.sum(sterms))*np.sum(sterms));
    if(abs(np.imag(ret)) > 1e-12): print(ret); raise ValueError;
    return np.real(ret);

def get_pcurrent(eris_or_driver, whichsites, spin, block, verbose=0):
    '''
    MPO for particle current from whichsites[0] to whichsites[1]
    positive is rightward, associated with positive bias st left lead chem potential
    is higher) 

    Ultimately, we want this for conductance. The formula is found in 
    Garnet's coupled cluster dynamics paper, JCP 2021, Eqs 69-70
    G/G0 = \pi <J>/(Vb/e), where Vb/e is a VOLTAGE, and
    <J> =  e/\hbar * hopping * i * \sum_sigma 
    < c_j+1,\sigma^\dagger c_j,\sigma - c_j,\sigma^\dagger c_j+1,\sigma >
    HOWEVER for convenience we wait till plotting to apply factor 
    \pi e/\hbar * hopping/(Vb/e)

    Args:
    eris_or_driver, Block2 driver
    whichsites, list of site indices. must be ordered, so that
    add_term( "cd", whichsites ) represents NEGATIVE current
    spin, int 0 or 1, meaning up or down current
    '''
    if(whichsites[1]-whichsites[0]!=1): raise ValueError;
    if(spin==0): spinstr = "cd";
    elif(spin==1): spinstr = "CD";
    else: raise ValueError;

    if(block):# construct MPO
        builder = eris_or_driver.expr_builder();
        builder.add_term(spinstr, whichsites, complex(0,-1)); # c on left, d on right = negative particle current
        builder.add_term(spinstr, whichsites[::-1], complex(0,1)); # c on right, d on left = positive particle current
        return eris_or_driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"), iprint=verbose);
    else: # construct ERIs
        Nspinorbs = len(eris_or_driver.h1e[0]);
        nloc = 2;
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=complex), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=complex);
        h1e[nloc*whichsites[0]+spin,nloc*whichsites[1]+spin] += complex(0,-1.0);
        h1e[nloc*whichsites[1]+spin,nloc*whichsites[0]+spin] += complex(0,1.0);
        return tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff, imag_cutoff = 1e-12);

def conductance_wrapper(psi, eris_or_driver, whichsite, block, verbose=0):
    '''
    Consider site whichsite. This wrapper:
    1) sums the spin currents from whichsite-1 to whichsite (LEFT part)
    2) sums the spin currents from whichsite to whichsite+1 (RIGHT part)
    3) averages over the results of 1 and 2 to find the current "through" whichsite
    Later we multiply this by  \pi e/\hbar * hopping/(Vb/e) to make it *conductance*
    '''
    if(block): compute_func = compute_obs;
    else: compute_func = tdfci.compute_obs;

    # left part
    pcurrent_left = 0.0;
    for spin in [0,1]:
        left_mpo = get_pcurrent(eris_or_driver, [whichsite-1, whichsite], spin, block, verbose=verbose);
        left_val = compute_func(psi, left_mpo, eris_or_driver);
        pcurrent_left += left_val;

    # right part
    #pcurrent_right = 0.0;
    #for spin in [0,1]:
    #right_mpo = get_pcurrent(eris_or_driver, [whichsite, whichsite+1], spin, block, verbose=verbose);
    #right_val = compute_func(psi, right_mpo, eris_or_driver);
    #pcurrent_right += right_val;

    # average
    ret = complex(1,0)*(pcurrent_left); # must add  e/\hbar * th/Vb later
    if(abs(np.imag(ret)) > 1e-12): print(ret); raise ValueError;
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

def H_SIAM_builder(params_dict, block, scratch_dir="tmp",verbose=0):
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
    tl, th, Vg, U, Vb = params_dict["tl"], params_dict["th"], params_dict["Vg"], params_dict["U"], params_dict["Vb"];
    NL, NR = params_dict["NL"], params_dict["NR"];
    Nsites = NL+1+NR;
    Ne=Nsites;
    if(block): assert(Ne%2 ==0); # need even number of electrons for TwoSz=0
    TwoSz = 0;

    # classify site indices (spin not included)
    llead_sites = np.arange(NL);
    central_sites = np.arange(NL,NL+1);
    rlead_sites = np.arange(NL+1,Nsites);
    all_sites = np.arange(Nsites);

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
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
    TwoSd = params_dict["TwoSd"]; # impurity spin magnitude, doubled to be an int
    TwoSdz_ladder = (2*np.arange(TwoSd+1) -TwoSd)[::-1];
    n_fer_dof = 4;
    n_imp_dof = len(TwoSdz_ladder);
    assert(TwoSd == 1); # for now, to get degeneracies right

    # classify site indices (spin not included)
    llead_sites = np.arange(NL);
    central_sites = np.arange(NL,NL+NFM);
    rlead_sites = np.arange(NL+NFM,Nsites);
    all_sites = np.arange(Nsites);

    # construct ExprBuilder
    if(block):
        if(params_dict["symmetry"] == "Sz"):
            driver = core.DMRGDriver(scratch="./block_scratch/"+scratch_dir[:-4], symm_type=core.SymmetryTypes.SZ|core.SymmetryTypes.CPX, n_threads=4);
            # using complex symmetry type, as above, seems linked to
            # Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM warnings
            # but only when TwoSz is input correctly
            # in latter case, we get a floating point exception even when complex sym is turned off!
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
        else:
            assert(BFM_first==0.0);
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
        #pass;

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

    # load data from json
    Vconf, Be, BFM = params_dict["Vconf"], params_dict["Be"], params_dict["BFM"];
    NL, NFM, NR, Nconf = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Nconf"];
    Nbuffer = 0;
    if("Nbuffer" in params_dict.keys()): Nbuffer = params_dict["Nbuffer"];

    # fermionic sites and spin
    Nsites = Nbuffer+NL+NFM+NR; # number of j sites in 1D chain
    Ne = params_dict["Ne"];
    TwoSz = params_dict["TwoSz"]; # fermion spin + impurity spin

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
    if("Bent" in params_dict.keys() and len(central_sites)==2): # B field that entangles 2 loc spins
        Bent = params_dict["Bent"];
        for j in central_sites[:-1]:
            builder.add_term("ZZ",[j,j+1],-Bent);
            builder.add_term("PM",[j,j+1],-Bent/2);
            builder.add_term("MP",[j,j+1],-Bent/2);

    # return
    mpo_from_builder = driver.get_mpo(builder.finalize(adjust_order=True, fermionic_ops="cdCD"));
    return driver, mpo_from_builder;
 
    # special case initialization
    if("Bsd_x" in params_dict.keys()): # B in the x on the j that couples to 1st loc spin
        Bsd_x = params_dict["Bsd_x"];
        s = central_sites[0];
        builder.add_term("cD",[s,s],-Bsd_x/2);


