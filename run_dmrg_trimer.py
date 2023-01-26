'''
Christian Bunker
M^2QM at UF
January 2023

Toy model of molecule with itinerant electrons
solved in time-dependent DMRG (approximate many body QM) method in transport/tddmrg
'''

from transport import fci_mod
from transport.fci_mod import ops_dmrg

import numpy as np
import matplotlib.pyplot as plt
import itertools

from pyblock3.algebra.mpe import MPE

# top level
verbose = 3;
np.set_printoptions(precision = 4, suppress = True);

#### utils

def print_H_alpha(H) -> None:
    assert(len(np.shape(H)) == 4);
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [i]: # [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

def get_h1e(mytm, myB_mm, myB_elec, myJH, myJK, debug = 0) -> np.ndarray:
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

    '''
    assert n_elecs == (1,0); # dmrg will only work for this!
    assert n_loc_dof % 2 == 0;
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!

    # return var
    h1e = np.zeros((n_sys_orbs, n_sys_orbs, n_loc_dof, n_loc_dof));

    #### fermionic terms
    # spin-independent hopping between n.n. sys orbs
    for sysi in range(n_sys_orbs-1):
        # iter over local dofs (up, down, etc)
        for loci in range(n_loc_dof):
            h1e[sysi,sysi+1,loci,loci] += -mytm; 
            h1e[sysi+1,sysi,loci,loci] += -mytm;
    if(n_sys_orbs > 2) and False: # last to first hopping
        for loci in range(n_loc_dof): 
            h1e[0,-1,loci,loci] += -mytm; 
            h1e[-1,0,loci,loci] += -mytm;

    #### spin terms
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);

    # Zeeman terms
    if(debug): print("Zeeman"); 
    for sysi in range(n_sys_orbs):
        # have to iter over local dofs paticle-by-particle
        # iter over all (2s+1)^n_mols many-body mol spin states
        for mol_statei in range(len(mol_states)):
            Sztot = sum(mol_states[mol_statei]);
            # iter over electron spin 
            for sigma in range(2):
                loci = 2*mol_statei+sigma;
                h1e[sysi,sysi,loci,loci] += myB_elec*(1/2-sigma) + myB_mm*Sztot;
                if(debug and sysi == 0): print("->",loci,1/2-sigma,mol_states[mol_statei],h1e[sysi,sysi,loci,loci]);

    # Heisenberg - regardless of elec location, couples mol spins
    if(debug): print("Heisenberg"); 
    for sysi in range(n_sys_orbs):

        # iter over many-body mol spin states twice
        for mol_statei in range(len(mol_states)):
            for mol_statej in range(len(mol_states)):
                # difference between states
                n_different = np.count_nonzero(mol_states[mol_statei]-mol_states[mol_statej]);
                if(n_different in [0,2]):

                    # iter over ind'l mols in spin states twice
                    for mola in range(n_mols):
                        for molb in range(mola):
                            # nearest neighbors, first-last
                            if(mola - molb == 1 or (mola == n_mols-1) and molb == 0):
                                
                                # quantum numbers
                                Szi_a = mol_states[mol_statei][mola];
                                Szi_b = mol_states[mol_statei][molb];
                                Szj_a = mol_states[mol_statej][mola];
                                Szj_b = mol_states[mol_statej][molb];

                                # S^z_a S^z_b - couples state to itself
                                if(mol_statei == mol_statej):
                                    # add term to both elec spin channels
                                    for sigma in range(2):
                                        loci = 2*mol_statei+sigma;
                                        h1e[sysi,sysi,loci,loci] += myJH*Szi_a*Szi_b;

                                # S^+_a S^-_b couples spin flipped states
                                if(Szi_a - Szj_a==1 and Szi_b-Szj_b==-1):
                                    if(debug and sysi == 0): print("->",2*mol_statei,mol_states[mol_statei],2*mol_statej,mol_states[mol_statej]);
                                    # add term to both elec spin channels
                                    for sigma in range(2):
                                        loci = 2*mol_statei+sigma;
                                        locj = 2*mol_statej+sigma;
                                        h1e[sysi,sysi,loci,locj] += (1/2)*myJH;
                                        # hc
                                        h1e[sysi,sysi,locj,loci] += (1/2)*myJH;

    # Kondo exchange - couples elec to molecule it is on
    if(debug): print("Kondo");
    for moli in range(n_mols):

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
                            h1e[moli,moli,loci,loci] += myJK*mol_states[mol_statei][moli]*(1/2-sigma);

                    # S^+ - couples statei to statej with moli flipped up by one
                    if(mol_states[mol_statei][moli]+1 == mol_states[mol_statej][moli]):
                        # all other have to be the same
                        if(n_different == 1):
                            if debug: print("->",2*mol_statei,mol_states[mol_statei],2*mol_statej,mol_states[mol_statej]);
                            # couple statei with elec up to statej wth elec down
                            h1e[moli,moli,2*mol_statei,2*mol_statej+1] += (1/2)*myJK;
                            # hc
                            h1e[moli,moli,2*mol_statej+1,2*mol_statei] += (1/2)*myJK;

    # return
    return h1e;

def get_SaSigb(aindex,bindex,spatial_orbs) -> np.ndarray:
    '''
    Get the operator mol spin S_a dotted into elec spin sigma on site b
    For calculating F_ab (Kumar Eq (5)
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

                # S^+ - couples statei to statej with moli flipped up by one
                if(mol_states[mol_statei][aindex]+1 == mol_states[mol_statej][aindex]):
                    # all other have to be the same
                    if(n_different == 1):
                        # couple statei with elec up to statej wth elec down
                        SaSigb[aindex,bindex,2*mol_statei,2*mol_statej+1] += (1/2);
                        # hc
                        SaSigb[aindex,bindex,2*mol_statej+1,2*mol_statei] += (1/2);

    # return
    return SaSigb;
        

def get_SaSb(aindex,bindex,spatial_orbs) -> np.ndarray:
    '''
    Get the operator mol spin S_a dotted into mol spin S_b
    For calculating D_ab (Kumar Eq (5)
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    if(aindex == bindex): raise ValueError;
    
    # construct as 4d in the spatial orbs, mol_states basis
    SaSb = np.zeros((spatial_orbs,spatial_orbs,len(mol_states),len(mol_states)));

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
                    # add term to all spatial blocks
                    for spacei in range(spatial_orbs):
                        SaSb[spacei,spacei,mol_statei,mol_statej] += Szi_a*Szi_b;

                # S^+_a S^-_b couples spin flipped states
                if(Szi_a - Szj_a==1 and Szi_b-Szj_b==-1):
                    #print("->",2*mol_statei,mol_states[mol_statei],2*mol_statej,mol_states[mol_statej]);
                    # add term to all spatial blocks
                    for spacei in range(spatial_orbs):
                        SaSb[spacei,spacei,mol_statei,mol_statej] += (1/2);
                        # hc
                        SaSb[spacei,spacei,mol_statej,mol_statei] += (1/2);

    # return                       
    return SaSb;


##################################################################################
#### toy model of molecule

# phys params, must be floats
tm = 0.0; # hopping within molecule
gfactor = 2; # electron g factor
B_by_mu = 0.2*tm; # B field / Bohr magneton
JH = 1.0;
JK = 0.01*tm;

# electrons
n_mols = 3; # number of magnetic molecules
s_mols = 1/2; # spin of the mols
n_elecs = 1;
n_loc_dof = int((2**n_elecs)*((2*s_mols+1)**n_mols));
n_elecs = (n_elecs,0); # spin blind

#### hamiltonian
n_sys_orbs = 3; # = n_mols # molecular orbs in the sys
n_fer_orbs = n_loc_dof*n_sys_orbs; # total fermionic orbs

#Sab_arr = get_Sab(0,2,n_sys_orbs);
#print_H_alpha(Sab_arr);
a,b = 1,0;
SaSigb_arr = get_SaSigb(a,b,n_sys_orbs);
print(a,b)
print(SaSigb_arr[a,b]);
print(SaSigb_arr[2,2]);
assert False

#### hamiltonian
hilbert_size = n_loc_dof**n_fer_orbs;
bdims = 5*n_fer_orbs**2*np.array([1.0,1.2,1.4]);
bdims = list(bdims.astype(int));
harr = get_h1e(tm, gfactor*B_by_mu, gfactor*B_by_mu,JH,JK,debug=verbose);
if(verbose): print("1. Hamiltonian\n-h1e = \n");
print_H_alpha(harr);
harr = fci_mod.mat_4d_to_2d(harr);
garr = np.zeros((len(harr),len(harr),len(harr),len(harr)));

#### DMRG solution
if(verbose): print("2. DMRG solution");

# MPS ansatz
h_obj, h_mpo, psi_init = fci_mod.arr_to_mpo(harr, garr, n_elecs, bdims[0]);
if verbose: print("- built H as compressed MPO: ", h_mpo.show_bond_dims() );
E_init = ops_dmrg.compute_obs(h_mpo, psi_init);
if verbose: print("- guessed gd energy = ", E_init);

# MPS ground state
dmrg_mpe = MPE(psi_init, h_mpo, psi_init);
# MPE.dmrg method controls bdims,noises, n_sweeps,conv tol (tol),verbose (iprint)
# noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
dmrg_obj = dmrg_mpe.dmrg(bdims=bdims, tol = 1e-8, iprint=-5);
if verbose: print("- variational gd energy = ", dmrg_obj.energies[-1]);

# MPS state -> observables
mps_occs = np.zeros((n_fer_orbs//2,));
mps_szs = np.zeros((n_fer_orbs//2,));
for orbi in range(0,n_fer_orbs,2):
    occ_mpo = h_obj.build_mpo(ops_dmrg.occ(np.array([orbi,orbi+1]), n_fer_orbs));
    sz_mpo  = h_obj.build_mpo(ops_dmrg.Sz( np.array([orbi,orbi+1]), n_fer_orbs));
    mps_occs[orbi//2] = ops_dmrg.compute_obs(occ_mpo,dmrg_mpe.ket);
    mps_szs [orbi//2] = ops_dmrg.compute_obs(sz_mpo, dmrg_mpe.ket);

print(mps_occs);
print(mps_szs);


