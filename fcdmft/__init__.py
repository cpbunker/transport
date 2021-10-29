'''
Compute the many body impurity Green's function using DMFT
For DMFT overview see: https://arxiv.org/pdf/1012.3609.pdf (Zgid, Chan paper)

fcdmft package due to Tianyu Zhu et al, Caltech

Wrapper functions due to Christian Bunker, UF
'''

#### setup the fcdmft the package

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

from fcdmft import dmft
from fcdmft.solver import scf_mu as scf
from transport import fci_mod
from pyscf import gto

import numpy as np

########################################################################
#### my wrappers that access package routines

def kernel(SR_1e, SR_2e, coupling, leadsite, verbose = 0): # main driver
    '''
    Driver of DMFT calculation for
    - scattering region, treated at high level, repped by SR_1e and SR_2e
    - noninteracting leads, treated at low level, repped by leadsite

    Difference between my code, Tianyu's code (e.g. fcdmft/dmft/gwdmft.kernel())
    is that the latter assumes periodicity, and so only takes local hamiltonian
    of impurity, and does a self-consistent loop. In contrast, this code
    specifies the environment and skips scf 
    '''

    # check inputs and unpack
    assert( isinstance(leadsite, site) );
    n_imp_orbs = np.shape(SR_1e)[0];
    SR_1e = np.array([SR_1e]); # up spin only
    SR_2e = np.array([SR_2e]);

    # for now just put defaults here
    Ha2eV = 27.211386; # hartree to eV
    iter_depth = 10;
    n_bath_orbs = 3;
    n_core = 0; # core orbitals
    filling = 0.5; # e's per spin orb
    chem_pot = 4*np.pi*np.pi*filling*filling; # fermi energy, at zero temp
    chem_pot = 0.0/Ha2eV; # orbitals below will be filled, above empty
    nao = 1; # pretty sure this does not do anything
    max_mem = 8000;
    n_orbs = n_imp_orbs + 2*n_bath_orbs;

    # surface green's function in the leads
    if(verbose): print("\n1. Surface Green's function");
    
    # hybridization defines interaction between imp and leads
    if(verbose): print("\n2. Bath discretization");
    hyb = np.empty(np.shape(g_nona), dtype = complex); # unfilled
    hyb[0] = np.dot(coupling.T, np.dot(g_nona, coupling) );
    if(verbose): print(" - hyb(E) = \n", hyb[0,:,:,0]);

    # convergence loop would start here

    # first attempt at bath disc
    # outputs n_bath_orbs bath energies, for each imp orb
    bath = dmft.gwdmft.get_bath_direct(hyb, leadsite.energies, n_bath_orbs);
    if(verbose): print(" - bath energies = ", bath[1]); assert False;

    # optimize bath disc

    # construct manybody hamiltonian of imp + bath
    if(verbose): print("\n4. Combine impurity and bath");
    h1e_imp, h2e_imp = dmft.gwdmft.imp_ham(SR_1e, SR_2e, *bath, n_core); # adds in bath states
        
    # find manybody gf of imp + bath
    # I hope this is equivalent to Zgid paper eq 28
    if(verbose): print("\n5. Impurity Green's function");
    meanfield = dmft.dmft_solver.mf_kernel(h1e_imp, h2e_imp, chem_pot, nao, np.array([np.eye(n_orbs)]), max_mem, verbose = 0);
    
    # use fci (which assumes only one kind of spin) to get Green's function
    assert(len(np.shape(meanfield.mo_coeff)) == 2); # ie no spin dof
    Gimp = dmft.dmft_solver.fci_gf(meanfield, leadsite.energies, leadsite.iE, verbose = verbose);
    return Gimp;


def surface_gf(energies, iE, H, V, tol = 1e-3, max_cycle = 10000, verbose = 0):
    '''
    surface dos in semi-infinite noninteracting lead, formula due to Haydock, 1972

    Args:
    - energies, 1d arr, energy range
    - iE, float, imag part to add to energies, so that gf is off real axis
    - H, 2d arr, repeated diagonal component of lead ham
    - V, 2d arr, repeated off diag component of lead ham
    '''

    # check inputs
    assert(len(np.shape(H)) == 3); # should contain spin
    assert(np.shape(H) == np.shape(V));
    assert(not np.any(np.imag(V) ) );

    # quick shortcut for diag inputs
    H_is_diag = (np.trace(H[0]) == np.sum(H.flat) ); # is a diagonal matrix
    H_is_same = not np.any(np.diagonal(H[0]) - H[0,0,0]); # all diags equal
    V_is_diag = (np.trace(V[0]) == np.sum(V.flat) );
    V_is_same = not np.any(np.diagonal(V[0]) - V[0,0,0]);

    # if these are all true, can use closed form eq for diag gf
    if(H_is_diag and H_is_same and V_is_diag and V_is_same):

        if(verbose): print(" - Diag shortcut");
        energies = energies + iE;
        gf = np.zeros((*np.shape(H),len(energies) ), dtype = complex);
        
        for i in range(np.shape(H[0])[0]): # iter over diag
            pref = (energies - H[0,0,0])/(2*V[0,0,0]);
            gf[0,i,i,:] = pref-np.lib.scimath.sqrt(pref*pref*(1-4*V[0,0,0]*V[0,0,0]/np.power(energies-H[0,0,0],2)));

    else: # do convergence

        # initial guess is inv(E+iE - H)
        # fcdmft code can get these energy dependent inverses using get_gf
        gf = dmft.dmft_solver.get_gf(H, np.zeros((*np.shape(H),len(energies))), energies, iE);

        # start convergence loop
        conv = False;
        cycle = 0
        while not conv:

            # last guess
            cycle += 1;
            gf0 = gf;

            # update
            sigma = dot_spinful_arrays(gf0, np.array([V[0].T]) ); #gf0 * V^\dagger
            sigma = dot_spinful_arrays(sigma, V, backwards = True);
            gf = dmft.dmft_solver.get_gf(H, sigma, energies, iE);

            # check convergence
            if( np.max(abs( gf[0,0,0,:] - gf0[0,0,0,:] )) < tol ):
                if(verbose): print(" - final cycle = ",cycle, abs(gf[0,0,0,:] - gf0[0,0,0,:]));
                conv = True;
            elif( cycle >= max_cycle ):
                if(verbose): print(" - reached max cycle = ", cycle);
                conv = True;

    assert(np.any(np.imag(gf)));
    return gf;



def junction_gf(g_L, t_L, g_R, t_R, E, H_SR):
    '''
    Given the surface green's function in the leads, as computed above,
    compute the gf at the junction between the leads, aka scattering region.
    NB the junction has its own local physics def'd by H_SR

    Args:
    - g_L, 1d arr, left lead noninteracting gf at each E
    - t_L, 2d arr, left lead coupling, constant in E
    - g_R, 1d arr, right lead noninteracting gf at each E
    - t_R, 2d arr, right lead coupling, constant in E
    - E, 1d array, energy values
    '''

    # check inputs
    assert(np.shape(t_L) == np.shape(H_SR) );
    assert(len(g_L) == len(E) );

    # vectorize by hand
    G = [];
    for Ei in range(len(E)): # do for each energy

        # gL, gR as of now are just numbers at each E, but should be matrices
        # however since leads are defined to be noninteracting, just identity matrices
        g_Lmat = g_L[Ei]*np.eye(*np.shape(H_SR));
        g_Rmat = g_R[Ei]*np.eye(*np.shape(H_SR));

        # integrate out leads, using self energies
        Sigma_L = np.dot(np.linalg.inv(g_Lmat),-t_L);
        Sigma_L = np.dot( -t_L, Sigma_L);
        Sigma_R = np.dot(np.linalg.inv(g_Rmat),-t_R);
        Sigma_R = np.dot( -t_R, Sigma_R);

        # local green's function
        G.append(np.linalg.inv( E[Ei]*np.eye(*np.shape(H_SR)) - H_SR - Sigma_L - Sigma_R));

    return np.array(G);


#### utils
def dot_spinful_arrays(a1, a2, backwards = False):
    '''
    given an array of shape (spin, norbs, norbs, nfreqs)
    and another array , either
    - an operator, shape (spin, norbs, norbs), indep of freq
    '''

    # unpack sizes
    spin, norbs, _, nfreqs = np.shape(a1);

    # return var
    result = np.zeros_like(a1, dtype = complex);

    # screen by kind of second array
    if( np.shape(a2) == (spin, norbs, norbs) ): # freq indep operator
        for s in range(spin):
            for iw in range(nfreqs):
                if not backwards:
                    result[s,:,:,iw] = np.dot(a1[s,:,:,iw], a2[s]);
                else:
                    result[s,:,:,iw] = np.dot(a2[s], a1[s,:,:,iw]);

    elif(False):
        pass;

    else: raise(ValueError("a2 "+str(np.shape(a2))+" is of wrong size") );

    return result;


def h1e_to_gf(E, h1e, g2e, nelecs, bdims, noises):
    '''
    Use dmrg routines in the solvers module to extract a green's function from
    a second quantized hamiltonian
    given an array of shape (spin, norbs, norbs, nfreqs)
    and another array , either
    - an operator, shape (spin, norbs, norbs), indep of freq
    '''

    # check inputs

    # unpack
    nsites = np.shape(h1e)[0];

    # init GFDMRG object
    dmrg_obj = solver.gfdmrg.GFDMRG();

    # input hams
    pointgroup = 'c1';
    Ecore = 0.0;
    isym = None;
    orb_sym = None;
    dmrg_obj.init_hamiltonian(pointgroup, nsites, sum(nelecs), nelecs[0] - nelecs[1], isym, orb_sym, Ecore, h1e, g2e);

    # get greens function
        # default params taken from fcdmft/examples/DMRG_GF_test/run_dmrg.py
    gmres_tol = 1e-9;
    conv_tol = 1e-8;
    nsteps = 10;
    cps_bond_dims=[1500];
    cps_noises=[0];
    cps_tol=1E-13;
    cps_n_steps=20;
    idxs = None;
    eta = None;
    dfparams = gmres_tol, conv_tol, nsteps, cps_bond_dims, cps_noises, cps_conv_tol, cps_n_steps, idxs, eta;
    G = dmrg_obj.greens_function(bdims, noises, *dfparams, E, None);
    # unpack sizes
    spin, norbs, _, nfreqs = np.shape(a1);

    return G;
    
    

