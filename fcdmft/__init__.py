'''
fcdmft package due to Tianyu Zhu et al, Caltech

Wrapper functions due to Christian Bunker, UF, October 2021

Compute the many body impurity Green's function using DMFT
For DMFT overview see: https://arxiv.org/pdf/1012.3609.pdf (Zgid, Chan paper)
'''

#### setup the fcdmft the package

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

from fcdmft import dmft
from fcdmft.solver import scf_mu as scf

import numpy as np

########################################################################
#### drivers

def kernel(energies, iE, SR_1e, SR_2e, LL, RL, solver = "cc", n_bath_orbs = 4, verbose = 0): # main driver
    '''
    Driver of DMFT calculation for
    - scattering region, treated at high level, repped by SR_1e and SR_2e
    - noninteracting leads, treated at low level, repped by leadsite

    Difference between my code, Tianyu's code (e.g. fcdmft/dmft/gwdmft.kernel())
    is that the latter assumes periodicity, and so only takes local hamiltonian
    of impurity, and does a self-consistent loop. In contrast, this code
    specifies the environment and skips scf

    NB chem potential fixed at 0 as convention

    Args:
    energies, 1d arr of energies
    iE, float, complex part to add to energies
    imp_occ, int, target impurity occupancy
    SR_1e, (spin,n_imp_orbs, n_imp_orbs) array, 1e part of scattering region hamiltonian
    SR_2e, 2e part of scattering region hamiltonian
    coupling, (spin, n_imp_orbs, n_imp_orbs) array, couples SR to lead
    LL, tuple of:
        LL_diag, (spin, n_imp_orbs, n_imp_orbs) array, onsite energy for left lead blocks
        LL_hop, (spin, n_imp_orbs, n_imp_orbs) array, hopping for left lead blocks
    RL, similarly for the right lead

    Optional args:
    solver, string, tells how to compute imp MBGF. options:
    - cc
    - fci, if n_orbs is sufficiently small
    n_bath_orbs, int, how many disc bath orbs to make

    Returns:
    G, 3d arr, (spin, norbs, norbs, nw) ie for each spin <nu| G(E) |nu'>
    where nu, nu' run over all quantum numbers except spin. Typically do ASU
    so there is only up sin.
    '''
    
    # check inputs
    assert(energies[0] < energies[-1]);
    assert(np.shape(SR_1e) == np.shape(SR_2e)[:3]);
    assert(np.shape(SR_1e) == np.shape(LL[0]));
    assert(np.shape(SR_1e) == np.shape(LL[1]));
    assert(np.shape(SR_1e) == np.shape(LL[2]));
    assert(len(LL) == len(RL) );
    
    # unpack
    spin, n_imp_orbs, _ = np.shape(SR_1e); # size of arrs on spin, norb axes
    LL_diag, LL_hop, LL_coup, mu_L = LL;
    RL_diag, RL_hop, RL_coup, mu_R = RL;
    n_core = 0; # core orbitals
    nao = 1; # pretty sure this does not do anything
    max_mem = 8000;
    n_orbs = n_imp_orbs + 2*n_bath_orbs;

    # surface green's function in the leads
    if(verbose): print("\n1. Surface Green's function");
    LL_surf = surface_gf(energies, iE, LL_diag, LL_hop, verbose = verbose);
    RL_surf = surface_gf(energies, iE, RL_diag, RL_hop, verbose = verbose);

    # hybridization between leads and SR
    # hyb = V*surface_gf*V^\dagger
    if(verbose): print("\n2. Hybridization");
    LL_hyb = dot_spinful_arrays(LL_surf, np.array([LL_coup[0].T]));
    LL_hyb = dot_spinful_arrays(LL_hyb, LL_coup, backwards = True);
    RL_hyb = dot_spinful_arrays(RL_surf, np.array([RL_coup[0].T]));
    RL_hyb = dot_spinful_arrays(RL_hyb, RL_coup, backwards = True);
    
    # bath disc: outputs n_bath_orbs bath energies, for each imp orb
    if(verbose): print("\n3. Bath discretization");
    hyb = LL_hyb + RL_hyb;
    bathe, bathv = dmft.gwdmft.get_bath_direct(hyb, energies, n_bath_orbs);

    # optimize
    bathe, bathv = dmft.gwdmft.opt_bath(bathe, bathv, hyb, energies, iE, n_bath_orbs);
    if(verbose): print(" - opt. bath energies = ", bathe);

    # construct manybody hamiltonian of imp + bath
    h1e_imp, h2e_imp = dmft.gwdmft.imp_ham(SR_1e, SR_2e, bathe, bathv, n_core); # adds in bath states

    # get chem pot that corresponds to desired occupancy
    #chem_pot = find_mu(h1e_imp, h2e_imp, 0.0, np.array([np.eye(n_orbs)]), imp_occ, max_mem, verbose = 0);
    chem_pot = 0.0; # corresponds to bath half-filling
    
    # find manybody gf of imp + bath
    # ie Zgid paper eq 28
    if(verbose): print("\n4. Impurity Green's function");
    meanfield = dmft.dmft_solver.mf_kernel(h1e_imp, h2e_imp, chem_pot, nao, np.array([np.eye(n_orbs)]), max_mem, verbose = verbose);

    # use fci (which is spin restricted) to get Green's function
    # choose solver
    assert(len(np.shape(meanfield.mo_coeff)) == 2); # ie spin restricted
    if(solver == 'cc'):

        # get MBGF, reduced density matrix
        G, rdm = dmft.dmft_solver.cc_gf(meanfield, energies, iE);
        rdm = rdm[:n_imp_orbs, :n_imp_orbs];
        
    elif(solver == 'fci'):

        # get MBGF
        assert(n_orbs <= 10); # so it doesn't stall
        G, soln = dmft.dmft_solver.fci_gf(meanfield, energies, iE, verbose = verbose);

        # get rdm for scattering region only
        rdm = dmft.dmft_solver.fci_sol_to_rdm(meanfield, soln, n_imp_orbs);

    else: raise(ValueError(solver+" is not a valid solver type"));
               
    return G; # package in up spin for shape consistency


def wingreen(energies, iE, kBT, MBGF, LL, RL, verbose = 0):
    '''
    Given the MBGF for the impurity + bath system, apply meir wingreen formula
    to get "density of current" j(E) at temp kBT. Then total particle current
    is given by J = \int dE j(E)
    '''
    
    # check inputs
    assert( len(energies) == np.shape(MBGF)[-1]);
    assert( np.shape(LL[0]) == np.shape(RL[0]) );
    
    # unpack
    LL_diag, LL_hop, LL_coup, mu_L = LL;
    RL_diag, RL_hop, RL_coup, mu_R = RL;
    n_imp_orbs = np.shape(LL_coup)[1];
    G_ret, G_adv, G_les, G_gre = decompose_gf(energies, MBGF[:,:n_imp_orbs, :n_imp_orbs], kBT);
    # 1: hybridization between leads and SR
    
    # surface gf (matrices of vectors of E)
    LL_surf = surface_gf(energies, iE, LL_diag, LL_hop, verbose = verbose);
    RL_surf = surface_gf(energies, iE, RL_diag, RL_hop, verbose = verbose);
    
    # hyb(E) = V*surface_gf(E)*V^\dagger
    LL_hyb = dot_spinful_arrays(LL_surf, np.array([LL_coup[0].T]));
    LL_hyb = dot_spinful_arrays(LL_hyb, LL_coup, backwards = True);
    RL_hyb = dot_spinful_arrays(RL_surf, np.array([RL_coup[0].T]));
    RL_hyb = dot_spinful_arrays(RL_hyb, RL_coup, backwards = True);

    # meir-wingreen Lambda matrix = -2*Im[hyb]
    Lambda_L = (-2)*np.imag(LL_hyb);
    Lambda_R = (-2)*np.imag(RL_hyb);

    # 2: thermal distributions (vectors of E)
    if(kBT == 0.0):
        nL = np.zeros_like(energies, dtype = int);
        nL[energies <= mu_L] = 1; # step function
        nR = np.zeros_like(energies, dtype = int);
        nR[energies <= mu_R] = 1; # step function
    else:
        nL = 1/(np.exp((energies - mu_L)/kBT) + 1);
        nR = 1/(np.exp((energies - mu_R)/kBT) + 1);

    # 4: meir wingreen formula
    therm = dot_spinful_arrays(Lambda_L, nL) - dot_spinful_arrays(Lambda_R, nR); # combines thermal contributions
    jEmat = dot_spinful_arrays(therm, G_ret - G_adv); # first term of MW Eq 6, before trace
    jEmat += dot_spinful_arrays((Lambda_L - Lambda_R), G_les);
    jE = (complex(0,1)/2)*np.trace(jEmat[0]); # trace over impurity sites

    # test code
    assert( np.max(abs(np.trace((complex(0,1)/2)*(dot_spinful_arrays((Lambda_L - Lambda_R), G_les))[0]))) < 1e-10 );
    if False:
        import matplotlib.pyplot as plt
        x = -np.imag(dot_spinful_arrays(therm, G_ret - G_adv))
        for i in range(np.shape(x)[1]):
            for j in range(np.shape(x)[2]):
                if (i==j):
                    plt.plot(energies, x[0,i,j,:], label = (i,j));
        plt.legend();
        plt.title("therm");
        plt.show();

    return jE;


def landauer(energies, iE, kBT, MBGF, LL, RL, verbose = 0):
    '''
    '''

    # check inputs
    assert( len(energies) == np.shape(MBGF)[-1]);
    assert( np.shape(LL[0]) == np.shape(RL[0]) );
    
    # unpack
    LL_diag, LL_hop, LL_coup, mu_L = LL;
    RL_diag, RL_hop, RL_coup, mu_R = RL;
    n_imp_orbs = np.shape(LL_coup)[1];
    G_ret, G_adv, G_les, G_gre = decompose_gf(energies, MBGF[:,:n_imp_orbs, :n_imp_orbs], kBT);

    # 1: hybridization between leads and SR
    
    # surface gf (matrices of vectors of E)
    LL_surf = surface_gf(energies, iE, LL_diag, LL_hop, verbose = verbose);
    RL_surf = surface_gf(energies, iE, RL_diag, RL_hop, verbose = verbose);

    # hyb(E) = V*surface_gf(E)*V^\dagger
    LL_hyb = dot_spinful_arrays(LL_surf, np.array([LL_coup[0].T]));
    LL_hyb = dot_spinful_arrays(LL_hyb, LL_coup, backwards = True);
    RL_hyb = dot_spinful_arrays(RL_surf, np.array([RL_coup[0].T]));
    RL_hyb = dot_spinful_arrays(RL_hyb, RL_coup, backwards = True);
    
    # meir-wingreen Lambda matrix = -2*Im[hyb]
    Lambda_L = (-2)*np.imag(LL_hyb);
    Lambda_R = (-2)*np.imag(RL_hyb);

    # 2: thermal distributions
    if(kBT == 0.0):
        nL = np.zeros_like(energies, dtype = int);
        nL[energies <= mu_L] = 1; # step function
        nR = np.zeros_like(energies, dtype = int);
        nR[energies <= mu_R] = 1; # step function
    else:
        nL = 1/(np.exp((energies - mu_L)/kBT) + 1);
        nR = 1/(np.exp((energies - mu_R)/kBT) + 1);

    # landauer formula
    jEmat = dot_spinful_arrays( dot_spinful_arrays(G_adv, Lambda_R), dot_spinful_arrays(G_ret, Lambda_L) );
    jE = np.trace( jEmat[0])*(nL-nR);
    return jE;


########################################################################
#### green's function finders

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
    energies = energies + complex(0,iE);

    # quick shortcut for diag inputs
    H_is_diag = (np.trace(H[0]) == np.sum(H.flat) ); # is a diagonal matrix
    H_is_same = not np.any(np.diagonal(H[0]) - H[0,0,0]); # all diags equal
    V_is_diag = (np.trace(V[0]) == np.sum(V.flat) );
    V_is_same = not np.any(np.diagonal(V[0]) - V[0,0,0]);

    # if these are all true, can use closed form eq for diag gf
    if(H_is_diag and H_is_same and V_is_diag and V_is_same):

        if(verbose): print(" - Diag shortcut");
        energies = np.real(energies); # no + iE necessary in this case
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


########################################################################
#### utils

def decompose_gf(energies, G, kBT):
    '''
    Decompose the full time-ordered many body green's function (from kernel)
    into r, a, <, > parts according to page 18 of
    http://www.physics.udel.edu/~bnikolic/QTTG/NOTES/MANY_PARTICLE_PHYSICS/BROUWER=theory_of_many_particle_systems.pdf

    NB chem potential fixed at 0 as convention
    '''

    # check inputs
    assert( len(energies) == np.shape(G)[-1]);

    # return values
    G_ret = np.empty(np.shape(G), dtype = complex);
    G_adv = np.empty(np.shape(G), dtype = complex);
    G_les = np.empty(np.shape(G), dtype = complex);
    G_gre = np.empty(np.shape(G), dtype = complex);

    # vectorize in energy by hand
    for wi in range(len(energies)):

        # temperature dependence comes as exponential factor
        if(kBT == 0.0):
            expT = 0.0;
            expTinv = 1e9;
        else:
            expT = np.exp(-energies[wi]/kBT);
            expTinv = np.exp(energies[wi]/kBT);

        # screen out nans
        #if(1-expT == 0): expT += 1e-9;
        #if(1+expTinv == 0): expTinv += 1e-9;

        # retarded gf
        G_ret[:,:,:,wi] = np.real(G[:,:,:,wi]) + complex(0,1)*((1+expT)/(1-expT))*np.imag(G[:,:,:,wi]);

        # advanced gf (just the conj of retarded)
        G_adv[:,:,:,wi] = np.real(G[:,:,:,wi]) - complex(0,1)*((1+expT)/(1-expT))*np.imag(G[:,:,:,wi]);

        # spectral function from G_ret
        spectral = (-2)*np.imag(G_ret[:,:,:,wi])

        # lesser gf
        G_les[:,:,:,wi] = complex(0,1)*spectral/(1+expTinv);

        # greater gf
        G_gre[:,:,:,wi] = -complex(0,1)*spectral/(1+expT);

    assert( not np.any(np.isnan(G_ret)) );
    assert( not np.any(np.isnan(G_adv)) );
    assert( not np.any(np.isnan(G_les)) );
    assert( not np.any(np.isnan(G_adv)) );
    return G_ret, G_adv, G_les, G_gre;


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

    elif(np.shape(a2) == (nfreqs,) ): # freq dependent scalar
        for s in range(spin):
            for i in range(norbs):
                for j in range(norbs):
                    result[s,i,j] = a1[s,i,j]*a2;

    elif(np.shape(a2) == np.shape(a1) ): # both are freq dependent ops
        for s in range(spin):
            for iw in range(nfreqs):
                result[s,:,:,iw] = np.matmul(a1[s,:,:,iw], a2[s,:,:,iw]);

    elif(False):
        pass;

    else: raise(ValueError("a2 "+str(np.shape(a2))+" is of wrong size") );

    return result;






























########################################################################
#### garbage

def find_mu(h1e, g2e, mu0, dm0, target, max_mem, max_cycle = 5, trust_region = 1.0, step = 0.2, nelec_tol = 2e-3, verbose = 0):
    '''
    Find chemical potential that reproduces the target occupancy on the impurity
    '''

    # check inputs
    assert(np.shape(h1e) == np.shape(dm0));

    # before starting loop
    mu_cycle = 0
    dmu = 0 # change in mu
    record = [] # records stuff as we cycle
    nao = 2; # only affects printouts
    nimp = 2;

    # loop
    while mu_cycle < max_cycle:
        
        # run HF for embedding problem
        mu = mu0 + dmu
        mf = dmft.dmft_solver.mf_kernel(h1e, g2e, mu, nao, dm0, max_mem)

        # run ground-state impurity solver to get 1-rdm
        rdm = dmft.dmft_solver.fci_rdm(mf, ao_orbs = range(nimp), verbose = verbose)
        nelec = np.trace(rdm)
        if mu_cycle > 0:
            dnelec_old = dnelec
        dnelec = nelec - target
        print("mu cycle ", mu_cycle, "mu = ", mu,"dmu = ", dmu,"nelec = ", nelec, "dnelec = ", dnelec);
        if abs(dnelec) < nelec_tol * target:
            break
        if mu_cycle > 0:
            if abs(dnelec - dnelec_old) < 1e-8:
                print(" line 294");
                #break
        record.append([dmu, dnelec])

        if mu_cycle == 0:
            if dnelec > 0:
                dmu = -1. * step
            else:
                dmu = step
        elif len(record) == 2:
            # linear fit
            dmu1 = record[0][0]; dnelec1 = record[0][1]
            dmu2 = record[1][0]; dnelec2 = record[1][1]
            dmu = (dmu1*dnelec2 - dmu2*dnelec1) / (dnelec2 - dnelec1)
        else:
            # linear fit
            dmu_fit = []
            dnelec_fit = []
            for rec in record:
                dmu_fit.append(rec[0])
                dnelec_fit.append(rec[1])
            dmu_fit = np.array(dmu_fit)
            dnelec_fit = np.array(dnelec_fit)
            idx = np.argsort(np.abs(dnelec_fit))[:2]
            dmu_fit = dmu_fit[idx]
            dnelec_fit = dnelec_fit[idx]
            a,b = np.polyfit(dmu_fit, dnelec_fit, deg=1)
            dmu = -b/a

        if abs(dmu) > trust_region:
            if dmu < 0:
                dmu = -trust_region
            else:
                dmu = trust_region

        mu_cycle += 1
        
    return mu


    


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
    
    

