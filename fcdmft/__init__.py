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

import numpy as np
import matplotlib.pyplot as plt

########################################################################
#### drivers

def kernel(energies, iE, SR_1e, SR_2e, chem_pot, dm_SR, LL, RL, n_bath_orbs, solver = "mf", verbose = 0): # main driver
    '''
    Driver of MBGF calculation
    - scattering region, treated at high level, repped by SR_1e and SR_2e
    - noninteracting leads, treated at low level, repped by LL, RL

    Difference between my code, Tianyu's code (e.g. fcdmft/dmft/gwdmft.kernel())
    is that the latter assumes periodicity, and so only takes local hamiltonian
    of impurity, and does a self-consistent loop. In contrast, this code
    specifies the environment and skips scf

    Args:
    energies, 1d arr of energies
    iE, float, complex part to add to energies
    imp_occ, int, target impurity occupancy
    SR_1e, (spin,n_imp_orbs, n_imp_orbs) array, 1e part of scattering region hamiltonian
    SR_2e, 2e part of scattering region hamiltonian
    chem_pot, chemical potential in the SR, controls occupancy
    dm_SR, (spin,n_imp_orbs, n_imp_orbs) array, initial density matrix of the SR
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
    assert( iE >= 0);
    assert(np.shape(SR_1e) == np.shape(dm_SR));
    assert(np.shape(SR_1e) == np.shape(LL[0]));
    assert(np.shape(SR_1e) == np.shape(LL[1]));
    assert(np.shape(SR_1e) == np.shape(LL[2]));
    assert(len(LL) == len(RL) );
    
    # unpack
    spin, n_imp_orbs, _ = np.shape(SR_1e); # size of arrs on spin, norb axes
    LL_diag, LL_hop, LL_coup, mu_L = LL;
    RL_diag, RL_hop, RL_coup, mu_R = RL;
    n_core = 0; # core orbitals
    max_mem = 8000;
    n_orbs = n_imp_orbs + 2*n_bath_orbs;

    # surface green's function in the leads
    if(verbose): print("\n1. Surface Green's function");
    LL_surf = surface_gf(energies, iE, LL_diag, LL_hop, verbose = verbose);
    RL_surf = surface_gf(energies, iE, RL_diag, RL_hop, verbose = verbose);

    # hybridization between leads and SR
    # hyb = V*surface_gf*V^\dagger
    if(verbose): print("\n2. Hybridization");
    LL_hyb = dot_spinful_arrays(LL_surf, LL_coup);
    LL_hyb = dot_spinful_arrays(LL_hyb, LL_coup, backwards = True);
    RL_hyb = dot_spinful_arrays(RL_surf, RL_coup);
    RL_hyb = dot_spinful_arrays(RL_hyb, RL_coup, backwards = True);
    
    # bath disc: outputs n_bath_orbs bath energies, for each imp orb
    if(n_bath_orbs > 0):
        
        if(verbose): print("\n3. Bath discretization");
        hyb = LL_hyb + RL_hyb;
        bathe, bathv = dmft.gwdmft.get_bath_direct(hyb, energies, n_bath_orbs);

        # optimize
        bathe, bathv = dmft.gwdmft.opt_bath(bathe, bathv, hyb, energies, iE, n_bath_orbs);
        if(verbose): print(" - opt. bath energies = ", bathe);

        # construct manybody hamiltonian of imp + bath
        h1e_imp, h2e_imp = dmft.gwdmft.imp_ham(SR_1e, SR_2e, bathe, bathv, n_core); # adds in bath states

        # include bath orbs in density matrix
        dm_guess = np.zeros_like(h1e_imp);
        for s in range(np.shape(dm_guess)[0]):
            for orbi in range(np.shape(dm_guess)[1]):
                if( orbi < np.shape(dm_SR)[1]): # copy from SR dm
                    dm_guess[s,orbi,orbi] = dm_SR[s,orbi, orbi];
                else:
                    dm_guess[s,orbi, orbi] = 1; # half filled bath orbs

    # don't do bath if nbo = 0
    else:
        h1e_imp, h2e_imp, dm_guess = SR_1e, SR_2e, dm_SR;

    # find manybody gf of imp + bath
    # ie Zgid paper eq 28
    if(verbose): print("\n4. Impurity Green's function with "+solver);
    meanfield = dmft.dmft_solver.mf_kernel(h1e_imp, h2e_imp, chem_pot, n_imp_orbs, dm_guess, max_mem, verbose = verbose);

    # choose solver to get Green's function
    if(solver == 'mf'):

        # get MBGF in hartree fock approx
        MBGF = dmft.dmft_solver.mf_gf(meanfield, energies, iE, verbose = verbose);
    
    elif(solver == 'cc'):

        # get MBGF
        if( spin == 1):
            MBGF = dmft.dmft_solver.cc_gf(meanfield, energies, iE, cas = False, nimp = n_imp_orbs, verbose = verbose);
        if( spin == 2):
            MBGF = dmft.dmft_solver.ucc_gf(meanfield, energies, iE, cas = False, nimp = n_imp_orbs, verbose = verbose);

    elif(solver == 'fci'):

        # get MBGF
        pass;
    
    elif(solver == 'dmrg'):

        from transport import tddmrg
        pass;

    else: raise(ValueError(solver+" is not a valid solver type"));
               
    return MBGF; 


def wingreen(energies, iE, kBT, MBGF, LL, RL, verbose = 0):
    '''
    Given the MBGF for the impurity + bath system, use Bruus Eq 10.57 to get
    linear response current j(E) at temp kBT.

    Skip trace bc doing all spin up formalism, so channels carry spin info (ie
    jE[0,0] = up spin current)
    
    Assumptions and connections to MW:
    - Interacting, so start with MW Eq 6
    - at equilibrium, LambdaL = LambdaR -> can use MW Eq 9 (w/out trace)
    - since there are no spin interactions in leads, Lambda's are always
        diagonal thus trace -> sum over sigma -> MW Eq 12
    '''
    
    # check inputs
    assert( len(energies) == np.shape(MBGF)[-1]);
    assert( np.shape(LL[0]) == np.shape(RL[0]) );
    
    # unpack
    LL_diag, LL_hop, LL_coup, mu_L = LL;
    RL_diag, RL_hop, RL_coup, mu_R = RL;
    n_imp_orbs = np.shape(LL_coup)[1];

    # thermal distributions (vectors of E)
    if(kBT == 0.0):
        nL = np.zeros_like(energies, dtype = int);
        nL[energies <= mu_L] = 1; # step function
        nR = np.zeros_like(energies, dtype = int);
        nR[energies <= mu_R] = 1; # step function
    else:
        nL = 1/(np.exp((energies - mu_L)/kBT) + 1);
        nR = 1/(np.exp((energies - mu_R)/kBT) + 1);
    
    # 1: hybridization between leads and SR
    
    # surface gf (matrices of vectors of E)
    LL_surf = surface_gf(energies, iE, LL_diag, LL_hop, verbose = verbose);
    RL_surf = surface_gf(energies, iE, RL_diag, RL_hop, verbose = verbose);
    
    # hyb(E) matrix = V*surface_gf(E)*V^\dagger
    LL_hyb = dot_spinful_arrays(LL_surf, LL_coup);
    LL_hyb = dot_spinful_arrays(LL_hyb, LL_coup, backwards = True);
    RL_hyb = dot_spinful_arrays(RL_surf, RL_coup);
    RL_hyb = dot_spinful_arrays(RL_hyb, RL_coup, backwards = True);

    # meir-wingreen Lambda(E) matrix = -2*Im[hyb]
    Lambda_L = (-2)*np.imag(LL_hyb);
    Lambda_R = (-2)*np.imag(RL_hyb);
    Lambda = dot_spinful_arrays(dot_spinful_arrays(Lambda_L, Lambda_R), invert(Lambda_L + Lambda_R));

    # 2: current density in terms of spectral density = -1/pi Im(MBGF)
    # ie Bruus, equation 10.57
    MBGF_trunc = MBGF[:,:n_imp_orbs,:n_imp_orbs,:];
    spectral = (-1/np.pi)*(MBGF_trunc - dagger(MBGF_trunc))/complex(0,2);
    jE = (nL - nR)*(dot_spinful_arrays(Lambda, spectral)); #
        
    if(verbose > 4):
        # compare with Meir Wingreen formula - Meir Wingreen Eq 6 (deprecated)
        G_ret, G_adv, G_les, G_gre = decompose_gf(energies, MBGF_trunc, (nL+nR)/2);
        therm = dot_spinful_arrays(Lambda_L, nL) - dot_spinful_arrays(Lambda_R, nR); # combines thermal contributions
        jEmat = dot_spinful_arrays(therm, G_ret - G_adv); # first term of MW Eq 6, before trace
        jEmat += dot_spinful_arrays((Lambda_L - Lambda_R), G_les);
        jEnew = (complex(0,1)/(4*np.pi))*np.trace(jEmat[0]); # hbar = 1, trace over impurity sites
        plt.plot(energies, np.real(jEnew), label = "MW Eq 6");
        plt.plot(energies, np.trace(jE[0]), linestyle = "dashed", label = "Bruus Eq 10.57");
        plt.legend();
        plt.show();

    return jE; # remove spin wrapper at end


def landauer(energies, iE, kBT, MBGF, LL, RL, verbose = 0):
    '''
    Given the MBGF for the impurity + bath system, calculate the current
    through the impurity, assuming the noninteracting case (Meir Wingreen Eq 7)
    '''

    # check inputs
    assert( len(energies) == np.shape(MBGF)[-1]);
    assert( np.shape(LL[0]) == np.shape(RL[0]) );
    
    # unpack
    LL_diag, LL_hop, LL_coup, mu_L = LL;
    RL_diag, RL_hop, RL_coup, mu_R = RL;
    n_imp_orbs = np.shape(LL_coup)[1];

    # thermal distributions (vectors of E)
    if(kBT == 0.0):
        nL = np.zeros_like(energies, dtype = int);
        nL[energies <= mu_L] = 1; # step function
        nR = np.zeros_like(energies, dtype = int);
        nR[energies <= mu_R] = 1; # step function
    else:
        nL = 1/(np.exp((energies - mu_L)/kBT) + 1);
        nR = 1/(np.exp((energies - mu_R)/kBT) + 1);

    # particular gf's
    G_ret, G_adv, G_les, G_gre = decompose_gf(energies, MBGF[:,:n_imp_orbs, :n_imp_orbs], (nL+nR)/2);

    # 1: hybridization between leads and SR
    
    # surface gf (matrices of vectors of E)
    LL_surf = surface_gf(energies, iE, LL_diag, LL_hop, verbose = verbose);
    RL_surf = surface_gf(energies, iE, RL_diag, RL_hop, verbose = verbose);

    # hyb(E) = V*surface_gf(E)*V^\dagger
    LL_hyb = dot_spinful_arrays(LL_surf, LL_coup);
    LL_hyb = dot_spinful_arrays(LL_hyb, LL_coup, backwards = True);
    RL_hyb = dot_spinful_arrays(RL_surf, RL_coup);
    RL_hyb = dot_spinful_arrays(RL_hyb, RL_coup, backwards = True);
    
    # meir-wingreen Lambda matrix = -2*Im[hyb]
    Lambda_L = (-2)*np.imag(LL_hyb);
    Lambda_R = (-2)*np.imag(RL_hyb);

    # landauer formula - Meir Wingreen Eq. 7
    jEmat = dot_spinful_arrays( dot_spinful_arrays(G_adv, Lambda_R), dot_spinful_arrays(G_ret, Lambda_L) );
    jE = np.trace( jEmat[0])*(nL-nR)/(2*np.pi); # hbar = 1

    # in terms of spectral = -1/pi Im(MBGF) only
    MBGF_trunc = MBGF[:,:2,:2,:];
    spectral = (-1/np.pi)*(MBGF_trunc - dagger(MBGF_trunc))/complex(0,2);
    #assert(np.max(abs(np.imag(spectral))) < 1e-10)
    jEnew = (nL - nR)*np.trace(dot_spinful_arrays(Lambda_L/2, spectral)[0])

    # test code
    if (verbose > 4):
        plt.plot(energies, np.imag(G_les[0,0,0]), label = "G<");
        plt.plot(energies, np.imag(G_les[0,1,1]), label = "G<");
        idenL = dot_spinful_arrays(G_ret, dot_spinful_arrays(Lambda_L, G_adv));
        idenR = dot_spinful_arrays(G_ret, dot_spinful_arrays(Lambda_R, G_adv));
        identity = complex(0,1)*(dot_spinful_arrays(idenL, nL) + dot_spinful_arrays(idenR, nR));
        plt.plot(energies, np.imag(identity[0,0,0]), linestyle = "dashed", label = "iden");
        plt.plot(energies, np.imag(identity[0,1,1]), linestyle = "dashed", label = "iden");
        plt.plot(energies, np.real(identity[0,0,0]), linestyle = "dashed", label = "iden real");
        plt.plot(energies, np.real(identity[0,1,1]), linestyle = "dashed", label = "iden real");
        plt.legend();
        plt.show();
        
    return jE;


########################################################################
#### green's function finders

def surface_gf(energies, iE, H, V, tol = 1e-3, max_cycle = 1000, verbose = 0):
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
    shortcut = True;
    for s in range(np.shape(H)[0]): # spin dof
        if(not np.trace(H[s]) == np.sum(H[s].flat)): # is a diagonal matrix
            shortcut = False;
        if(np.any(np.diagonal(H[s]) - H[s,0,0])): # all diags equal
            shortcut = False;
        if(not np.trace(V[s]) == np.sum(V[s].flat) ):
           shortcut = False;
        if(np.any(np.diagonal(V[0]) - V[0,0,0])):
            shortcut = False;

    # if these are all true, can use closed form eq for diag gf
    if(shortcut):

        if(verbose): print(" - Diag shortcut");
        energies = np.real(energies); # no + iE necessary in this case
        gf = np.zeros((*np.shape(H),len(energies) ), dtype = complex);

        for s in range(np.shape(H)[0]): # iter over spin
            for orbi in range(np.shape(H)[1]): # iter over diag
                pref = (energies - H[s,orbi,orbi])/(2*V[s,orbi,orbi]);
                gf[s,orbi,orbi,:] = pref-np.lib.scimath.sqrt(pref*pref*(1-4*V[s,orbi,orbi]*V[s,orbi,orbi]/np.power(energies-H[s,orbi,orbi],2)));

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
            sigma = dot_spinful_arrays(gf0, V ); #gf0 * V^\dagger
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


def spdm(energies, iE, G):
    '''
    Get the single particle density matrix from the many body green's function
    '''
    import matplotlib.pyplot as plt
    # check inputs
    assert(len(energies) == np.shape(G)[-1]);

    # return val
    P = np.zeros(np.shape(G)[:3], dtype = complex);

    # fill
    for s in range(np.shape(P)[0]):
        for i in range(np.shape(P)[1]):
            for j in range(np.shape(P)[2]):
                fE = np.exp(complex(0,1)*energies*iE)*G[s,i,j]; # func to integrate
                P[s,i,j] = complex(0,-1)*np.trapz(fE, energies);
                if(i==j and False):
                    plt.plot(energies, np.real(fE));
                    plt.plot(energies, np.imag(fE));
                    plt.title(i);
                    plt.show();
                    
    return P;
    


########################################################################
#### utils



def dagger(g):
    '''
    Get hermitian conjugate of a spin by norb by norb (energy) object
    '''

    # check inputs
    assert(len(np.shape(g)) == 4);

    # hermitian conjugate
    gdagger = np.zeros_like(g);
    for s in range(np.shape(g)[0]):
        for wi in range(np.shape(g)[-1]):
            gdagger[s,:,:,wi] = np.conj(g[s,:,:,wi].T);

    return gdagger;


def invert(g):
    '''
    Get inverse of a spin by norb by norb (energy) object
    '''

    # check inputs
    assert(len(np.shape(g)) == 4);

    # hermitian conjugate
    ginv = np.zeros_like(g);
    for s in range(np.shape(g)[0]):
        for wi in range(np.shape(g)[-1]):
            ginv[s,:,:,wi] = np.linalg.inv(g[s,:,:,wi]);

    return ginv;


def dot_spinful_arrays(a1_, a2_, backwards = False):
    '''
    given an array of shape (spin, norbs, norbs, nfreqs)
    and another array , either
    - an operator, shape (spin, norbs, norbs), indep of freq
    '''

    # unpack
    a1, a2 = np.copy(a1_), np.copy(a2_); # not in place
    spin, norbs, _, nfreqs = np.shape(a1);

    # return var
    result = np.zeros_like(a1, dtype = complex);

    # screen by kind of second array
    if( np.shape(a2) == (spin, norbs, norbs) ): # freq indep operator
        for s in range(spin):
            for iw in range(nfreqs):
                if not backwards:
                    result[s,:,:,iw] = np.matmul(a1[s,:,:,iw], a2[s]);
                else:
                    result[s,:,:,iw] = np.matmul(a2[s], a1[s,:,:,iw]);

    elif(np.shape(a2) == (nfreqs,) ): # freq dependent scalar
        assert( not backwards);
        for s in range(spin):
            for i in range(norbs):
                for j in range(norbs):
                    result[s,i,j] = a1[s,i,j]*a2;

    elif(np.shape(a2) == np.shape(a1) ): # both are freq dependent ops
        assert(not backwards);
        for s in range(spin):
            for iw in range(nfreqs):
                assert( len(np.shape(a1[s,:,:,iw])) == 2 and np.shape(a1)[1] == np.shape(a1)[2]);
                result[s,:,:,iw] = np.matmul(a1[s,:,:,iw], a2[s,:,:,iw]);

    elif(isinstance(a2, float)):
        return a2*a1;

    elif(False):
        pass;

    else: raise(ValueError("a2 "+str(np.shape(a2))+" is of wrong size") );

    assert(np.shape(result) == np.shape(a1));
    return result;
                             

def decompose_gf(energies, G, nFD):
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
    
    # "real", "imag" part of time orderd MBGF
    Gdagger = dagger(G);
    realG = (1/2)*(G + Gdagger);
    imagG = (1/complex(0,2))*(G-Gdagger);

    # vectorize in energy by hand
    for s in range(np.shape(G)[0]):
        for wi in range(len(energies)):

            # retarded gf
            G_ret[s,:,:,wi] = realG[s,:,:,wi] + complex(0,1)*imagG[s,:,:,wi];

            # advanced gf (just the conj of retarded)
            G_adv[s,:,:,wi] = realG[s,:,:,wi] - complex(0,1)*imagG[s,:,:,wi];

            # spectral function =  -Im(Gr - Ga) = -2Im(Gr)
            spectral = (-2)*(1/complex(0,2))*(G_ret[s,:,:,wi] - np.conj(G_ret[s,:,:,wi].T));

            # lesser gf
            G_les[s,:,:,wi] = complex(0,1)*spectral*nFD[wi];

            # greater gf
            G_gre[s,:,:,wi] = -complex(0,1)*spectral*(1-nFD[wi]);

    # check outputs
    assert( np.max(abs(G_ret - G_adv - (G_gre - G_les) )) < 1e-15);
    if not (np.max(abs(np.real(G_les))) < 1e-10):
        print(np.max(abs(np.real(G_les))));
        #assert False;
    assert( not np.any(np.isnan(np.array([G_ret, G_adv, G_les, G_gre]) ) ) );
    return G_ret, G_adv, G_les, G_gre;


def find_mu(h1e, g2e, mu0, nimp, target, max_mem, max_cycle = 20, trust_region = 0.1, step = 0.01, nelec_tol = 1e-2, verbose = 0):
    '''
    Find chemical potential that reproduces the target occupancy on the impurity

    dm0, the initial density matrix, is hardcoded in to guess:
    - that the first "target" imp orbs are filled to 1
    - that the rest of the imp orbs are unfilled
    - that all the bath orbs are half filled

    Very Important! if the cycle converges due to dnelec converging, but not to
    zero, then just returns mu=mu0
    '''

    # check inputs
    assert(np.shape(h1e) == np.shape(g2e)[:3]);
    assert( isinstance( target, int) );

    # before starting loop
    norbs = np.shape(h1e)[1];
    mu_cycle = 0
    dmu = 0 # change in mu
    record = [] # records stuff as we cycle

    # initial guess dm
    dm0 = np.zeros((norbs, norbs));
    for i in range(norbs):
        if(i < target): # filled imp state
            dm0[i,i] = 1;
        elif( i < nimp): # unfilled imp state
            dm0[i,i] = 0;
        else: # half filled bath state
            dm0[i,i] = 0.5;
    dm0 = np.array([dm0]); # spin wrapper

    # loop
    while mu_cycle < max_cycle:
        
        # run HF for embedding problem
        mu = mu0 + dmu
        mf = dmft.dmft_solver.mf_kernel(h1e, g2e, mu, nimp, dm0, max_mem, verbose = verbose)

        # run ground-state impurity solver to get rdm, nelec
        dm = mf.make_rdm1()
        nelec = np.trace(dm[:nimp,:nimp])
        if mu_cycle > 0:
            dnelec_old = dnelec
        dnelec = nelec - target

        # diagnostic
        if(verbose): print("mu cycle ", mu_cycle, "mu = ", mu,"dmu = ", dmu,"nelec = ", nelec, "dnelec = ", dnelec, "record = ",record);

        # check convergence
        if abs(dnelec) < nelec_tol * target:
            if(verbose): print(" - nelec converged");
            break
        if mu_cycle > 0:
            if abs(dnelec - dnelec_old) < nelec_tol/10:
                if(verbose): print(" - dnelec converged");
                if(abs(dnelec) < nelec_tol * target):
                    break;
                else:
                    if(verbose): print(" - nelec not converged, returning mu0");
                    mu = mu0;
                    break;
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

    # return result of density matrix assoc'd with target occ
    return mu, np.array([dm]);
    

