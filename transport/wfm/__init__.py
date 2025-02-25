'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
general formalism: all sites map to all the different
degrees of freedom of the system
'''

from transport import tdfci
from transport.tdfci import utils as fci_mod

import numpy as np

##################################################################################
#### driver of transmission coefficient calculations

def kernel(h, tnn, tnnn, tl, E, Ajsigma, is_psi_jsigma, is_Rhat, all_debug = True, verbose = 0, ):
    '''
    coefficient for a transmitted up and down electron
    Args
    -h, array, block hamiltonian matrices
    -tnn, array, nearest neighbor block hopping matrices
    -tnnn, array, next nearest neighbor block hopping matrices
    -tl, float, hopping in leads, not necessarily same as hopping on/off SR
        or within SR which is defined by tnn, tnnn matrices
    -E, float, energy of the incident electron
    -Ajsigma, incident particle amplitude at site 0 in spin channel j
    -is_psi_jsigma, whether to return computed wavefunction
    -is_Rhat, whether to return Rhat operator or just R, T probabilities
    Optional args
    -verbose, how much printing to do
    -all_debug, whether to enforce a bunch of extra assert statements

    Returns
    tuple of R coefs (vector of floats for each sigma) and T coefs (likewise)
    UNLESS is_psi_jsigma = True, in which case
    returns the computed wavefunction
    UNLESS is_Rhat = True, in which case
    returns n_loc_dof \times n_loc_dof matrix Rhat, which
    transforms incoming spin states to reflected spin states
    '''
    if(not isinstance(h, np.ndarray)): raise TypeError;
    if(not isinstance(tnn, np.ndarray)): raise TypeError;
    if(not isinstance(tnnn, np.ndarray)): raise TypeError;
    
    # check that lead hams are diagonal
    for hi in [0, -1]: # LL, RL
        isdiag = h[hi] - np.diagflat(np.diagonal(h[hi])); # subtract off diag
        if(all_debug and np.any(isdiag)): # True if there are nonzero off diag terms
            raise Exception("Not diagonal\n"+str(h[hi]))
    for sigma in range(len(Ajsigma)): # always set incident mu = 0
        if(Ajsigma[sigma] != 0):
            pass;
            assert(h[0,sigma,sigma] == 0);

    # check incident amplitude
    assert( isinstance(Ajsigma, np.ndarray));
    assert( len(Ajsigma) == np.shape(h[0])[0] );

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # determine velocities in the left, right leads
    ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tl)); # vector with sigma components
    ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tl));
    v_L = 2*tl*np.sin(ka_L); # vector with sigma components
    v_R = 2*tl*np.sin(ka_R); # a, hbar defined as 1   <---- !!! change !!!

    # green's function
    if(verbose): print("\nEnergy = {:.6f}".format(np.real(E+2*tl))); # start printouts
    Gmat = Green(h, tnn, tnnn, tl, E, verbose = verbose); # spatial and spin indices separate
    
    # from Green's function, determine wavefunction elements \psi_j\sigma
    psi_jsigma = complex(0,1)*np.dot(Gmat[:,0], Ajsigma*v_L);
    if(is_psi_jsigma): return psi_jsigma;
    
    # from Green's func, determine matrix elements < \sigma | rhat | \sigma'> of the
    # reflection operator Rhat, which scatters \sigma' -> \sigma
    Rhat_matrix = 2*tl*complex(0,1)*Gmat[0,0]*np.sin(ka_L) - np.eye(n_loc_dof);
    if(is_Rhat): return Rhat_matrix;

    # determine matrix elements
    i_flux = np.sqrt(np.dot(Ajsigma, Ajsigma*np.real(v_L))); # sqrt of i flux

    # from matrix elements, determine R and T coefficients
    # (eq:Rcoef and eq:Tcoef in paper)
    Rs = np.zeros(n_loc_dof, dtype = float); # force as float bc we check that imag part is tiny
    Ts = np.zeros(n_loc_dof, dtype = float);
    for sigma in range(n_loc_dof): # iter over spin dofs
        # sqrt of r flux, numerator of eq:Rcoef in manuscript
        r_flux = (complex(0,1)*np.dot(Gmat[0,0,sigma], Ajsigma*v_L)-Ajsigma[sigma])*np.sqrt(np.real(v_L[sigma]));
        r_el = r_flux/i_flux;
        Rcoef_to_add = r_el*np.conjugate(r_el);
        if(abs(np.imag(Rcoef_to_add))>1e-10): 
            print("Imag(Rs[{:.0f}]) = {:.10f}".format(sigma, np.imag(Rcoef_to_add)));
            assert(abs(np.imag(Rcoef_to_add))<1e-10);
        Rs[sigma] = np.real(Rcoef_to_add);
        # sqrt of t flux, numerator of eq:Tcoef in manuscript
        t_flux = complex(0,1)*np.dot(Gmat[N+1,0,sigma], Ajsigma*v_L)*np.sqrt(np.real(v_R[sigma]));
        t_el = t_flux/i_flux;
        Tcoef_to_add = t_el*np.conjugate(t_el);
        if(abs(np.imag(Tcoef_to_add))>1e-10): 
            print("Imag(Ts[{:.0f}]) = {:.10f}".format(sigma, np.imag(Tcoef_to_add)));
            assert(abs(np.imag(Tcoef_to_add))<1e-10);
        Ts[sigma] = np.real(Tcoef_to_add);
    
    return Rs, Ts;

def Hmat(h, tnn, tnnn) -> np.ndarray:
    '''
    Make the hamiltonian H for reduced dimensional N+2 x N+2 system
    where there are N sites in the scattering region (SR), 1 LL site, 1 RL site
    Args
    -h, 2d array, on site blocks at each of the N+2 sites of the system
    -tnn, 2d array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, 2d array, next nearest neighbor hopping btwn sites, N-2 blocks

    returns 2d array with spatial and spin indices mixed
    '''
    if(not len(tnn) +1 == len(h)): raise ValueError;
    if(not len(tnnn)+2 == len(h)): raise ValueError;

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0]; # general dofs that are not the site number
    H =  np.zeros((n_loc_dof*(N+2), n_loc_dof*(N+2) ), dtype = complex);
    # outer shape: num sites x num sites (0 <= j <= N+1)
    # shape at each site: n_loc_dof, runs over all other degrees of freedom

    # construct H
    for sitei in range(0,N+2): # iter site dof only
        for sitej in range(0,N+2): # same
                
            for loci in range(np.shape(h[0])[0]): # iter over local dofs
                for locj in range(np.shape(h[0])[0]):

                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    if(sitei == sitej): # input from local h to main diag
                        H[ovi, ovj] = h[sitei][loci, locj];

                    elif(sitei == sitej+1): # input from tnn to lower diag <---- !!! change !!!
                        H[ovi, ovj] = tnn[sitej][loci, locj];

                    elif(sitei+1 == sitej): # input from tnn to upper diag
                        H[ovi, ovj] = tnn[sitei][loci, locj];

                    elif(sitei == sitej+2): # input from tnnn to 2nd lower diag
                        H[ovi, ovj] = tnnn[sitej][loci, locj];

                    elif(sitei+2 == sitej): # input from tnnn to 2nd upper diag
                        H[ovi, ovj] = tnnn[sitei][loci, locj];
                        
    return H; # end Hmat

def Hprime(h, tnn, tnnn, tl, E, verbose = 0) -> np.ndarray:
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    -tl, float, hopping in leads, distinct from hopping within SR def'd by tnn, tnnn
    -E, float, energye of the state to evaluate the self energy at. NB -2*tl <= E <= +2*tl

    returns 2d array with spatial and spin indices mixed up
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0]; # encompasses all dofs that are not the site number

    # base hamiltonian
    Hp = Hmat(h, tnn, tnnn); # SR on site, hopping blocks
    
    # compute left lead self energies directly through g_ functions
    gLmat = g_closed(h[0], -tl*np.eye(n_loc_dof), E, -1); # argument of surface gf = *lead* hopping
    # matrices multiplying g = coupling of SR to leads, which here is tnn[0] (tnn[-1]) for the left (right) lead
    # my convention is that hopping^\dagger is on lower diagonal. 
    # See Khomyakov 2005 Eq. (22), NB the \dagger convention there is flipped
    assert(np.shape(gLmat) == np.shape(tnn[0]));
    SigmaLmat = np.matmul(np.conj(tnn[0]).T, np.matmul(gLmat, tnn[0]));   
    
    # right lead self energies
    gRmat = g_closed(h[-1], -tl*np.eye(n_loc_dof), E, 1); # argument of surface gf = *lead* hopping
    SigmaRmat = np.matmul(tnn[-1], np.matmul(gRmat, np.conj(tnn[-1]).T)); 

    # check that modes with given energy are allowed in *at least some* LL channels
    assert(np.any(np.imag(np.diag(SigmaLmat))) );
    for sigmai in range(n_loc_dof):
        if(abs(np.imag(SigmaLmat[sigmai,sigmai])) > 1e-10 and abs(np.imag(SigmaRmat[sigmai,sigmai])) > 1e-10 ):
            assert(np.sign(np.imag(SigmaLmat[sigmai,sigmai])) == np.sign(np.imag(SigmaRmat[sigmai,sigmai])));
    if(verbose > 3):
        ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tl)); # vector running over sigma
        ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tl));
        for sigmai in range(n_loc_dof):
            print(" - chan "+str(sigmai)+", kL = {:.3f}+{:.3f}j, SigmaL = {:.3f}+{:.3f}j, -teika = {:.3f}+{:.3f}j"
                  .format(np.real(ka_L[sigmai]), np.imag(ka_L[sigmai]), 
                   np.real(SigmaLmat[sigmai,sigmai]), np.imag(SigmaLmat[sigmai,sigmai]),
                   np.real(-tl*np.exp(complex(0,ka_L[sigmai]))), np.imag(-tl*np.exp(complex(0,ka_L[sigmai])))));
            print(" - chan "+str(sigmai)+", kR = {:.3f}+{:.3f}j, SigmaR = {:.3f}+{:.3f}j, -teika = {:.3f}+{:.3f}j"
                  .format(np.real(ka_R[sigmai]), np.imag(ka_R[sigmai]), 
                   np.real(SigmaRmat[sigmai,sigmai]), np.imag(SigmaRmat[sigmai,sigmai]),
                   np.real(-tl*np.exp(complex(0,ka_R[sigmai]))), np.imag(-tl*np.exp(complex(0,ka_R[sigmai])))));
                  
    # add self energies to Hprime
    Hp[0:n_loc_dof, 0:n_loc_dof] += SigmaLmat;
    Hp[-n_loc_dof:, -n_loc_dof:] += SigmaRmat;

    return Hp;
    
def g_closed(diag, offdiag, E, inoutsign) -> np.ndarray:
    '''
    Surface Green's function of a periodic semi-infinite tight-binding lead
    The closed form comes from the diagonal and off-diagonal spatial blocks both being 
    diagonal in channel space, so Eq. 7 of my PRA paper is realized
    
    Returns: 
    '''
    if(np.shape(diag) != np.shape(offdiag) or np.shape(diag) == ()): raise ValueError;
    if(inoutsign not in [1,-1]): raise ValueError;
    # check diagonality
    if(np.any(diag-np.diagflat(np.diagonal(diag)))): raise Exception("Not diagonal:\n"+str(diag)); 
    
    # everything is vectorized by channel
    diag = np.diagonal(diag);
    offdiag = np.diagonal(offdiag);
    
    # scale the energy
    lam = (E-diag)/(2*offdiag); # offdiag includes - sign
    # make sure the sign of Im[g] is correctly assigned
    assert( np.max(abs(np.imag(lam))) < 1e-10);
    Lambda_minusplus = np.real(lam) + inoutsign*np.lib.scimath.sqrt(np.real(lam)*np.real(lam) - 1);
    
    # return as same sized array
    if  (inoutsign ==-1): return np.diagflat((1/offdiag)/Lambda_minusplus); # incoming state (left lead)
    elif(inoutsign == 1): return np.diagflat((1/offdiag)*Lambda_minusplus); # outgoing state (right lead)
    
def g_iter(diag, offdiag, E, ith, g_prev, inoutsign, imE = 1e-3) -> np.ndarray:
    '''
    '''
    if(np.shape(diag) != np.shape(offdiag) or np.shape(diag) != np.shape(g_prev)): raise ValueError;
    if(not isinstance(ith, int)): raise TypeError;
    if(inoutsign not in [1,-1]): raise ValueError;
    if(inoutsign != 1): raise NotImplementedError; # the order of offdiag, offdiag^\dagger differs in LL
    eye_like = np.eye(len(diag));
    
    # g_retarded \equiv lim(\eta->0) g(E+i\eta)
    E = complex(np.real(E),imE); # NB the E argument is in general complex
     
    if(ith==0): # 0th iteration
        return np.linalg.inv(eye_like*E - diag);
        
    else: # higher iteration
        return np.linalg.inv(eye_like*E - diag - np.matmul(offdiag, np.matmul(g_prev, np.conj(offdiag.T))));

def Green(h, tnn, tnnn, tl, E, verbose = 0) -> np.ndarray:
    '''
    Greens function for system described by
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    -tl, float, hopping in leads, distinct from hopping within SR def'd by above arrays
    -E, float, incident energy

    returns 4d array with spatial and spin indices separate
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # get system Green's function in matrix form
    # for easy inversion, Hp should be 2d array with spatial and spin indices mixed
    Hp = Hprime(h, tnn, tnnn, tl, E, verbose=verbose); 
    Gmat = np.linalg.inv( E*np.eye(*np.shape(Hp)) - Hp );

    # make 4d
    Gmat = fci_mod.mat_2d_to_4d(Gmat, n_loc_dof); # separates spatial and spin indices
    return Gmat;

##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








