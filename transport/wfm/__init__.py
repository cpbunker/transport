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
#### figure standardization

UniversalColors = ["darkblue", "darkgreen", "darkred", "darkcyan", "darkgray","darkorange", "darkmagenta","hotpink", "saddlebrown"];
UniversalAccents = ["black","red"];
ColorsMarkers = ["o","^","s","d","*"];
AccentsMarkers = ["X","+"];
UniversalMarkevery = (40,40);
UniversalPanels = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)"];

##################################################################################
#### driver of transmission coefficient calculations

def kernel(h, tnn, tnnn, tl, E, converger, Ajsigma, 
           is_psi_jsigma, is_Rhat, all_debug = True, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    Args
    -h, array, block hamiltonian matrices
    -tnn, array, upper diagonal nearest neighbor block hopping matrices
    -tnnn, array, upper diagonal next nearest neighbor block hopping matrices
    -tl, float, hopping in leads, not necessarily same as hopping on/off SR
        or within SR which is defined by tnn, tnnn matrices
    -E, complex, energy of the incident electron
    -imE, float, the small imaginary part of the energy (if the iterative scheme for the surface green's function is used)
    -conv_tol, float, the convergence criteria (if the iterative scheme is used)
    -Ajsigma, incident particle amplitude at site 0 in spin channel sigma
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
    
    # check that lead blocks are okay
    for hi in [0, -1]: # LL, RL
        isdiag = h[hi] - np.diagflat(np.diagonal(h[hi])); # subtract off diag
        if(all_debug and np.any(isdiag)): # True if there are nonzero off diag terms
            #raise Exception("Not diagonal\n"+str(np.real(h[hi])))
            pass # diagonality does not apply for diatomic unit cell
    for sigma in range(len(Ajsigma)): # always set incident mu = 0
        if(Ajsigma[sigma] != 0):
            pass; # for diatomic unit cell, u != 0 breaks this
            assert(h[0,sigma,sigma] == 0);

    # check incident amplitude
    assert( isinstance(Ajsigma, np.ndarray));
    assert( len(Ajsigma) == np.shape(h[0])[0] );

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # compute velocities directly through g_ functions -> vectors over channels
    SigmaLmat, SigmaRmat = SelfEnergies(h, tnn, tnnn, E, converger, verbose = verbose);
    if(all_debug): # assert the Sigmas are diagonal
        for Sigmamat in [SigmaLmat, SigmaRmat]:
            Sigma_isdiag = Sigmamat - np.diagflat(np.diagonal(Sigmamat));
            assert(not np.any(Sigma_isdiag));
    if(all_debug): assert(np.any(np.imag(np.diagonal(SigmaLmat))) );
    # ^^ checks that given energy allows propagating modes in *at least some* LL channels
 
    # velocities
    # note that bc Sigma is added only to the boundary sites, v_LB = v_RA = 0
    v_L, v_R = -2*np.imag(np.diagonal(SigmaLmat)), -2*np.imag(np.diagonal(SigmaRmat));

    # Green's function
    if(verbose): print("\nEnergy = {:.6f}".format(np.real(E))); # start printouts
    Gmat = Green(h, tnn, tnnn, tl, E, converger, verbose = verbose); # spatial and spin indices separate
    
    # from Green's function, determine wavefunction elements \psi_j\sigma
    psi_jsigma = complex(0,1)*np.dot(Gmat[:,0], Ajsigma*v_L);
    if(is_psi_jsigma): return psi_jsigma;
    
    # from Green's func, determine matrix elements < \sigma | rhat | \sigma'> of the
    # reflection operator Rhat, which scatters \sigma' -> \sigma
    #Rhat_matrix = 2*tl*complex(0,1)*Gmat[0,0]*np.sin(ka_L) - np.eye(n_loc_dof);
    # need to update Rhat for Rice-Mele
    if(is_Rhat): return Rhat_matrix;

    # determine matrix elements
    i_flux = np.sqrt(np.dot(Ajsigma, Ajsigma*np.real(v_L))); # sqrt of i flux

    # from matrix elements, determine R and T coefficients
    # (eq:Rcoef and eq:Tcoef in paper)
    Rs = np.zeros(n_loc_dof, dtype = float); # force as float bc we check that imag part is tiny
    Ts = np.zeros(n_loc_dof, dtype = float);
    for sigma in range(n_loc_dof): # iter over internal dofs
        # sqrt of r flux, numerator of eq:Rcoef 
        r_flux = (complex(0,1)*np.dot(Gmat[0,0,sigma], Ajsigma*v_L)-Ajsigma[sigma])*np.sqrt(np.real(v_L[sigma]));
        r_el = r_flux/i_flux;
        Rcoef_to_add = r_el*np.conjugate(r_el);
        if(abs(np.imag(Rcoef_to_add))>1e-10): 
            print("Imag(Rs[{:.0f}]) = {:.10f}".format(sigma, np.imag(Rcoef_to_add)));
            assert(abs(np.imag(Rcoef_to_add))<1e-10);
        Rs[sigma] = np.real(Rcoef_to_add);
        # sqrt of t flux, numerator of eq:Tcoef 
        t_flux = complex(0,1)*np.dot(Gmat[N+1,0,sigma], Ajsigma*v_L)*np.sqrt(np.real(v_R[sigma]));
        t_el = t_flux/i_flux;
        Tcoef_to_add = t_el*np.conjugate(t_el);
        if(abs(np.imag(Tcoef_to_add))>1e-10): 
            print("Imag(Ts[{:.0f}]) = {:.10f}".format(sigma, np.imag(Tcoef_to_add)));
            assert(abs(np.imag(Tcoef_to_add))<1e-10);
        Ts[sigma] = np.real(Tcoef_to_add);
    
    # NB R and T will be vectors over site flavor and spin channel
    # but due to velocities being nonzero only on boundary sites, R and T are
    # nonzero only at boundary sites
    if(all_debug): assert(abs(np.sum(Rs)+np.sum(Ts) - 1)<1e-10);
    # user should extract Rs at first site flavor, Ts at last site flavor
    return Rs, Ts;

def Green(h, tnn, tnnn, tl, E, converger, verbose = 0) -> np.ndarray:
    '''
    Greens function for system described by
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, upper diagonal nearest neighbor hopping, N-1 blocks
    -tnnn, array, upper diagonal next nearest neighbor hopping, N-2 blocks
    -tl, float, hopping in leads, distinct from hopping within SR def'd by above arrays
    -E, complex, energy of the incident electron
    -converger, either a keyword to use a closed-form green's function, 
     or (if the iterative scheme for the surface green's function is used) tuple of
       -imE, float, the small imaginary part of the energy 
       -conv_tol, float, the convergence criteria 
    returns 4d array with spatial and spin indices separate
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # get system Green's function in matrix form
    # for easy inversion, Hp should be 2d array with spatial and spin indices mixed
    Hp = Hprime(h, tnn, tnnn, tl, E, converger, verbose=verbose); 
    Gmat = np.linalg.inv( E*np.eye(*np.shape(Hp)) - Hp );

    # make 4d
    Gmat = fci_mod.mat_2d_to_4d(Gmat, n_loc_dof); # separates spatial and spin indices
    return Gmat;

def Hmat(h, tnn, tnnn) -> np.ndarray:
    '''
    Make the hamiltonian H for reduced dimensional N+2 x N+2 system
    where there are N sites in the scattering region (SR), 1 LL site, 1 RL site
    Args
    -h, 2d array, on site blocks at each of the N+2 sites of the system
    -tnn, 2d array, upper diagonal nearest neighbor hopping, N-1 blocks
    -tnnn, 2d array, upper diagonal next nearest neighbor hopping, N-2 blocks

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

                    elif(sitei == sitej+1): # input from tnn to lower diag
                        H[ovi, ovj] = (np.conj(tnn[sitej]).T)[loci, locj];

                    elif(sitei+1 == sitej): # input from tnn to upper diag
                        H[ovi, ovj] = tnn[sitei][loci, locj];

                    elif(sitei == sitej+2): # input from tnnn to 2nd lower diag
                        H[ovi, ovj] = (np.conj(tnnn[sitej]).T)[loci, locj];

                    elif(sitei+2 == sitej): # input from tnnn to 2nd upper diag
                        H[ovi, ovj] = tnnn[sitei][loci, locj];
                        
    return H; # end Hmat

def Hprime(h, tnn, tnnn, tlmag, E, converger, verbose = 0) -> np.ndarray:
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, upper diagonal nearest neighbor hopping, N-1 blocks
    -tnnn, array, upper diagonal next nearest neighbor hopping, N-2 blocks
    -tl, float>0, sets scale of hopping in leads (diff from hopping in SR)
    -E, complex, energy to evaluate the self energy at
    -converger, either a keyword to use a closed-form green's function, 
     or (if the iterative scheme for the surface green's function is used) tuple of
       -imE, float, the small imaginary part of the energy 
       -conv_tol, float, the convergence criteria 
    returns 2d array with spatial and spin indices mixed up
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0]; # encompasses all dofs that are not the unit cell index
                                   # here that means site flavor and spin

    # base hamiltonian
    Hp = Hmat(h, tnn, tnnn); # SR on site, hopping blocks
   
    # compute lead self energies directly through g_ functions
    SigmaLmat, SigmaRmat = SelfEnergies(h, tnn, tnnn, E, converger, verbose = verbose);

    # add self energies to Hprime
    Hp[0:n_loc_dof, 0:n_loc_dof] += SigmaLmat;
    Hp[-n_loc_dof:, -n_loc_dof:] += SigmaRmat;

    for sigmai in range(n_loc_dof):
        if(abs(np.imag(SigmaLmat[sigmai,sigmai])) > 1e-10 and abs(np.imag(SigmaRmat[sigmai,sigmai])) > 1e-10 ):
            assert(np.sign(np.imag(SigmaLmat[sigmai,sigmai])) == np.sign(np.imag(SigmaRmat[sigmai,sigmai])));
    if(verbose > 3):
        ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tlmag)); # vector running over sigma
        ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tlmag));
        for sigmai in range(n_loc_dof):
            print(" - chan "+str(sigmai)+", SigmaL = {:.3f}+{:.3f}j, -teika = {:.3f}+{:.3f}j"
                  .format( 
                   np.real(SigmaLmat[sigmai,sigmai]), np.imag(SigmaLmat[sigmai,sigmai]),
                   np.real(-tlmag*np.exp(complex(0,ka_L[sigmai]))), np.imag(-tlmag*np.exp(complex(0,ka_L[sigmai])))));
            print(" - chan "+str(sigmai)+", SigmaR = {:.3f}+{:.3f}j, -teika = {:.3f}+{:.3f}j"
                  .format(
                   np.real(SigmaRmat[sigmai,sigmai]), np.imag(SigmaRmat[sigmai,sigmai]),
                   np.real(-tlmag*np.exp(complex(0,ka_R[sigmai]))), np.imag(-tlmag*np.exp(complex(0,ka_R[sigmai])))));
                  
    if(verbose>4): 
        print("Re[Hp]=\n",np.real(Hp));
        print("Im[Hp]=\n",np.imag(Hp)); 
    return Hp;

def SelfEnergies(h, tnn, tnnn, E, converger, verbose = 0) -> tuple:
    '''
    Self energy of each lead (left, right) for a two-lead system
    Args: (see Hprime above)
    -E, complex, energy to evaluate the self energy at
       NB the g_closed function assumes a band -2*tl <= E <= +2*tl
       More complicated systems will have different bands

    Returns: tuple of left lead self-energy, right lead self-energy
             each are matrices acting on both site flavor and spin dofs
    '''

    # compute lead self energies directly through g_ functions
    gLargs = (h[0], tnn[0], E, -1); 
    gRargs = (h[-1], tnn[-1], E, 1); 
    if(converger=="g_closed"): # closed-form monatomic surface green's func
        gLmat = g_closed(*gLargs);
        gRmat = g_closed(*gRargs);
    elif(converger=="g_RiceMele"): # closed-form diatomic surface green's func
        gLmat = g_RiceMele(*gLargs);
        gRmat = g_RiceMele(*gRargs);
    elif(isinstance(converger,tuple)): # iterative soln of the surface green's func
        gLmat = g_iter(*gLargs, *converger);
        gRmat = g_iter(*gRargs, *converger);
    else: raise Exception("converger = "+str(converger)+" not supported");

    # get the self energy matrices
    # matrices multiplying g = coupling of SR to leads, which here is tnn[0] (tnn[-1]) for the left (right) lead
    # my convention is that hopping^\dagger is on lower diagonal. 
    # See Khomyakov 2005 Eq. (22), NB the \dagger convention there is flipped
    assert(np.shape(gLmat) == np.shape(tnn[0]));
    SigmaLmat = np.matmul(np.conj(tnn[0]).T, np.matmul(gLmat, tnn[0]));
    SigmaRmat = np.matmul(tnn[-1], np.matmul(gRmat, np.conj(tnn[-1]).T)); 

    return SigmaLmat, SigmaRmat;
    
def g_closed(diag, offdiag, E, inoutsign) -> np.ndarray:
    '''
    Surface Green's function of a periodic semi-infinite tight-binding lead
    The closed form comes from the diagonal and off-diagonal spatial blocks both being 
    diagonal in channel space, so Eq. 7 of my PRA paper is realized

    Args:
    -diag, matrix in channel space, same-spatial-site matrix elements of H
    -off_diag, matrix in channel space, upper diagonal nearest-neighbor matrix elems of H
    -E, complex, band energy. **this function assumes** -2*tl <= E <= +2*tl
    -inoutsign, telling us if we are computing incoming or outgoing state
    
    Returns: 
    -the surface green's function, matrix in channel space--same shape as diag
       -->Properties:
          Im[g] < 0 always, symmetric function of (E)
          Re[g] has sign(E), antisymmetric function of (E)
    '''
    if(np.shape(diag) != np.shape(offdiag) or np.shape(diag) == ()): raise ValueError;
    if(inoutsign not in [1,-1]): raise ValueError;
    # check diagonality
    if(np.any(diag-np.diagflat(np.diagonal(diag)))): raise Exception("Not diagonal:\n"+str(diag)); 
    # everything is vectorized by channel
    diag = np.diagonal(diag);
    offdiag = np.diagonal(offdiag);

    #raise NotImplementedError("everything should call g_RiceMele");

    # this decomposition gives correct sign of sqrt always, but need to double-check
    if(True):
        #raise NotImplementedError("Need to verify all plots")
        reduced = (E-diag)/(2*offdiag);
        term1 = reduced/offdiag;
        term2 = np.lib.scimath.sqrt(reduced*reduced-1);
        return np.diagflat(np.real(term1)-np.sign(E-diag)*abs(np.real(term2))-complex(0,1)*abs(np.imag(term2)));
    
    # scale the energy
    lam = (E-diag)/(2*offdiag); # offdiag includes - sign
    # make sure the sign of Im[g] is correctly assigned
    assert( np.max(abs(np.imag(lam))) < 1e-10);
    Lambda_minusplus = np.real(lam) + inoutsign*np.lib.scimath.sqrt(np.real(lam)*np.real(lam) - 1);
    
    # return as same sized array
    if  (inoutsign ==-1): return np.diagflat((1/offdiag)/Lambda_minusplus); # incoming state (left lead)
    elif(inoutsign == 1): return np.diagflat((1/offdiag)*Lambda_minusplus); # outgoing state (right lead)

def g_RiceMele(diag, offdiag, E, inoutsign) -> np.ndarray:
    '''
    Surface Green's function of the Rice-Mele model
    The closed form comes from Yen-Ting Lin's thesis at Aachen 

    Args:
    -diag, matrix in channel space, same-spatial-site matrix elements of H
    -off_diag, matrix in channel space, upper diagonal nearest-neighbor matrix elmements of H
    -E, complex, 
    -inoutsign, telling us if we are computing incoming or outgoing state
    
    Returns: 
    -the surface green's function, matrix in channel space--same shape as diag
       ie has both site flavor and spin dofs
    '''
    if(np.shape(diag) != np.shape(offdiag) or np.shape(diag) == ()): raise ValueError;
    if(inoutsign not in [1,-1]): raise ValueError;
    # check RM compatibility
    if(len(diag)%2 != 0): raise ValueError("diag is not Rice-Mele type");
    n_spin = len(diag)//2; # 2 internal dofs are A-B orbital, rest are spin
    offdiag_check = 1*offdiag;
    offdiag_check[n_spin:,:n_spin] = 0*offdiag[n_spin:,:n_spin];
    # ^ remove lower left, which should be only nonzero part
    if(np.any(offdiag_check)): raise ValueError("offdiag is not Rice-Mele type");
    
    # decompose into u, w, v
    # for spin included, these will be vectors over spin channels (length nspin)
    u = (np.diagonal(diag)[:n_spin]-np.diagonal(diag)[n_spin:])/2; 
    v = np.diagonal(diag[:n_spin,n_spin:]);
    w = np.diagonal(offdiag[n_spin:,:n_spin]);

    # not totally sure how to deal with u0 at this point
    u0 = (np.diagonal(diag)[:n_spin]+np.diagonal(diag)[n_spin:])/2;
    for u0_sigmasigma in u0: assert(abs(u0_sigmasigma)<1e-10);
    # ^ just to flag when u0 turns on
    if False:
        print("u  [{:.0f}] = ".format(len(u)), np.real(u));
        print("v  [{:.0f}] = ".format(len(v)), np.real(v));
        print("w  [{:.0f}] = ".format(len(w)), np.real(w));
        print("u0 [{:.0f}] = ".format(len(u0)),np.real(u0));
        assert False

    def Z_of_x(x):
        return (x+u)*(x-u)+(w+v)*(w-v);

    # functional form
    prefactor = 1/(2*w*w*(E-u0-u));
    squared_val = np.power( Z_of_x(E-u0), 2);
    sqrt_val = np.lib.scimath.sqrt( 4*w*w*(u+E-u0)*(u-E+u0)+squared_val)
    for sigma in range(n_spin):   # Enforce Im[g_ret]<0
        if(np.imag(sqrt_val[sigma]*prefactor[sigma]) > 0): sqrt_val[sigma] = (-1)*sqrt_val[sigma]; 
    g = prefactor*( Z_of_x(E-u0) + sqrt_val);

    # return as same sized array
    gmat = np.zeros(np.shape(diag), dtype=complex);
    if(inoutsign ==-1): # for left lead, fill all spin channels of last B orb
          # others components remain 0 since multiplication by h_01 zeros them out
        gmat[:n_spin,:n_spin] = 0*np.diagflat(g);    # <A|g_00|A>
        gmat[n_spin:,n_spin:] = np.diagflat(g);      # <B|g_00|B>
    elif(inoutsign== 1): # for right lead, fill all spin channels of first A orb
        gmat[:n_spin,:n_spin] = np.diagflat(g);      # <A|g_00|A>
        gmat[n_spin:,n_spin:] = 0*np.diagflat(g);    # <B|g_00|B>
    return gmat;

def dispersion_RiceMele(diag, offdiag, ks) -> np.ndarray:
    '''
    *Vectorized in ks* dispersion function E(k)
    Returns an array of shape (2,len(ks)) where 2 runs over the plus, minus bands
    '''

    # decompose into u, w, v
    u0 = np.sum(np.diagonal(diag))/len(diag); 
    u = (diag[0,0]-diag[1,1])/2; assert(len(diag)==2);
    v = diag[0,-1];
    w = offdiag[-1,0];

    band = np.sqrt(u*u+v*v+w*w+2*v*w*np.cos(ks));
    return np.array([u0-band,u0+band]);
    
def inverted_RiceMele(diag, offdiag, Es) -> np.ndarray:
    '''
    inverted dispersion function E(k)
    Returns an array of shape (2,len(ks)) where 2 runs over the plus, minus bands
    '''

    # decompose into u, w, v
    u0 = np.sum(np.diagonal(diag))/len(diag); 
    u = (diag[0,0]-diag[1,1])/2; assert(len(diag)==2);
    v = diag[0,-1];
    w = offdiag[-1,0];

    ret = np.arccos(1/(2*v*w)*((Es-u0)**2 - u**2 - v**2 - w**2));
    return ret;

def bandedges_RiceMele(diag, offdiag) -> np.ndarray:
    '''
    '''

    # decompose into u, w, v
    u0 = np.sum(np.diagonal(diag))/len(diag); 
    u = (diag[0,0]-diag[1,1])/2; assert(len(diag)==2);
    v = diag[0,-1];
    w = offdiag[-1,0];

    B_pm = np.array([np.sqrt(u*u+(w+v)*(w+v)),
                       np.sqrt(u*u+(w-v)*(w-v))]);
    B_edges = np.array([u0 - np.max(B_pm), u0-np.min(B_pm), u0+np.min(B_pm), u0+np.max(B_pm)]);
    return B_edges

def string_RiceMele(diag, offdiag, energies=True, tex=True) -> str:
    '''
    '''

    # decompose into u, w, v
    u0 = np.sum(np.diagonal(diag))/len(diag); 
    u = (diag[0,0]-diag[1,1])/2; assert(len(diag)==2);
    v = diag[0,-1];
    w = offdiag[-1,0];

    B_edges = bandedges_RiceMele(diag, offdiag);
    # strings
    params_str = "$u = {:.2f}, v ={:.2f}, w ={:.2f}$".format(u,v,w);
    energies_str = ", $E_{gap}="+"{:.2f}".format(B_edges[2]-B_edges[1])+", E_{band}=$"+"{:.2f}$".format(B_edges[1]-B_edges[0]);
    if(energies): params_str += energies_str;
    if(not tex): params_str = params_str.replace("$","");
    return params_str;

def g_iter(diag, offdiag, E, inoutsign, imE, conv_tol, 
              conv_rep=5, min_iter=int(1e2), max_iter=int(1e5), full=False) -> np.ndarray:
    '''
    Surface Green's function of a periodic semi-infinite tight-binding lead
    When the diag and off-diagonal blocks are not diagonal in channel basis,
    we must solve iteratively
    Args:
    -diag, matrix in channel space, same-spatial-site block
    -off_diag, matrix in channel space, upper diagonal nearest-neighbor block
    -E, complex, band energy (can be negative, complex type but im(E)=0
    -inoutsign, telling us if we are computing incoming or outgoing state
    -imE, float, the small imaginary part to add to the real energy
    -conv_tol, float, the threshold relative change in the surface dos for convergence
    -conv_rep, int, the number of iterations that must meet conv_tol before full convergence is achieved
    -min_iter, int, the number of iterations to do before checking for convergence
    -max_iter, int, the number of iterations to do before declaring failure
    -full, bool, tells us whether to return the surface green's function, the convergence metric, and the number of iterations used, or just the former
    
    Returns: 
    -the surface green's function, matrix in channel space--same shape as diag
    '''
    if(np.shape(diag) != np.shape(offdiag)): raise ValueError;
    if(inoutsign not in [1,-1]): raise ValueError;
    if(len(diag)>2): raise NotImplementedError("Iterative diatomic and spin combined not supported");

    # iterative g
    g = np.zeros_like(diag);
    dos_ith = np.zeros((max_iter,), dtype=float); # convergence metric
    conv_ith = np.zeros((max_iter,), dtype=int); # tracks whether [i,i-1] met conv tol

    # repeated iterations
    for ith in range(max_iter):

        # update g
        g = g_ith(diag, offdiag, E, inoutsign, imE, ith, g); 

        # update convergence metric (surface density of states)
        if(inoutsign==1): dos_ith[ith] = (-1/np.pi)*np.imag(g)[0,0];
        elif(inoutsign==-1): dos_ith[ith] = (-1/np.pi)*np.imag(g)[-1,-1];

        # check convergence
        if(ith>min_iter):
            conv_check = abs((dos_ith[ith]-dos_ith[ith-1])/dos_ith[ith])
            if(conv_check<conv_tol): conv_ith[ith] = 1; # this iter met tol
            if(np.sum(conv_ith[ith+1-conv_rep:ith+1])==conv_rep):
                # fully converged 
                #print(">>> {:.0f} >>> {:.6f} >>> {:.6f} >>> {:.0f}".format(ith,np.real(E),conv_check,conv_ith[ith]));
                if(full): return g, abs((dos_ith[ith]-dos_ith[ith-1])/dos_ith[ith]), ith;
                else: return g;

    # if we got here, we never converged
    raise Exception("g({:.6f}+{:.6f}j) convergence was {:.0e}, failed to meet {:.0e} tolerance after {:.0e} iterations".format(np.real(E), imE, conv_check, conv_tol, max_iter));
    
def g_ith(diag, offdiag, E, inoutsign, imE, ith, g_prev) -> np.ndarray:
    '''
    Single iteration of the surface Green's function
    See Sec C.1 of Zuxin's "Molecular Junction" notes
    '''
    if(not isinstance(ith, int)): raise TypeError;
    eye_like = np.eye(len(diag));
    
    # g_retarded \equiv lim(\eta->0) g(E+i\eta)
    E = complex(np.real(E),abs(imE)); # NB the E argument is in general complex
     
    if(ith==0): # 0th iteration
        return np.linalg.inv(eye_like*E - diag);
        
    else: # higher iteration
        if(inoutsign==1): # right lead outgoing state
            txgxt = np.matmul(offdiag, np.matmul(g_prev, np.conj(offdiag.T)));
        elif(inoutsign == -1): # left lead incoming state
            txgxt = np.matmul(np.conj(offdiag.T), np.matmul(g_prev, offdiag));
        return np.linalg.inv(eye_like*E - diag - txgxt);

##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








