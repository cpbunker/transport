'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
general formalism: all sites map to all the different
degrees of freedom of the system
'''

from transport import fci_mod

import numpy as np

##################################################################################
#### driver of transmission coefficient calculations

def kernel(h, tnn, tnnn, tl, E, Ajsigma, verbose = 0, all_debug = True):
    '''
    coefficient for a transmitted up and down electron
    Args
    -h, array, block hamiltonian matrices
    -tnn, array, nearest neighbor block hopping matrices
    -tnnn, array, next nearest neighbor block hopping matrices
    -tl, float, hopping in leads, not necessarily same as hopping on/off SR
        or within SR which is defined by th matrices
    -E, float, energy of the incident electron
    -Ajsigma, incident particle amplitude at site 0 in spin channel j
    Optional args
    -verbose, how much printing to do
    -all_debug, whether to enforce a bunch of extra assert statements

    Returns
    tuple of R coefs (vector of floats for each sigma) and T coefs (likewise)
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
            #assert(h[0,sigma,sigma] == 0);

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
    v_R = 2*tl*np.sin(ka_R); # a, hbar defined as 1

    # green's function
    if(verbose): print("\nEnergy = {:.6f}".format(np.real(E+2*tl))); # start printouts
    Gmat = Green(h, tnn, tnnn, tl, E, verbose = verbose); # spatial and spin indices separate

    # determine matrix elements
    i_flux = np.sqrt(np.dot(Ajsigma, Ajsigma*np.real(v_L))); # sqrt of i flux

    # from matrix elements, determine R and T coefficients
    # (eq:Rcoef and eq:Tcoef in paper)
    Rs = np.zeros(n_loc_dof, dtype = float); # force as float bc we check that imag part is tiny
    Ts = np.zeros(n_loc_dof, dtype = float);
    for sigma in range(n_loc_dof): # iter over spin dofs
        # sqrt of r flux, numerator of eq:Rcoef in manuscript
        r_flux = (np.complex(0,1)*np.dot(Gmat[0,0,sigma], Ajsigma*v_L)-Ajsigma[sigma])*np.sqrt(np.real(v_L[sigma]));
        r_el = r_flux/i_flux;
        Rs[sigma] = np.real(r_el*np.conjugate(r_el));
        # sqrt of t flux, numerator of eq:Tcoef in manuscript
        t_flux = np.complex(0,1)*np.dot(Gmat[N+1,0,sigma], Ajsigma*v_L)*np.sqrt(np.real(v_L[sigma]));
        t_el = t_flux/i_flux;
        Ts[sigma] = np.real(t_el*np.conjugate(t_el));
    
    return Rs, Ts;

def Hmat(h, tnn, tnnn, verbose = 0):
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

                    elif(sitei == sitej+1): # input from tnn to lower diag
                        H[ovi, ovj] = tnn[sitej][loci, locj];

                    elif(sitei+1 == sitej): # input from tnn to upper diag
                        H[ovi, ovj] = tnn[sitei][loci, locj];

                    elif(sitei == sitej+2): # input from tnnn to 2nd lower diag
                        H[ovi, ovj] = tnnn[sitej][loci, locj];

                    elif(sitei+2 == sitej): # input from tnnn to 2nd upper diag
                        H[ovi, ovj] = tnnn[sitei][loci, locj];
                        
    return H; # end Hmat

def Hprime(h, tnn, tnnn, tl, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).
    Args
    -h, array, on site blocks at each of the N+2 sites of the system
    -tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    -tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    -tl, float, hopping in leads, distinct from hopping within SR def'd by tnn, tnnn

    returns 2d array with spatial and spin indices mixed
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0]; # general dofs that are not the site number

    # base hamiltonian
    Hp = Hmat(h, tnn, tnnn, verbose = verbose); # SR on site, hopping blocks
    
    # self energies in LL
    # need a self energy for all incoming/outgoing spin states (all local dof)
    SigmaLs = np.zeros(n_loc_dof, dtype = complex);
    for Vi in range(n_loc_dof): # iters over all local dof
        # scale the energy
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tl);
        # make sure sign of SigmaL is correctly assigned
        assert( abs(np.imag(lamL)) < 1e-10);
        lamL = np.real(lamL);
        # reflected self energy
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); 
        SigmaL = -tl/LambdaLminus; 
        Hp[Vi,Vi] += SigmaL;
        SigmaLs[Vi] = SigmaL
    del V, lamL, LambdaLminus, SigmaL

    # self energies in RL
    SigmaRs = np.zeros(n_loc_dof, dtype = complex);
    for Vi in range(n_loc_dof): # iters over all local dof
        # scale the energy
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tl);
        # make sure the sign of SigmaR is correctly assigned
        assert( abs(np.imag(lamR)) < 1e-10);
        lamR = np.real(lamR); # makes sure sign of SigmaL is correctly assigned
        # transmitted self energy
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1);
        SigmaR = -tl*LambdaRplus;
        Hp[Vi-n_loc_dof,Vi-n_loc_dof] += SigmaR;
        SigmaRs[Vi] = SigmaR;
    del V, lamR, LambdaRplus, SigmaR;

    # check that modes with given energy are allowed in some LL channels
    SigmaLs, SigmaRs = np.array(SigmaLs), np.array(SigmaRs);
    assert(np.any(np.imag(SigmaLs)) );
    for sigmai in range(len(SigmaLs)):
        if(abs(np.imag(SigmaLs[sigmai])) > 1e-10 and abs(np.imag(SigmaRs[sigmai])) > 1e-10 ):
            assert(np.sign(np.imag(SigmaLs[sigmai])) == np.sign(np.imag(SigmaRs[sigmai])));
    if(verbose > 3):
        ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tl)); # vector running over sigma
        ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tl));
        v_L = 2*tl*np.sin(ka_L); # a/hbar defined as 1
        v_R = 2*tl*np.sin(ka_R);
        for sigma in range(len(ka_L)):
            print(" - sigma = "+str(sigma)+", v_L = {:.4f}+{:.4f}j, Sigma_L = {:.4f}+{:.4f}j"
                  .format(np.real(v_L[sigma]), np.imag(v_L[sigma]), np.real(SigmaLs[sigma]), np.imag(SigmaLs[sigma])));
            print(" - sigma = "+str(sigma)+", v_R = {:.4f}+{:.4f}j, Sigma_R = {:.4f}+{:.4f}j"
                  .format(np.real(v_R[sigma]), np.imag(v_R[sigma]), np.real(SigmaRs[sigma]), np.imag(SigmaRs[sigma])));

    return Hp;

def Green(h, tnn, tnnn, tl, E, verbose = 0):
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

    # get 2d green's function matrix
    Hp = Hprime(h, tnn, tnnn, tl, E, verbose=verbose); # for easy inversion, 2d array with spatial and spin indices mixed
    #if(verbose): print(">>> H' = \n", Hp );
    #if(verbose): print(">>> EI - H' = \n", E*np.eye(np.shape(Hp)[0]) - Hp );
    Gmat = np.linalg.inv( E*np.eye(*np.shape(Hp)) - Hp );

    # make 4d
    Gmat = mat_2d_to_4d(Gmat, n_loc_dof); # separates spatial and spin indices
    return Gmat;

##################################################################################
#### utils

def scal_to_vec(scal, n_dof):
    '''
    Take a number or operator, which is a scalar in real space,
    energy space, etc and make it a constant vector in that space
    '''

    return np.full((n_dof, *np.shape(scal)), scal).T;

def vec_1d_to_2d(vec, n_loc_dof):
    '''
    Take a 1d vector (ie with spatial and spin dofs mixed)
    to a 2d vector(ie with spatial and spin dofs separated)
    '''
    if( not isinstance(vec, np.ndarray)): raise TypeError;
    if( len(vec) % n_loc_dof != 0): raise ValueError;

    # unpack
    n_spatial_dof = len(vec) // n_loc_dof;
    new_vec = np.zeros((n_spatial_dof,n_loc_dof), dtype=vec.dtype);

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only               
        for loci in range(n_loc_dof): # iter over local dofs

            # site, loc indices -> overall indices
            ovi = sitei*n_loc_dof + loci;

            # update
            new_vec[sitei, loci] = vec[ovi];

    return new_vec;

def mat_2d_to_4d(mat, n_loc_dof):
    '''
    Take a 2d matrix (ie with spatial and spin dofs mixed)
    to a 4d matrix (ie with spatial and spin dofs separated)
    '''
    if( not isinstance(mat, np.ndarray)): raise TypeError;
    if( len(mat) % n_loc_dof != 0): raise ValueError;

    # unpack
    n_spatial_dof = len(mat) // n_loc_dof;
    new_mat = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof), dtype=mat.dtype);

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only
        for sitej in range(n_spatial_dof): # same
                
            for loci in range(n_loc_dof): # iter over local dofs
                for locj in range(n_loc_dof):

                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    # update
                    new_mat[sitei, sitej, loci, locj] = mat[ovi,ovj];

    return new_mat;

def mat_4d_to_2d(mat):
    '''
    Take a 4d matrix (ie with spatial and spin dofs separated)
    to a 2d matrix (ie with spatial and spin dofs mixed)
    '''
    if( not isinstance(mat, np.ndarray)): raise TypeError;
    if( np.shape(mat)[0] != np.shape(mat)[1]): raise ValueError;
    if( np.shape(mat)[2] != np.shape(mat)[3]): raise ValueError;

    # unpack
    n_loc_dof = np.shape(mat)[-1];
    n_spatial_dof = np.shape(mat)[0];
    n_ov_dof = n_loc_dof*n_spatial_dof;
    new_mat = np.zeros((n_ov_dof,n_ov_dof), dtype=mat.dtype);

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only
        for sitej in range(n_spatial_dof): # same
                
            for loci in range(n_loc_dof): # iter over local dofs
                for locj in range(n_loc_dof):

                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    # update
                    new_mat[ovi,ovj] = mat[sitei, sitej, loci, locj];

    return new_mat;

##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








