'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
general formalism: all sites map to all the different
degrees of freedom of the system
'''

import fci_mod

import numpy as np

import sys

##################################################################################
####

def Hmat(h, t, verbose = 0):
    '''
    Make the hamiltonian H for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    h, 1d arr 
    t, float, hopping
    '''

    # check
    assert(len(h) == len(t) + 1);

    # unpack
    N = len(h) - 2; # num scattering region sites, ie N+2 = num spatial dof
    n_loc_dof = np.shape(h[0])[0]; # dofs that will be mapped onto row in H
    H =  np.zeros((n_loc_dof*(N+2), n_loc_dof*(N+2) ), dtype = complex);
    # outer shape: num sites x num sites (degree of freedom is loc of itinerant e)
    # shape at each site: runs over all other degrees of freedom)

    # first construct matrix of matrices
    for sitei in range(0,N+2): # iter sites dof only
        for sitej in range(0,N+2): # same
                
            for loci in range(np.shape(h[0])[0]): # iter over local dofs
                for locj in range(np.shape(h[0])[0]):
                    
                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    if(sitei == sitej): # input from local h to main diag
                        H[ovi, ovj] += h[sitei][loci, locj];

                    elif(sitei == sitej+1): # input from T to lower diag
                        H[ovi, ovj] += t[sitej][loci, locj];

                    elif(sitei+1 == sitej): # input from T to upper diag
                        H[ovi, ovj] += t[sitei][loci, locj];                

    if verbose > 3: print("\nH_SR[0] = \n",np.real(H[n_loc_dof:2*n_loc_dof,n_loc_dof:2*n_loc_dof]));
    return H; 


def Hprime(h, t, tl, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    h, block diag hamiltonian matrices
    t, block off diag hopping matrix
    tl, hopping in leads, not necessarily same as hopping on/off SR as def'd by t matrices
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # add self energies to hamiltonian
    Hp = Hmat(h, t, verbose = verbose); # regular ham

    # self energies at LL
    # need a self energy for each LL boundary condition
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tl);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL = -tl/LambdaLminus;
        Hp[Vi,Vi] = SigmaL;

    # self energies at RL
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tl);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR = -tl*LambdaRplus;
        Hp[Vi-n_loc_dof,Vi-n_loc_dof] = SigmaL;
    
    if verbose > 3: print("\nH' = \n",Hp);
    return Hp;


def Green(h, t, tl, E, verbose = 0):
    '''
    Greens function for system described by
    - potential V[i] at site i
    - lattice spacing a
    - incident mass m
    -incident energy E

    Assumes that incident flux is up spin only!!!!
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # get green's function matrix
    Hp = Hprime(h, t, tl, E, verbose = verbose);
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    # of interest is the qith row which contracts with the source q
    return G;


def Tcoef(h, t, tl, E, qi, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    h, block diag hamiltonian matrices
    t, block off diag hopping matrix
    tl, hopping in leads, not necessarily same as hopping on/off SR as def'd by t matrices
    E, energy of the incident electron
    qi, source vector (loc dof only)
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));
    assert( isinstance(t, np.ndarray));
    assert( isinstance(qi, np.ndarray));
    assert( len(qi) == np.shape(h[0])[0] );

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # self energies at LL
    # need a self energy for each LL boundary condition
    SigmaL = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tl);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL.append( -tl/LambdaLminus);

    # self energies at RL
    SigmaR = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tl);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR.append( -tl*LambdaRplus);

    # check self energies
    #print(">>>>",tl,E-V,SigmaR[0]);
    assert( isinstance(SigmaL[0], complex) and isinstance(SigmaR[0], complex)); # check right dtype

    # green's function
    G = Green(h, t, tl, E, verbose = verbose);
    if verbose > 3: print("\nG[:,qi] = ",G[:,qi]);

    # coefs
    qivector = np.zeros(np.shape(G)[0]);
    for j in range(len(qi)):
        qivector[j] = qi[j]; # fill
    Gqi = np.dot(G, qivector);
    Ts = np.zeros(n_loc_dof, dtype = complex);
    for Ti in range(n_loc_dof):
        
        # Caroli expression 
        Ts[Ti] = 4*np.imag(SigmaR[Ti])*Gqi[Ti-n_loc_dof]*np.imag(SigmaL[Ti])*Gqi.conj()[Ti-n_loc_dof];
        assert( abs(np.imag(Ts[Ti])) <= 1e-8);

    return tuple(Ts);


##################################################################################
#### functions specific to the cicc model

def E_disp(k,a,t):
    # vectorized conversion from k to E(k), measured from bottom of band
    return -2*t*np.cos(k*a);

def k_disp(E,a,t):
    return np.arccos(E/(-2*t))/a;

def h_cicc_eff(J, t, i1, i2, Nsites):
    '''
    construct hams
    formalism works by
    1) having 3 by 3 block's each block is differant site for itinerant e
          H_LL T    0
          T    H_SR T
          0    T    H_RL        T is hopping between leads and scattering region
    2) all other dof's encoded into blocks

    Args:
    - J, float, eff heisenberg coupling
    - t, float, hopping btwn sites
    = i1, int, site of 1st imp
    - i2, int, site of 2nd imp
    - Nsites, int, total num sites in SR
    '''
    
    # heisenberg interaction matrices
    Se_dot_S1 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,1,0,0,0,0,0,0],
                        [0,0,-1,0,2,0,0,0],
                        [0,0,0,-1,0,2,0,0],
                        [0,0,2,0,-1,0,0,0],
                        [0,0,0,2,0,-1,0,0],
                        [0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,1] ]);

    Se_dot_S2 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-1,0,0,2,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,-1,0,0,2,0],
                        [0,2,0,0,-1,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,2,0,0,-1,0],
                        [0,0,0,0,0,0,0,1] ]);

    # insert these local interactions
    h_cicc =[];
    for sitei in range(Nsites): # iter over all sites
        if(sitei == i1):
            h_cicc.append(Se_dot_S1);
        elif(sitei == i2):
            h_cicc.append(Se_dot_S2);
        else:
            h_cicc.append(np.zeros_like(Se_dot_S1) );
    h_cicc = np.array(h_cicc);

    # hopping connects like spin orientations only, ie is identity
    tl_arr = []
    for sitei in range(Nsites-1):
        tl_arr.append(-t*np.eye(*np.shape(Se_dot_S1)) );
    tl_arr = np.array(tl_arr);

    return h_cicc, tl_arr;


if __name__ == "__main__": # test code

    pass;





    
    


    








