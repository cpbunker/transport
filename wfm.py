'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
general formalism: all sites map to all the different
degrees of freedom of the system
'''

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


def Hprime(h, t, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    h, 1d arr, length N+2, on site energies
    t, float, hopping
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];
    tval = t[0][0,0];

    # add self energies to hamiltonian
    Hp = Hmat(h, t, verbose = verbose); # regular ham

    # self energies at LL
    # need a self energy for each LL boundary condition
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tval);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL = -tval/LambdaLminus;
        Hp[Vi,Vi] = SigmaL;

    # self energies at RL
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tval);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR = -tval*LambdaRplus;
        Hp[Vi-n_loc_dof,Vi-n_loc_dof] = SigmaL;
    
    if verbose > 3: print("\nH' = \n",Hp);
    return Hp;


def Green(h, t, E, qi, verbose = 0):
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
    tval = t[0][0,0];

    # get green's function matrix
    Hp = Hprime(h, t, E, verbose = verbose);
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    # of interest is the qith row which contracts with the source q
    if verbose > 3: print("\nG[:,qi] = ",G[:,qi]);
    return G;


def Tcoef(h, t, E, qi, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];
    tval = t[0][0,0];

    # self energies at LL
    # need a self energy for each LL boundary condition
    SigmaL = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tval);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL.append( -tval/LambdaLminus);

    # self energies at RL
    SigmaR = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tval);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR.append( -tval*LambdaRplus);

    # check self energies
    assert( isinstance(SigmaL[0], complex) and isinstance(SigmaR[0], complex)); # check right dtype

    # green's function
    G = Green(h, t, E, qi, verbose = verbose);

    # coefs
    Ts = np.zeros(n_loc_dof, dtype = complex);
    for Ti in range(n_loc_dof):
        
        # Caroli expression
        Ts[Ti] = 4*np.imag(SigmaR[Ti])*G[Ti-n_loc_dof,qi]*np.imag(SigmaL[Ti])*G.conj()[qi,Ti-n_loc_dof];
        assert( abs(np.imag(Ts[Ti])) <= 1e-8);

    return tuple(Ts);


if __name__ == "__main__": # test code

    import matplotlib.pyplot as plt

    # top level
    plt.style.use('seaborn-dark-palette');
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;

    # Siam inputs
    tl = 1.0;
    Vb = 1.5;
    Vg = 10;
    a = 1.0; # lattice spacing, defs length scale

    # scattering off an up spin
    J = -tl*tl/Vg
    upimp = np.array([[0.1,0.0],[0.0,0.0]]);
    h=[];
    for i in range(3):
        if(i==1):
            h.append(upimp);
        else:
            h.append(np.zeros_like(upimp));
    h = np.array(h);
    print(h);

    # test at max verbosity
    if True:
        myT = Tcoef(h, tl,1.0-2.0, verbose = verbose);
        print("**********",myT);



    
    


    








