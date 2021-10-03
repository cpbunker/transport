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


def Green(h, t, E, verbose = 0):
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
    return G;


def Tcoef(h, t, E, qi, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    h, block diag hamiltonian matrices
    t, off diag hopping matrix
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
    G = Green(h, t, E, verbose = verbose);
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
        tl_arr.append(t*np.eye(*np.shape(Se_dot_S1)) );
    tl_arr = np.array(tl_arr);

    return h_cicc, tl_arr;


if __name__ == "__main__": # test code

    import matplotlib.pyplot as plt

    # top level
    plt.style.use('seaborn-dark-palette');
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;

    # siam inputs
    Nsites = 3; # total num SR sites
    i1, i2 = 1, Nsites; # locations of impurities
    tl = 1.0;
    Vg = 10;
    alat = 1.0; # lattice spacing
    Jeff = 2*tl*tl/Vg; # eff heisenberg # double check

    # souce specifies which basis vector is boundary condition
    sourcei = 3; # incident up, imps + down, down

    # make diag and off diag block matrices
    h_cicc, tl_arr = h_cicc_eff(Jeff, tl, i1, i2, Nsites);

    # E, k mesh
    alat = 1.0; # should always cancel for E and kx0
    kmin, kmax = 0.0, np.pi/(10*alat); # should also be near bottom of band
    Npts = 20;
    kvals = np.linspace(kmin, kmax, Npts, dtype = complex); # k mesh
    kx0vals = kvals*(i2-i1)*alat; # k*x0 mesh
    Evals = E_disp(kvals, alat, tl); # E mesh
    Tvals = []

    # get T(E) data
    for Ei in range(len(Evals) ):
        Tvals.append(list(Tcoef(h_cicc, tl_arr, Evals[Ei], sourcei)) );
    Tvals = np.array(Tvals);

    # check data
    if True:
        # total Sz should be conserved
        Sztot_by_sourcei = np.array([1.5,0.5,0.5,-0.5,0.5,-0.5,-0.5,-1.5]);
        Sztot = Sztot_by_sourcei[sourcei]; # total Sz spec'd by source
        for si in range(np.shape(h_cicc[0])[0]): # iter over all transmitted total Sz states
            if( Sztot_by_sourcei[si] != Sztot): # ie a transmitted state that doesn't conserve Sz
                for Ei in range(len(Tvals)):
                    assert(abs(Tvals[Ei,si]) <= 1e-8 ); # should be zero

    # plot total T at each E, k, kx0
    fig, axes = plt.subplots(3);
    Ttotals = np.sum(Tvals, axis = 1);
    axes[0].scatter(Evals+2*tl, Ttotals, marker = 's');
    axes[1].scatter(kvals, Ttotals, marker = 's');
    axes[2].scatter(kx0vals, Ttotals, marker = 's');
    #axes[0].plot(Evals+2*tl, (Evals+2*tl)/Jeff );
    #axes[0].plot(Evals+2*tl, Jeff*alat/np.pi *np.sqrt(2/((Evals+2*tl)*2*alat*alat*tl) ) );

    # format and show
    axes[0].set_xlabel("$E + 2t_l$");
    axes[1].set_xlabel("$k$");
    axes[2].set_xlabel("$kx_{0}$");
    axes[0].set_title("Incident up electron");
    axes[0].set_ylabel("$T$");
    for ax in axes:
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();



    
    


    








