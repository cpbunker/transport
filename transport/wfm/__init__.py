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

def kernel(h, th, tl, E, qi, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    h, block diag hamiltonian matrices
    th, block off diag hopping matrix 
    tl, hopping in leads, not necessarily same as hopping on/off SR as def'd by t matrices
    E, energy of the incident electron
    qi, source vector (loc dof only)
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));
    assert( isinstance(th, np.ndarray));
    assert( isinstance(qi, np.ndarray));
    assert( len(qi) == np.shape(h[0])[0] );
    for hi in [0, -1]: # check that RL and LL hams are diagonal
        isdiag = h[hi] - np.diagflat(np.diagonal(h[hi]))
        assert(not np.any(isdiag));
    for i in range(len(qi)): # check source channel mu_LL = 0
        if(qi[i] != 0):
            pass;
            #assert(h[0,i,i] == 0);

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
    G = Green(h, th, tl, E, verbose = verbose);

    # coefs
    qivector = np.zeros(np.shape(G)[0]); # go from block space to full space
    for j in range(len(qi)):
        qivector[j] = qi[j]; # fill from block space
    Gqi = np.dot(G, qivector);
    if verbose > 3: print("\nG*q[0] = ",Gqi[0]);

    # transmission coefs
    Ts = np.zeros(n_loc_dof, dtype = float); # must be real
    for Ti in range(n_loc_dof):
        
        # Caroli expression 
        caroli = 4*np.imag(SigmaR[Ti])*Gqi[Ti-n_loc_dof]*np.imag(SigmaL[Ti])*Gqi.conj()[Ti-n_loc_dof];
        assert( abs(np.imag(caroli)) <= 1e-10); # screen for errors
        Ts[Ti] = np.real(caroli);

    return tuple(Ts);


def Hmat(h, t, verbose = 0):
    '''
    Make the hamiltonian H for N+2 x N+2 system
    where there are N sites in the scattering region (SR), 1 LL site, 1 RL site

    h, arr of on site blocks at each of the N+2 sites
    t, arr of N-1 hopping blocks between the N+2
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

                    elif(sitei == sitej+1): # input from t to lower diag
                        H[ovi, ovj] += t[sitej][loci, locj];

                    elif(sitei+1 == sitej): # input from t to upper diag
                        H[ovi, ovj] += t[sitei][loci, locj];                

    if verbose > 3: print("\nH = \n",H);
    return H; 


def Hprime(h, th, tl, E, verbose = 0):
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
    Hp = Hmat(h, th, verbose = verbose); # regular ham from SR on site blocks h and hop blocks th
    
    # self energies at LL
    # need a self energy for each LL boundary condition
    SigmaLs = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tl); 
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL = -tl/LambdaLminus; 
        Hp[Vi,Vi] += SigmaL;
        SigmaLs.append(SigmaL);
    del lamL, LambdaLminus, SigmaL

    # self energies at RL
    SigmaRs = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tl);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR = -tl*LambdaRplus;
        Hp[Vi-n_loc_dof,Vi-n_loc_dof] += SigmaR;
        SigmaRs.append(SigmaR);
    del lamR, LambdaRplus, SigmaR;
    
    if verbose > 3: print("\nH' = \n",Hp);
    if verbose > 3: print("SigmaL, SigmaR = ",SigmaLs, SigmaRs);
    return Hp;


def Green(h, th, tl, E, verbose = 0):
    '''
    Greens function for system described by
    - potential h[i] at site i of the scattering region (SR, sites i=1 to i=N)
    - hopping th on and off the SR
    - hopping tl in the leads (sites i=-\inf to 0, N+1 to +\inf)
    - incident energy E
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # get green's function matrix
    Hp = Hprime(h, th, tl, E, verbose = verbose);
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    # of interest is the qith row which contracts with the source q
    return G;

##################################################################################
#### wrappers

def Data(source, h_LL, V_hyb, h_SR, V_SR, h_RL, tl, lims, numpts = 21, retE = True, verbose = 0):
    '''
    Given a LL + SR + RL wave function matching system, defined by
    - blocks h_SR[i] on the ith site of the SR
    - hopping V_SR[i] btwn the ith and the i+1th site of the SR
    - hopping th onto the SR
    - hopping tl in the leads

    construct the transmission coefficients for an incident electron defined by source
    for all ka in kalims or energy in Elims

    Other args:
    - numpts, int, how many x axis vals
    - Energy, bool, tells whether to do vs ka or vs E
    '''

    # check inputs
    assert(np.shape(source)[0] == np.shape(h_SR[0])[0]);
    assert( np.shape(h_LL) == np.shape(h_SR[0]));
    assert(len(h_SR) == 1 + len(V_SR));

    # unpack
    N_SR = len(h_SR); # number of sites in the scattering region

    # package as block hams 
    hblocks = [h_LL] # sets mu in LL
    tblocks = [V_hyb]; # hopping from LL to SR
    hblocks.append(h_SR[0]); # site 1 in SR
    for n in range(1,N_SR):
        hblocks.append(h_SR[n]); # site n in SR, n=2...n=N_SR \
        tblocks.append(V_SR[n-1]); # V_SR[n] gives hopping from nth to n+1th site
    hblocks.append(h_RL); # sets mu in RL
    tblocks.append(V_hyb); # hopping from SR to RL
    hblocks = np.array(hblocks);
    tblocks = np.array(tblocks);
    if(verbose):
        print(" - tl = ", tl);
        print(" - th = ", -tblocks[0,0,0]);
        print(" - V = ", -tblocks[1,0,0]);

    if retE: # iter over energy
        Evals = np.linspace(lims[0], lims[1], numpts, dtype = complex);
        Tvals = [];
        for Energy in Evals:
            Tvals.append(kernel(hblocks, tblocks, tl, Energy, source));
        Tvals = np.array(Tvals);
        return Evals, Tvals;

    else:
        kavals = np.linspace(lims[0], lims[1], numpts, dtype = complex);
        Tvals = [];
        for ka in kavals:
            Energy = -2*tl*np.cos(ka);
            Tvals.append(kernel(hblocks, tblocks, tl, Energy, source));           
        Tvals = np.array(Tvals);
        return kavals, Tvals;

##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








