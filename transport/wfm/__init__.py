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

def kernel(h, tnn, tnnn, tl, E, qj, reflect = False, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    h, array, block hamiltonian matrices
    tnn, array, nearest neighbor block hopping matrices
    tnnn, array, next nearest neighbor block hopping matrices
    tl, float, hopping in leads, not necessarily same as hopping on/off SR
        or within SR which is defined by th matrices
    E, float, energy of the incident electron
    qj, source vector at site 0 (on site dof only, e.g. spin)
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));
    assert( isinstance(tnn, np.ndarray));
    assert(len(tnn)+1 == len(h));
    assert( isinstance(tnnn, np.ndarray));
    assert(len(tnnn)+2 == len(h));
    assert( isinstance(qj, np.ndarray));
    assert( len(qj) == np.shape(h[0])[0] );
    
    # check that lead hams are diagonal
    for hi in [0, -1]: # LL, RL
        isdiag = h[hi] - np.diagflat(np.diagonal(h[hi])); # subtract off diag
        if( np.any(isdiag)): # True if there are nonzero off diag terms
            raise Exception("Not diagonal\n"+str(h[hi]))
    for i in range(len(qj)): # check source channel mu_LL = 0
        if(qj[i] != 0):
            pass;
            #assert(h[0,i,i] == 0);

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # self energies at LL
    # need a self energy for each LL boundary condition
    SigmaL = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        lamL = (E-h[0][Vi,Vi])/(-2*tl);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # reflected
        SigmaL.append( -tl/LambdaLminus);

    # self energies at RL
    SigmaR = [];
    for Vi in range(n_loc_dof): # iters over all bcs    
        lamR = (E-h[-1][Vi,Vi])/(-2*tl);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted
        SigmaR.append( -tl*LambdaRplus);

    # check that modes with given energy are allowed in some LL channels
    assert(np.any(np.imag(SigmaL)) );
    for sigmai in range(len(SigmaL)):
        if(abs(np.imag(SigmaL[sigmai])) > 1e-10 and abs(np.imag(SigmaR[sigmai])) > 1e-10 ):
            assert(np.sign(np.imag(SigmaL[sigmai])) == np.sign(np.imag(SigmaR[sigmai])));

    # green's function
    G = Green(h, tnn, tnnn, tl, E, verbose = verbose);

    # contract G with source to pick out matrix elements we need
    qjvector = np.zeros(np.shape(G)[0], dtype = complex); # go from block space to full space
    for j in range(len(qj)):
        qjvector[j] = qj[j]; # fill from block space
    Gqj = np.dot(G, qjvector);

    # compute reflection and transmission coeffs
    coefs = np.zeros(n_loc_dof, dtype = float); # force zero
    for sigmai in range(n_loc_dof): # on site degrees of freedom
        if(verbose > 3): print("\nEnergy, sigmai = ",E,sigmai);
        rcoef = (-2*complex(0,1)*np.imag(SigmaL[sigmai])*Gqj[sigmai] -qj[sigmai]); # zwierzycki Eq 17
        tcoef = 2*complex(0,1)*Gqj[sigmai - n_loc_dof]*np.lib.scimath.sqrt(np.imag(SigmaL[sigmai])*np.imag(SigmaR[sigmai])) # zwierzycki Eq 26
        
        # benchmarking
        if(verbose > 3):
            print("- Gqj = ",Gqj);
            print("- R = ",rcoef*np.conj(rcoef));
            print("- T = ",tcoef*np.conj(tcoef));
        assert( abs(np.imag(rcoef*np.conj(rcoef))) < 1e-10 ); # must be real
        assert( abs(np.imag(tcoef*np.conj(tcoef))) < 1e-10 ); # must be real

        # return var
        if(reflect): # want R
            coefs[sigmai] = np.real(rcoef*np.conj(rcoef));
        else: # want T
            coefs[sigmai] = np.real(tcoef*np.conj(tcoef));

    return coefs;


def Hmat(h, tnn, tnnn, verbose = 0):
    '''
    Make the hamiltonian H for N+2 x N+2 system
    where there are N sites in the scattering region (SR), 1 LL site, 1 RL site

    h, array, on site blocks at each of the N+2 sites of the system
    tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    '''

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
                        H[ovi, ovj] = h[sitei][loci, locj];

                    elif(sitei == sitej+1): # input from tnn to lower diag
                        H[ovi, ovj] = tnn[sitej][loci, locj];

                    elif(sitei+1 == sitej): # input from tnn to upper diag
                        H[ovi, ovj] = tnn[sitei][loci, locj];

                    elif(sitei == sitej+2): # input from tnnn to 2nd lower diag
                        H[ovi, ovj] = tnnn[sitej][loci, locj];

                    elif(sitei+2 == sitej): # input from tnnn to 2nd upper diag
                        H[ovi, ovj] = tnnn[sitei][loci, locj];

    if(False):
        print("\n>>> H construction\n");
        print("- shape(H_j) = ", np.shape(h[0]));
        print("- shape(tnn_j) = ", np.shape(tnn[0]));
        print("- shape(tnnn_j) = ", np.shape(tnnn[0]));
        print("- shape(H) = ",np.shape(H));
        print("- H = \n",np.real(H));
        print(np.real(H)[::8,::8])
        print(np.real(H)[16:24,16:24]);
        assert False;
    return H; 


def Hprime(h, tnn, tnnn, tl, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    h, array, on site blocks at each of the N+2 sites of the system
    tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    tl, float, hopping in leads, distinct from hopping within SR def'd by above arrays
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # add self energies to hamiltonian
    Hp = Hmat(h, tnn, tnnn, verbose = verbose); # H matrix from SR on site, hopping blocks
    
    # self energies at LL
    # need a self energy for each LL boundary condition
    SigmaLs = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[0][Vi,Vi];
        lamL = (E-V)/(-2*tl); 
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # reflected
        SigmaL = -tl/LambdaLminus; 
        Hp[Vi,Vi] += SigmaL;
        SigmaLs.append(SigmaL);
    del lamL, LambdaLminus, SigmaL

    # self energies at RL
    SigmaRs = [];
    for Vi in range(n_loc_dof): # iters over all bcs
        V = h[-1][Vi,Vi];     
        lamR = (E-V)/(-2*tl);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted
        SigmaR = -tl*LambdaRplus;
        Hp[Vi-n_loc_dof,Vi-n_loc_dof] += SigmaR;
        SigmaRs.append(SigmaR);
    del lamR, LambdaRplus, SigmaR;
    
    if verbose > 3:
        SigmaLs, SigmaRs = np.array(SigmaLs), np.array(SigmaRs);
        print("\n****\nE = ",E);
        ka_L = np.arccos((E-h[0,0,0])/(-2*tl));
        print("ka_L = ",ka_L);
        print("KE_L = ",-2*tl*np.cos(ka_L) );
        print("SigmaL = ",SigmaLs);
        ka_R = np.arccos((E-h[-1,0,0])/(-2*tl));
        print("ka_R = ",ka_R);
        print("KE_R = ",-2*tl*np.cos(ka_R) );
        print("SigmaR = ",SigmaRs);
    return Hp;


def Green(h, tnn, tnnn, tl, E, verbose = 0):
    '''
    Greens function for system described by
    h, array, on site blocks at each of the N+2 sites of the system
    tnn, array, nearest neighbor hopping btwn sites, N-1 blocks
    tnnn, array, next nearest neighbor hopping btwn sites, N-2 blocks
    tl, float, hopping in leads, distinct from hopping within SR def'd by above arrays
    E, float, incident energy
    '''

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # get green's function matrix
    Hp = Hprime(h, tnn, tnnn, tl, E, verbose = verbose);
    #if(verbose): print(">>> EI - H' = \n", Hp )
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    # of interest is the qith row which contracts with the source q
    return G;



##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








