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

def kernel(h, tnn, tnnn, tl, E, Ajsigma, reflect = False, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    h, array, block hamiltonian matrices
    tnn, array, nearest neighbor block hopping matrices
    tnnn, array, next nearest neighbor block hopping matrices
    tl, float, hopping in leads, not necessarily same as hopping on/off SR
        or within SR which is defined by th matrices
    E, float, energy of the incident electron
    Ajsigma, incident particle amplitude at site 0 in spin channel j
    '''

    # check input types
    assert( isinstance(h, np.ndarray));
    assert( isinstance(tnn, np.ndarray));
    assert(len(tnn)+1 == len(h));
    assert( isinstance(tnnn, np.ndarray));
    assert(len(tnnn)+2 == len(h));
    
    # check that lead hams are diagonal
    for hi in [0, -1]: # LL, RL
        isdiag = h[hi] - np.diagflat(np.diagonal(h[hi])); # subtract off diag
        if( np.any(isdiag)): # True if there are nonzero off diag terms
            raise Exception("Not diagonal\n"+str(h[hi]))
    for i in range(len(Ajsigma)): # always set incident mu = 0
        if(Ajsigma[i] != 0):
            assert(h[0,i,i] == 0);

    # check incident amplitude
    assert( isinstance(Ajsigma, np.ndarray));
    assert( len(Ajsigma) == np.shape(h[0])[0] );
    sigma0 = -1; # incident spin channel
    for sigmai in range(len(Ajsigma)): # find incident spin channel and check that there is only one
        if(Ajsigma[sigmai] != 0):
            if(sigma0 != -1): # then there was already a nonzero element, bad
                raise(Exception("Ajsigma has too many nonzero elements:\n"+str(Ajsigma)));
            else: sigma0 = sigmai;
    assert(sigma0 != -1);

    # unpack
    N = len(h) - 2; # num scattering region sites
    n_loc_dof = np.shape(h[0])[0];

    # determine velocities in the left, right leads
    v_L, v_R = np.zeros_like(Ajsigma), np.zeros_like(Ajsigma)
    ka_L = np.arccos((E-np.diagonal(h[0]))/(-2*tl)); # vector_sigma
    ka_R = np.arccos((E-np.diagonal(h[-1]))/(-2*tl));
    v_L = 2*tl*np.sin(ka_L); # a/hbar defined as 1
    v_R = 2*tl*np.sin(ka_R);

    # green's function
    if(verbose): print("\nEnergy = ",np.real(E+2*tl)); # start printouts
    G = Green(h, tnn, tnnn, tl, E, verbose = verbose);

    # contract G with source to pick out matrix elements we need
    Avector = np.zeros(np.shape(G)[0], dtype = complex); # go from spin space to spin+site space
    for sigmai in range(n_loc_dof):
        Avector[sigmai] = Ajsigma[sigmai]; # fill from spin space
    G_0sigma0 = np.dot(G, Avector); # G contracted with incident amplitude
                                    # picks out matrix elements of incident
                                    # still has 1 free spatial, spin index for transmitted
    if(verbose): print(G_0sigma0);
    # compute reflection and transmission coeffs
    coefs = np.zeros(n_loc_dof, dtype = float); 
    for sigmai in range(n_loc_dof): # iter over spin dofs

        # T given in my manuscript as Eq 20
        T = G_0sigma0[-n_loc_dof+sigmai]*np.conj(G_0sigma0[-n_loc_dof+sigmai])*v_R[sigmai]*v_L[sigma0];
        
        # R given in my manuscript as Eq 21
        R = (complex(0,-1)*G_0sigma0[0+sigmai]*v_L[sigma0] - Ajsigma[sigmai])*np.conj(complex(0,-1)*G_0sigma0[0+sigmai]*v_L[sigma0] - Ajsigma[sigmai])*v_L[sigmai]/v_L[sigma0];   

        # benchmarking
        if(verbose > 1):
            print(" - sigmai = ",sigmai,", T = ",T,", R = ",R);
            myvar = complex(0,1)*G_0sigma0[0+sigmai]*v_L[sigmai] - Ajsigma[sigmai]
            print(myvar, myvar*np.conj(myvar));
        #if( abs(np.imag(T)) > 1e-10 ): raise(Exception("T = "+str(T)+" must be real"));
        #if( abs(np.imag(R)) > 1e-10 ): raise(Exception("R = "+str(R)+" must be real"));

        # return var
        if(reflect): # want R
            coefs[sigmai] = np.real(R);
        else: # want T
            coefs[sigmai] = np.real(T);

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
        for sigmai in range(len(ka_L)):
            print(" - sigmai = ",sigmai,", v_L = ", v_L[sigmai],"v_R = ",v_R[sigmai]);
        if False:
            print(" - sigmai = ",sigmai,", ka_L = ", ka_L[sigmai],", KE_L = ", 2*tl-2*tl*np.cos(ka_L[sigmai]),", Sigma_L = ", SigmaLs[sigmai]);
            print(" - sigmai = ",sigmai,", ka_R = ", ka_R[sigmai],", KE_R = ", 2*tl-2*tl*np.cos(ka_R[sigmai]),", Sigma_R = ", SigmaRs[sigmai]);              #

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
    #if(verbose): print(">>> H' = \n", Hp );
    #if(verbose): print(">>> EI - H' = \n", E*np.eye(np.shape(Hp)[0]) - Hp );
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    # of interest is the qith row which contracts with the source q
    return G;



##################################################################################
#### test code

if __name__ == "__main__":

    pass;





    
    


    








