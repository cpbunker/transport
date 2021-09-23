'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
ASU formalims: even sites are spin up, odd sites are spin down
'''

import numpy as np

import sys

##################################################################################
####

def Hmat(h, t, verbose = 0):
    '''
    Make the hamiltonian H for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    h, 1d arr, length N+2, onsite energy at each site
    t, float, hopping
    '''

    # unpack
    N = 2*(len(h) - 2); # num scattering region sites
    H = np.zeros((N+4, N+4), dtype = complex); # total sites by total sites

    # construct matrix
    for i in range(0,N+4,2): # iter over up orbs only
        for j in range(0,N+4,2):
            if( i == j): # on site energy
                sitei = int(i/2); # location of the local 2by2 site matrix in h
                H[i,j] += h[sitei][0,0]; # up spin
                H[i+1,j+1] += h[sitei][1,1]; # down spin
                H[i,j+1] += h[sitei][0,1]; # up to down spin coupling
                H[i+1,j] += h[sitei][1,0]; # down spin
            elif( (i==j+2) or (i+2==j) ): # hopping
                H[i, j] += -t;
                H[i+1,j+1] += -t;

    if verbose > 3: print("\nH = \n",np.real(H));
    return H; 


def Hprime(h, t, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    h, 1d arr, length N+2, on site energies
    t, float, hopping
    '''

    # unpack
    N = 2*(len(h) - 2); # num scattering region sites

    # self energies at LL
    SigmaL = []; # fill with up, down
    for V in [h[0][0,0], h[0][1,1]]: # up, down potential at LL
        lamL = (E-V)/(-2*t);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL.append(-t/LambdaLminus);

    # self energies at LL
    SigmaR = []; # fill with up, down
    for V in [h[-1][0,0], h[-1][1,1]]: # up, down potential at RL      
        lamR = (E-V)/(-2*t);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR.append(-t*LambdaRplus);
              
    # add self energies to hamiltonian
    Hp = Hmat(h, t, verbose = verbose); # regular ham
    Hp[0,0] += SigmaL[0]; # incident up spin
    Hp[1,1] += SigmaL[1]; # incident down spin
    Hp[N+2,N+2] += SigmaR[0]; # transmitted up spin
    Hp[N+3,N+3] += SigmaR[1]; # transmitted down spin
    if verbose: print("\nSelf energies = ",SigmaL, SigmaR);
    
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
    N = 2*(len(h) - 2); # num scattering region sites

    # get green's function matrix
    Hp = Hprime(h, t, E, verbose = verbose);
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    if verbose > 3: print("\nG[N+2,0] = ",G[N+2,0],"\nG[N+3,0] = ",G[N+3,0]);
    return G;


def Tcoef(h, t, E, verbose = 0):
    '''
    coefficient for a transmitted up and down electron
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));
    assert( np.shape(h)[-1] == 2);

    # unpack
    N = 2*(len(h) - 2); # num scattering region sites

    # self energies at LL
    SigmaL = []; # fill with up, down
    for V in [h[0][0,0], h[0][1,1]]: # up, down potential at LL
        lamL = (E-V)/(-2*t);
        LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
        SigmaL.append(-t/LambdaLminus);

    # self energies at LL
    SigmaR = []; # fill with up, down
    for V in [h[-1][0,0], h[-1][1,1]]: # up, down potential at RL      
        lamR = (E-V)/(-2*t);
        LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
        SigmaR.append(-t*LambdaRplus);

    # green's function
    G = Green(h, t, E, verbose = verbose);

    Tup = 4*np.imag(SigmaR[0])*G[N+2,0]*np.imag(SigmaL[0])*G.conj()[0,N+2]; # Caroli expression
    Tdown = 4*np.imag(SigmaR[1])*G[N+3,0]*np.imag(SigmaL[1])*G.conj()[0,N+3];
    assert( abs(np.imag(Tup)) <= 1e-8);
    assert( abs(np.imag(Tdown)) <= 1e-8);
    return Tup, Tdown;


def Esweep(h, t, Emin, Emax, Npts, verbose = 0):
    '''
    Automate looking at T as a function of E
    '''

    # unpack
    Es = np.linspace(Emin, Emax, Npts, dtype = complex);
    Tups = np.zeros_like(Es);
    Tdowns = np.zeros_like(Es);

    # sweep
    for Ei in range(len(Es) ):

        Tups[Ei], Tdowns[Ei] = Tcoef(h, t, Es[Ei], verbose = verbose);

    return Es, Tups, Tdowns;


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



    
    


    








