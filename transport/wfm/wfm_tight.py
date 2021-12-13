'''
Christian Bunker
M^2QM at UF
September 2021

Wave function matching (WFM) in 1D
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
    N = len(h) - 2;
    H = np.zeros((N+2, N+2), dtype = complex);

    # construct matrix
    for i in range(N+2):
        for j in range(N+2):
            if( i == j): # diags include V
                H[i,j] += h[i];
            elif( (i==j+1) or (i+1==j) ): # just off main diag are finite diff terms
                H[i, j] += -t;

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
    N = len(h) - 2; # num scattering sites
    lamL = (E-h[0])/(-2*t);
    LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
    lamR = (E-h[-1])/(-2*t);
    LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector
    Hp = Hmat(h, t, verbose = verbose);

    # add self energies
    SigmaL = -t/LambdaLminus;
    SigmaR = -t*LambdaRplus;
    Hp[0,0] += SigmaL;
    Hp[N+1,N+1] += SigmaR;
    
    if verbose > 3: print("\nH' = \n",Hp);
    if verbose > 3: print("SigmaL, SigmaR = ",SigmaL, SigmaR);
    if verbose > 3: print("ka_R = ",np.imag(SigmaR)/t );
    if verbose > 3: print("KE_r = ",-2*t*np.cos(np.imag(SigmaR)/t) );
    return Hp;


def Green(h, t, E, verbose = 0):
    '''
    Greens function for system described by
    - potential V[i] at site i
    - lattice spacing a
    - incident mass m
    -incident energy E
    '''

    # check inputs
    assert( isinstance(h, np.ndarray));

    Hp = Hprime(h, t, E, verbose = verbose);
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    if verbose > 3: print("\nG = ",G[0,-1]*G.conj()[-1,0])
    return G;


def Tcoef(h, t, E, verbose = 0):

    # check inputs
    assert( isinstance(h, np.ndarray));

    # unpack
    lamL = (E-h[0])/(-2*t);
    LambdaLminus = lamL - np.lib.scimath.sqrt(lamL*lamL - 1); # incident
    lamR = (E-h[-1])/(-2*t);
    LambdaRplus = lamR + np.lib.scimath.sqrt(lamR*lamR - 1); # transmitted wavevector

    # green's function
    G = Green(h, t, E, verbose = verbose);

    # compute self energies
    SigmaL = -t/LambdaLminus;
    SigmaR = -t*LambdaRplus;

    T = 4*np.imag(SigmaR)*G[-1,0]*np.imag(SigmaL)*G.conj()[0,-1]; # Caroli expression
    assert( abs(np.imag(T)) <= 1e-8);
    return T;


def Esweep(h, t, Emin, Emax, Npts, verbose = 0):
    '''
    Automate looking at T as a function of E
    '''

    # unpack
    Es = np.linspace(Emin, Emax, Npts);
    Ts = np.zeros_like(Es);

    # sweep
    for Ei in range(len(Es) ):

        Ts[Ei] = Tcoef(h, t, Es[Ei], verbose = verbose);

    return Es, Ts;


if __name__ == "__main__": # test code

    import matplotlib.pyplot as plt

    # top level
    plt.style.use('seaborn-dark-palette');
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 4;

    # Siam inputs
    mytls = np.array([10.0, 1.0]);
    Vb = 1.5;
    Vg = 10;
    a = 1.0; # lattice spacing, defs length scale

    # plot at diff lattice spacings
    fig, ax = plt.subplots();
    for tl in mytls:

        #T as a function of energy
        # step potential
        h = np.array([0,0,Vb,Vb]);

        # test at max verbosity
        if True:
            Tcoef(h, tl,1.0-2.0, verbose = 5);
            print("**********");

        # sweep over energy and plot
        npts = 400
        Es, Ts = Esweep(h, tl,-20,50, npts);
        ax.scatter(Es+2*tl, Ts, label = "discrete, $t_l = "+str(tl)+"$", marker = 's');

    # plot results from different tl values together
    if True:

        # format
        ax.set_xlabel("$E+2t_l$");
        ax.set_ylabel("$T$");
        ax.set_ylim(0,1.1);
        ax.axvline(Vb, color = "black", linestyle = "dashed", label="$E = V_b$");
        #ax.axvline(2.0+Vb/2, color = "grey", linestyle = "dashed", label="");

        # prediction
        prediction = 4*np.sqrt(Es*(Es-Vb) )/np.power(np.sqrt(Es) + np.sqrt(Es-Vb),2)
        ax.plot(Es, prediction, label = "continuous" );

        # show
        ax.set_title("Scattering from a tight binding potential barrier");
        ax.legend();
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        plt.show();

    '''
    plt.scatter(Vbs, maxs);
    plt.plot(Vbs, 1-Vbs*Vbs/16);
    plt.show();
    '''

    
    


    








