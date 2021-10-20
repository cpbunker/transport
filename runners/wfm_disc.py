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

def Hmat(V, a, m, verbose = 0):
    '''
    Make the hamiltonian H for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    V, 1d arr, length N+2, potential at each site
    a, float, lattice spacing
    m, float, mass of incident particle
    '''

    # unpack
    N = len(V) - 2;
    H = np.zeros((N+2, N+2), dtype = complex);

    # construct matrix
    for i in range(N+2):
        for j in range(N+2):
            if( i == j): # diags include V
                H[i,j] += 1/(m*a*a) + V[i]; # V neg by convention
            elif( (i==j+1) or (i+1==j) ): # just off main diag are finite diff terms
                H[i, j] += -1/(2*m*a*a);

    if verbose > 3: print("\nH = \n",np.real(-H));
    return H;


def Hprime(V, a, m, E, verbose = 0):
    '''
    Make H' (hamiltonian + self energy) for N+2 x N+2 system
    where there are N sites in the scattering region (SR).

    V, 1d arr, length N+2, potential everywhere
        - V[0] = VL = left lead bias
        - V[0<i<N+1] = SR potential
        - V[N+1] = VR = right lead bias
    a, float, lattice spacing
    m, float, mass of incident particle
    E, float, incident energy
    '''

    # unpack
    N = len(V) - 2; # num scattering sites
    kL = np.lib.scimath.sqrt(2*m*(E-V[0]) ); # incident wavevector
    kR = np.lib.scimath.sqrt(2*m*(E-V[-1]) ); # transmitted wavevector
    if False: # compare k vals to tight binding scheme
        print("kL, kR = ", kL, kR);
        kLp = np.lib.scimath.arccos( (E-2-V[0])/(2*t) )/a;
        kRp = np.lib.scimath.arccos( (E-2-V[-1])/(2*t) )/a;
        print("kL', kR' = ", kLp, kRp );

    # make H matric
    Hp = Hmat(V, a, m, verbose = verbose);

    # add self energies
    Hp[0,0] += -np.exp(complex(0, kL*a))/(2*m*a*a); # LL self energy
    Hp[N+1,N+1] += -np.exp(complex(0, kR*a))/(2*m*a*a);
            
    if verbose > 3: print("\nH' = \n",Hp);
    return Hp;


def Green(V, a, m, E, verbose = 0):
    '''
    Greens function for system described by
    - potential V[i] at site i
    - lattice spacing a
    - incident mass m
    -incident energy E
    '''

    # check inputs
    assert( isinstance(V, np.ndarray));

    Hp = Hprime(V, a, m, E, verbose = verbose);
    G = np.linalg.inv( E*np.eye(np.shape(Hp)[0] ) - Hp );

    if verbose > 3: print("\nG = ",G[0,-1],G.conj()[-1,0])
    return G;


def Tcoef(V, a, m, E, verbose = 0):

    # check inputs
    assert( isinstance(V, np.ndarray));

    # unpack
    kL = np.lib.scimath.sqrt(2*m*(E-V[0]) );
    kR = np.lib.scimath.sqrt(2*m*(E-V[-1]) );

    # compute self energies
    SigmaL = -np.exp(complex(0, kL*a))/(2*m*a*a);
    SigmaR = -np.exp(complex(0, kR*a))/(2*m*a*a);

    # get green's function
    G = Green(V, a, m, E, verbose = verbose);

    T = 4*np.imag(SigmaR)*G[-1,0]*np.imag(SigmaL)*G.conj()[0,-1]; # Caroli expression
    assert( abs(np.imag(T)) <= 1e-8 );
    return T;


if __name__ == "__main__": # test code

    import matplotlib.pyplot as plt

    # top level
    plt.style.use('seaborn-dark-palette');
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 4;

    # Siam like inputs
    tl = 1.0; # hopping, defs energy scale
    Vb = 1.5;
    Vg = 10;
    myas = np.array([0.1,1.0]); # lattice spacing, defs length scale

    # WFM inouts
    J = tl*tl/Vg; # Schreiffer Wolf result 
    m = 1/(2*tl); # mass det'd by length and energy scales

    # plot at diff lattice spacings
    fig, ax = plt.subplots();
    for a in myas:

        # finite potential barrier
        # T as a function of energy
        V = np.array([0,0,Vb,Vb]);

        # test at max verbosity
        if True:
            Tcoef(V, a, m, 1.0, verbose = 5);
            print("**********");

        npts = 40
        Es = np.linspace(1.0, 5.0, npts);
        Ts = np.zeros_like(Es);
        Gs = np.zeros_like(Es);
        for Ei in range(len(Es)):
            Gval = Green(V, a, m, Es[Ei])[-1,0];
            Gs[Ei] = Gval*np.conj(Gval);
            Ts[Ei] = Tcoef(V, a, m, Es[Ei]);

        ax.scatter(Es, Ts, label = "discrete, $a = "+str(a)+"$", marker = 's');

    # plot
    if True:

        # format
        ax.set_xlabel("$E$");
        ax.set_ylabel("$T$");
        ax.set_ylim(0,1);
        ax.axvline(Vb, color = "black", linestyle = "dashed", label = "$E = V_b$");
        #ax.axvline(2.0+Vb/2, color = "grey", linestyle = "dashed", label="");

        # prediction
        prediction = 4*np.sqrt(Es*(Es-Vb) )/np.power(np.sqrt(Es) + np.sqrt(Es-Vb),2)
        ax.plot(Es, prediction, label = "continuous" );

        # show
        ax.set_title("Scattering from a discretized potential barrier");
        ax.legend();
        ax.minorticks_on();
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        plt.show();
    

    








