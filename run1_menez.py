'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a spin-1/2 impurity

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# tight binding params
tl = 1.0;
th = 1.0;
Delta = 0.0; # zeeman splitting on imp

if False: # sigma dot S

    fig, ax = plt.subplots();
    for Jeff in [0.1,0.2,0.4]:

        # 2nd qu'd operator for S dot s
        h1e = np.zeros((4,4))
        g2e = wfm.utils.h_kondo_2e(Jeff, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

        # leads and zeeman splitting
        hzeeman = np.array([[Delta, 0, 0, 0],
                        [0,0, 0, 0],
                        [0, 0, Delta, 0],
                        [0, 0, 0, 0]]); # zeeman splitting
        hLL = np.copy(hzeeman);
        hRL = np.copy(hzeeman);
        hSR += hzeeman; # zeeman splitting is everywhere

        # source = up electron, down impurity
        source = np.zeros(np.shape(hSR)[0]);
        source[1] = 1;

        # package together hamiltonian blocks
        hblocks = np.array([hLL, hSR, hRL]);
        tblocks = np.array([-th*np.eye(*np.shape(hSR)),-th*np.eye(*np.shape(hSR))]);
        if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

        # sweep over range of energies
        # def range
        Emin, Emax = -1.999*tl, -1.8*tl
        numE = 30;
        Evals = np.linspace(Emin, Emax, numE, dtype = complex);
        Tvals = [];
        for E in Evals:
            Tvals.append(wfm.kernel(hblocks, tblocks, tl, E, source));

        # plot Tvals vs E
        Tvals = np.array(Tvals);
        #ax.scatter(Evals + 2*tl,Tvals[:,1], marker = 's',label = "$T$");
        sc_pc = ax.scatter(Evals + 2*tl,Tvals[:,2], marker = 's');

        # menezes prediction in the continuous case
        # all the definitions, vectorized funcs of E
        kappa = np.lib.scimath.sqrt(Evals);
        jprime = Jeff/(4*kappa);
        l1, = ax.plot(np.linspace(Emin,Emax,100)+2*tl, Jeff*Jeff/(16*(np.linspace(Emin,Emax,100)+2*tl)),
         label = "$J/t$ = "+str(Jeff));

    # format and plot
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    ax.set_xlabel("$E+2t_l $");
    ax.set_ylabel("$T_{down}$");
    ax.set_xlim(0,0.2);
    ax.set_ylim(0,0.2);
    if(Delta): ax.legend(title = "$J = $"+str(Jeff)+"\n$\Delta$ = "+str(Delta));
    else: ax.legend();
    plt.show();



if False: # 2 site hubbard that downfolds into J sigma dot S

    # add'l physical terms
    Vg = 10;
    U = 100.0;
    Jeff = 2*th*th*U/((U-Vg)*Vg);

    # SR physics
    hSR = np.array([[0,0,-th,th,0,0], # up down, -
                    [0,Vg,0, 0,0,0], # up, up
                   [-th,0,Vg, 0,0,-th], # up, down (source)
                    [th,0, 0, Vg,0, th], # down, up
                    [0, 0, 0,  0,Vg,0],    # down, down
                    [0,0,-th,th,0,U+2*Vg]]); # -, up down

    # leads also have gate voltage
    hLL = Vg*np.eye(*np.shape(hSR));
    hLL[0,0] = 0;
    hRL = Vg*np.eye(*np.shape(hSR));
    hRL[0,0] = 0;

    # shift by gate voltage so source is at zero
    hLL += -Vg*np.eye(*np.shape(hLL));
    hSR += -Vg*np.eye(*np.shape(hSR));
    hRL += -Vg*np.eye(*np.shape(hRL));

    # package together hamiltonian blocks
    hblocks = np.array([hLL, hSR, hRL]);
    tblocks = np.array([-th*np.eye(*np.shape(hSR)),-th*np.eye(*np.shape(hSR))]);
    if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

    # source = up electron, down impurity
    source = np.zeros(np.shape(hSR)[0]);
    source[2] = 1;
    
    # sweep over range of energies
    # def range
    Emin, Emax = -1.999*tl, -1.8*tl
    numE = 30;
    Evals = np.linspace(Emin, Emax, numE, dtype = complex);
    Tvals = [];
    for E in Evals:
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E, source));

    # plot Tvals vs E
    fig, ax = plt.subplots();
    Tvals = np.array(Tvals);
    ax.scatter(Evals + 2*tl,Tvals[:,3], marker = 's');

    # menezes prediction in the continuous case
    # all the definitions, vectorized funcs of E
    kappa = np.lib.scimath.sqrt(Evals);
    jprime = Jeff/(4*kappa);
    ax.plot(np.linspace(Emin,Emax,100)+2*tl, Jeff*Jeff/(16*(np.linspace(Emin,Emax,100)+2*tl)),
     label = "$J/t$ = "+str(int(100*Jeff)/100)); 

    # format and plot
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    ax.set_xlabel("$E+2t_l $");
    ax.set_ylabel("$T_{down}$");
    ax.set_xlim(0,0.2);
    ax.set_ylim(0,0.2);
    if(Delta): ax.legend(title = "$J = $"+str(Jeff)+"\n$\Delta$ = "+str(Delta));
    else: ax.legend();
    plt.show();




if True: # onsite U 

    # add'l physical terms
    Vg = -0.1;
    U = 0.2;
    Jeff = 2*(Vg+U);

    # imp ham
    hSR = np.array([[ Vg + U, Vg+U],
                     [Vg+U, Vg + U]]);

    # hybridization to imp
    V_hyb = -th*np.array([[1,0],
                          [0,1]]);   

    # source = up electron, down impurity
    source = np.zeros(np.shape(hSR)[0]);
    source[0] = 1;

    # package together hamiltonian blocks
    hblocks = np.array([np.zeros_like(hSR), hSR, np.zeros_like(hSR)]);
    tblocks = np.array([np.copy(V_hyb), np.copy(V_hyb)])
    if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

    # sweep over range of energies
    # def range
    Emin, Emax = -1.999*tl, -1.5*tl
    numE = 99;
    Evals = np.linspace(Emin, Emax, numE, dtype = complex);
    Tvals = [];
    for E in Evals:
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, E, source));
    Tvals = np.array(Tvals);
    
    # plot Tvals vs E
    fig, ax = plt.subplots();
    ax.plot(Evals + 2*tl,Tvals[:,0], label = "up");
    ax.plot(Evals + 2*tl,Tvals[:,1], label = "down");

    # menezes prediction in the continuous case
    # all the definitions, vectorized funcs of E
    kappa = np.lib.scimath.sqrt(Evals);
    jprime = Jeff/(4*kappa);
    ax.plot(Evals+2*tl, Jeff*Jeff/(16*(Evals+2*tl)), label = "$J/t$ = "+str(Jeff));

    # format and show
    ax.set_xlim(Emin+2*tl, Emax+2*tl);
    plt.legend();
    plt.show();








