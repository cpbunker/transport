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
analytical = True; # whether to compare to menezes' calc
reflect = False; # whether to get R or T

# tight binding params
tl = 1.0;
th = 1.0;
Delta = 0.05; # zeeman splitting on imp
Vb = 0.0; # barrier voltage in RL
Nx3 = 0; # distance between imp and barrier

if False: # sigma dot S

    fig, ax = plt.subplots();
    colors = ["darkblue","darkgreen","darkred"];
    Jeffs = [0.1,0.2,0.4];
    for Ji in range(len(Jeffs)):
        Jeff = Jeffs[Ji];

        # 2nd qu'd operator for S dot s
        h1e = np.zeros((4,4))
        g2e = wfm.utils.h_kondo_2e(Jeff, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

        # zeeman splitting
        hzeeman = np.array([[Delta, 0, 0, 0],
                        [0,0, 0, 0],
                        [0, 0, Delta, 0], # spin flip gains PE delta
                        [0, 0, 0, 0]]);
        hSR += hzeeman;

        # truncate to coupled channels
        hSR = hSR[1:3,1:3];
        hzeeman = hzeeman[1:3,1:3];

        # leads
        hLL = np.copy(hzeeman);
        hRL = np.copy(hzeeman)+Vb*np.eye(np.shape(hLL)[0]);

        # source = up electron, down impurity
        sourcei, flipi = 0,1
        source = np.zeros(np.shape(hSR)[0]);
        source[sourcei] = 1;

        # package together hamiltonian blocks
        hblocks = [hLL,hSR];
        for x3i in range(Nx3): hblocks.append(np.zeros_like(hRL)); # vary imp to barrier distance
        hblocks.append(hRL);
        hblocks = np.array(hblocks);

        # hopping
        tnn = [-th*np.eye(*np.shape(hSR)),-th*np.eye(*np.shape(hSR))]; # on and off imp
        for x3i in range(Nx3): tnn.append(-tl*np.eye(*np.shape(hSR))); # vary imp to barrier distance
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Jeff == 0.1): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

        # sweep over range of energies
        # def range
        Emin, Emax = -1.99*tl, -1.99*tl+0.2;
        numE = 30;
        Evals = np.linspace(Emin, Emax, numE, dtype = complex);
        Tvals, Rvals = [], [];
        for E in Evals:
            if(E in Evals[:3]): # verbose
                Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, verbose = verbose));
                Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));
            else:
                Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source));
                Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));
                
        # plot Tvals vs E
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.scatter(np.real(Evals + 2*tl),Tvals[:,flipi], color = colors[Ji], marker = "s");
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #ax.plot(np.real(Evals + 2*tl), totals-1, color="red", label = "total - 1");

        # menezes prediction in the continuous case
        if(analytical):
            ax.plot(np.linspace(-1.9999*tl, Emax, 100)+2*tl, Jeff*Jeff/(16*(np.linspace(-1.9999*tl,Emax,100)+2*tl)), color = colors[Ji], linewidth = 2);

    # format and plot
    ax.set_xlim(0,0.2);
    ax.set_xticks([0,0.1,0.2]);
    ax.set_xlabel("$E+2t$", fontsize = "x-large");
    ax.set_ylim(0,0.2);
    ax.set_yticks([0,0.1,0.2]);
    if(reflect): ax.set_ylabel("$R_{flip}$", fontsize = "x-large");
    else: ax.set_ylabel("$T_{flip}$", fontsize = "x-large");
    plt.show();
    raise(Exception);

if False: # benchmark R + T

    fig, ax = plt.subplots();
    for Jeff in [0.1]:

        # 2nd qu'd operator for S dot s
        h1e = np.zeros((4,4))
        g2e = wfm.utils.h_kondo_2e(Jeff, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

        # zeeman splitting
        hzeeman = np.array([[Delta, 0, 0, 0],
                        [0,0, 0, 0],
                        [0, 0, Delta, 0], # spin flip gains PE delta
                        [0, 0, 0, 0]]);
        hSR += hzeeman;

        # truncate to coupled channels
        hSR = hSR[1:3,1:3];
        hzeeman = hzeeman[1:3,1:3];

        # leads
        hLL = np.copy(hzeeman);
        hRL = np.copy(hzeeman)+Vb*np.eye(np.shape(hLL)[0]);

        # source = up electron, down impurity
        sourcei, flipi = 0,1
        source = np.zeros(np.shape(hSR)[0]);
        source[sourcei] = 1;

        # package together hamiltonian blocks
        hblocks = [hLL,hSR];
        for x3i in range(Nx3): hblocks.append(np.zeros_like(hRL)); # vary imp to barrier distance
        hblocks.append(hRL);
        hblocks = np.array(hblocks);

        # hopping
        tnn = [-th*np.eye(*np.shape(hSR)),-th*np.eye(*np.shape(hSR))]; # on and off imp
        for x3i in range(Nx3): tnn.append(-tl*np.eye(*np.shape(hSR))); # vary imp to barrier distance
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Jeff == 0.1): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

        # sweep over range of energies
        # def range
        Emin, Emax = -1.999*tl, -1.999*tl+0.2;
        numE = 30;
        Evals = np.linspace(Emin, Emax, numE, dtype = complex);
        Tvals, Rvals = [], [];
        for E in Evals:
            if(E in Evals[:3]): # verbose
                Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, verbose = verbose));
                Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));
            else:
                Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source));
                Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));
                
        # plot Tvals vs E
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.plot(np.real(Evals + 2*tl),Tvals[:,sourcei], label = "source T", color = "black");
        ax.scatter(np.real(Evals + 2*tl),Tvals[:,flipi], label = "flip T", color = "black", marker = "s");
        ax.plot(np.real(Evals + 2*tl),Rvals[:,sourcei], label = "source R", color = "darkblue",);
        ax.plot(np.real(Evals + 2*tl),Rvals[:,flipi], label = "flip R", color = "darkgreen");
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        ax.plot(np.real(Evals + 2*tl), totals-1, color="red", label = "total - 1");

        # menezes prediction in the continuous case
        if(analytical):
            ax.plot(np.linspace(-1.9999*tl, Emax, 100)+2*tl, Jeff*Jeff/(16*(np.linspace(-1.9999*tl,Emax,100)+2*tl)));

    # format and plot
    #ax.set_xlim(0,0.2);
    #ax.set_xticks([0,0.1,0.2]);
    ax.set_xlabel("$E+2t$", fontsize = "x-large");
    #ax.set_ylim(0,0.2);
    #ax.set_yticks([0,0.1,0.2]);
    if(reflect): ax.set_ylabel("$R_{flip}$", fontsize = "x-large");
    else: ax.set_ylabel("$T_{flip}$", fontsize = "x-large");
    plt.legend();
    plt.show();
    raise(Exception);


if True: # full 2 site hubbard treatment (downfolds to S dot S)

    fig, ax = plt.subplots();
    for Vg in [17,9,4.8]:

        # parameters
        U = 100.0;
        Jeff = 2*th*th*U/((U-Vg)*Vg); # better for U >> Vg
        print("Jeff = ",Jeff);

        # SR physics: site 1 is in chain, site 2 is imp with large U
        hSR = np.array([[0,-th,th,0], # up down, -
                       [-th,Vg, 0,-th], # up, down (source)
                        [th, 0, Vg, th], # down, up (flip)
                        [0,-th,th,U+2*Vg]]); # -, up down

        # source = up electron, down impurity
        source = np.zeros(np.shape(hSR)[0]);
        sourcei, flipi = 1,2;
        source[sourcei] = 1;

        # lead physics
        # leads also have gate voltage except 0th channel
        hLL = Vg*np.eye(*np.shape(hSR));
        hLL[0,0] = 0;
        hRL = Vg*np.eye(*np.shape(hSR));
        hRL[0,0] = 0;

        # do the downfolding explicitly
        matA = np.array([[0, 0],[0,0]]);
        matB = np.array([[-th,-th],[th,th]]);
        matC = np.array([[-th,th],[-th,th]]);
        matD = np.array([[-Vg, 0],[0,U+Vg]]);
        mat_downfolded = matA - np.dot(matB, np.dot(np.linalg.inv(matD), matC))  
        print("mat_df = \n",mat_downfolded);
        Jeff = 2*abs(mat_downfolded[0,0]);
        print("Jeff = ",Jeff);
        mat_downfolded += np.eye(2)*Jeff/4
        print("mat_df = \n",mat_downfolded);

        # shift by gate voltage so source is at zero
        hLL += -Vg*np.eye(*np.shape(hLL));
        hSR += -Vg*np.eye(*np.shape(hSR));
        hRL += -Vg*np.eye(*np.shape(hRL));
        print("hLL = \n",hLL);
        print("hSR = \n",hSR);

        # package together hamiltonian blocks
        hblocks = np.array([hLL, hSR, hRL]);
        tnn_mat = -th*np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]);
        tnn = np.array([np.copy(tnn_mat), np.copy(tnn_mat)]);
        tnnn = np.zeros_like(tnn)[:-1];
        if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn, "\ntnnn:", tnnn); 
        
        # sweep over range of energies
        # def range
        Emin, Emax = -1.99*tl, -1.8*tl
        numE = 30;
        Evals = np.linspace(Emin, Emax, numE, dtype = complex);
        Tvals, Rvals = [], [];
        for E in Evals:
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));

        # plot Tvals vs E
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.scatter(Evals + 2*tl,Tvals[:,flipi], marker = 's');
        if False:
            for Ti in range(len(source)):
                ax.plot(Evals+2*tl, Tvals[:,Ti], linestyle = "dashed",label = Ti);

            # check that T+R = 1
            totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
            ax.plot(Evals + 2*tl, totals-1, color="red");

        # menezes prediction in the continuous case
        ax.plot(np.linspace(-1.9999*tl, Emax, 100)+2*tl, Jeff*Jeff/(16*(np.linspace(-1.9999*tl,Emax,100)+2*tl)), linewidth = 2);

    # format and plot
    ax.set_xlim(0,0.2);
    ax.set_xticks([0,0.1,0.2]);
    ax.set_xlabel("$E+2t$", fontsize = "x-large");
    ax.set_ylim(0,0.2);
    ax.set_yticks([0,0.1,0.2]);
    ax.set_ylabel("$T_{flip}$", fontsize = "x-large");
    plt.show();


if False: # onsite U 

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








