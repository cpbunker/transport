'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 3:
Single electron incident on two identical spin 1 impurities, following Eric's paper
NB impurities have more complicated dynamics than in cicc case:
- on site spin anisotropy
- isotropic exchange interaction between them

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th 
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools

# top level
verbose = 3;

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4],[5,6,7]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.0_","0.0_","-1.0_","1.0_","0.0_","-1.0_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets32 = [[0,2,6],[0,3,5],[1,2,5]]; # total spin 3/2 subspace

# source
sourcei = 2; #| down, 1, 1 > = [1 2 5]
source = np.zeros(len(dets32));
source[sourcei] = 1;
source_str = "|";
for si in dets32[sourcei]: source_str += state_strs[si];
source_str += ">";
print("\nSource:\n"+source_str);

# entangled pair
pair = (0,1); #|up, 1, 0> = [0 2 6] and |up,0,1> = [0,3,5]
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets32[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

#########################################################
#### plots


if True: # fig 6

    # siam inputs
    tl = 1.0;
    th = 1.0;
    tp = 1.0;

    # eric params
    JK = 0.1;
    DeltaD = 1*JK/2/10;
    D1 = JK/2 + DeltaD/2;
    D2 = JK/2 - DeltaD/2;
    for Vg in [-1,0,1]:

        # entangled states in leads
        h1e_JK0, g2e_JK0 = wfm.utils.h_switzer(D1, D2, 0, 0, 0);
        hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets32);
        if(verbose): print("Hamiltonian, JK=0\n",hSR_JK0);
        hSR_JK0 = wfm.utils.entangle(hSR_JK0, *pair);
        if(verbose): print("Entangled Hamiltonian, JK=0\n",hSR_JK0);

        # transform into eigenbasis of leads
        _, Udiag = np.linalg.eigh(hSR_JK0); # eigvecs matrix takes us to eigenbasis
        hSR_JK0_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_JK0, Udiag)); # force diagonal
        if(abs(hSR_JK0_diag[0,1]) < 1e-10): # okay to force diagonal
            hSR_JK0_diag = np.diagflat(np.diagonal(hSR_JK0_diag));
        del hSR_JK0; # to make sure I always use _diag version
        if(verbose): print("Diagonal Hamiltonian, JK=0\n",hSR_JK0_diag);

        # physics of scattering region 
        hblocks = [np.copy(hSR_JK0_diag)]; # array of [LL, octa, tetra, RL]
        for impi in range(2):
            if(impi == 0): # on imp 1
                h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, JK, 0);
            else: # on imp 2
                h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, 0, JK);

            # convert to many body form
            h = fci_mod.single_to_det(h1e,g2e, species, states, dets_interest=dets32);

            # transform to eigenbasis
            h = wfm.utils.entangle(h, *pair);
            h_diag = np.dot( np.linalg.inv(Udiag), np.dot(h, Udiag));

            # gate voltage
            if(True):
                if(impi == 0): h_diag += (Vg/2)*np.eye(len(source));
                else: h_diag += (-Vg/2)*np.eye(len(source));

            # add to list
            hblocks.append(h_diag);

        # end loop over impi
        del h, h_diag, h1e, g2e; 

        # finish hblocks with RL
        hblocks.append(np.copy(hSR_JK0_diag)); 
        hblocks = np.array(hblocks);
        for blocki in range(len(hblocks)): # set mu_LL = 0 for source channel
            hblocks[blocki] -= hSR_JK0_diag[sourcei,sourcei]*np.eye(np.shape(hblocks[blocki])[0])
        if (verbose):
            print("\nD1 = ",D1,", JK = ",JK,);
            print("\nhblocks:\n", hblocks);

        # hopping
        tnn = []; # nearest neighbor hopping in SR
        tnn.append(-th*np.eye(len(source))); # hop onto imp 1
        tnn.append(-tp*np.eye(len(source))); # hop onto imp 2
        tnn.append(-th*np.eye(len(source))); # hop onto RL
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];

        # iter over rhoJ, getting T
        Tvals = [];
        rhoJvals = np.linspace(0.05,2.5,40);
        Erhovals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
        for rhoJa in rhoJvals:

            # energy and K fixed by J, rhoJ
            E_rho = JK*JK/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
            if False:
                print("E, ka = ",E_rho, ka_rho);
                print("rhoJa = ", abs(JK1/np.pi)/np.sqrt(E_rho*tl));
            E_rho = E_rho - 2*tl; # measure from mu
            kx0 = 0*np.pi;

            # get T from this setup
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E_rho, source));

        # plot
        fig, ax = plt.subplots();
        Tvals = np.array(Tvals);
        ax.plot(rhoJvals, Tvals[:,sourcei], color = "darkblue", linewidth = 2, label = "$|i\,>$");
        ax.plot(rhoJvals, Tvals[:,pair[0]], color = "darkgreen", linewidth = 2, label = "$|+>$");
        ax.plot(rhoJvals, Tvals[:,pair[1]], color = "darkred", linewidth = 2, label = "$|->$");
        ax.legend(loc = "upper left", fontsize = "large");
        
        # inset
        axins = inset_axes(ax, width="50%", height="50%", loc = "upper right");
        axins.plot(Erhovals,Tvals[:,0], color = "darkgreen", linewidth = 2);
        axins.set_xlabel("$E+2t_l$", fontsize = "x-large");
        xlim, ylim = (-0.000005,0.001), (0.0,1.0);
        axins.set_xlim(*xlim);
        axins.set_xticks([0,0.001]);
        axins.set_ylim(*ylim);
        axins.set_yticks([*ylim]);
        
        # format
        ax.set_xlabel("$\\rho\,J a$", fontsize = "x-large");
        ax.set_ylabel("$T$", fontsize = "x-large");
        ax.set_xlim(np.min(rhoJvals), np.max(rhoJvals));
        ax.set_ylim(0,1.05);
        plt.show();

    # done itering over JK
    raise(Exception);


if True: # entanglement control w/ Vg split

    # cicc inputs
    tl = 1.0;
    th = 1.0;
    tp = 1.0;
    rhoJa = 0.5;

    # eric params
    JK = 0.1;
    DeltaD = 0*JK/2/10;
    D1 = JK/2 + DeltaD/2;
    D2 = JK/2 - DeltaD/2;

    # entangled states in leads
    h1e_JK0, g2e_JK0 = wfm.utils.h_switzer(D1, D2, 0, 0, 0);
    hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets32);
    if(verbose): print("Hamiltonian, JK=0\n",np.real(hSR_JK0));
    hSR_JK0 = wfm.utils.entangle(hSR_JK0, *pair);
    if(verbose): print("Entangled Hamiltonian, JK=0\n",hSR_JK0);

    # transform into eigenbasis of leads
    _, Udiag = np.linalg.eigh(hSR_JK0); # eigvecs matrix takes us to eigenbasis
    hSR_JK0_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_JK0, Udiag)); # force diagonal
    if(abs(hSR_JK0_diag[0,1]) < 1e-10): # okay to force diagonal
        hSR_JK0_diag = np.diagflat(np.diagonal(hSR_JK0_diag));
    del hSR_JK0; # to make sure I always use _diag version
    if(verbose): print("Diagonal Hamiltonian, JK=0\n",hSR_JK0_diag);

    # plot arrays
    Vgvals = np.linspace(-2.0,2.0,15);
    Tvals = np.zeros((3, len(Vgvals)));

    # iter over Vg
    for Vgi in range(len(Vgvals)):

        # physics of scattering region 
        hblocks = [np.copy(hSR_JK0_diag)]; # array of [LL, octa, tetra, RL]
        for impi in range(2):
            if(impi == 0): # on imp 1
                h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, JK, 0);
            else: # on imp 2
                h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, 0, JK);

            # convert to many body form
            h = fci_mod.single_to_det(h1e,g2e, species, states, dets_interest=dets32);
            h = wfm.utils.entangle(h, *pair);
            h_diag = np.dot( np.linalg.inv(Udiag), np.dot(h, Udiag));
            
            # add to list
            hblocks.append(h_diag);

        # end loop over impi
        del h, h_diag, h1e, g2e;

        # hopping
        tnn = []; # nearest neighbor hopping in SR
        tnn.append(-th*np.eye(len(source))); # hop onto imp 1
        tnn.append(-tp*np.eye(len(source))); # hop onto imp 2
        tnn.append(-th*np.eye(len(source))); # hop onto RL
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];

        # finish hblocks with RL
        hblocks.append(np.copy(hSR_JK0_diag)); 
        hblocks = np.array(hblocks);
        for blocki in range(len(hblocks)): # set mu_LL = 0 for source channel
            hblocks[blocki] -= hSR_JK0_diag[sourcei,sourcei]*np.eye(len(source));

        # gate voltage
        hblocks[1] += (Vgvals[Vgi]/2)*np.eye(len(source));
        hblocks[2] += (-Vgvals[Vgi]/2)*np.eye(len(source));
        if (verbose and Vgi == 0):
            print("\nD1 = ",D1,", D2 = ",D2,", JK = ",JK);
            print("\nhblocks:\n", hblocks);

        # 2 channels of interest
        for pairi in range(3): 

            # energy and K fixed by J, rhoJa
            E_rho = JK*JK/(rhoJa*rhoJa*np.pi*np.pi*tl); # measured from bottom of band !!!
            E_rho = E_rho - 2*tl; # measure from mu

            # get T from this setup
            Tvals[pairi, Vgi] = wfm.kernel(hblocks, tnn, tnnn, tl, E_rho, source)[pairi];

    # plot
    fig, ax = plt.subplots();
    ax.plot(Vgvals, Tvals[sourcei], color = "darkblue", linewidth = 2, label = "$|i\,>$");
    ax.plot(Vgvals, Tvals[pair[0]], color = "darkgreen", linewidth = 2, label = "$|+>$");
    ax.plot(Vgvals, Tvals[pair[1]], color = "darkred", linewidth = 2, label = "$|->$");
    ax.legend(loc = "upper left", fontsize = "large");
    
    # format
    ax.set_xlabel("$V_g$", fontsize = "x-large");
    ax.set_ylabel("$T$", fontsize = "x-large");
    ax.set_xlim(np.min(Vgvals), np.max(Vgvals));
    ax.set_ylim(0,1.0);
    plt.show();

    # done itering over JK
    raise(Exception);


if False: # intro Vg on both imps -> plot against k'a

    # tight binding params
    tl = 1.0; # lead hopping
    th = 1.0; # on/off SR
    tp = 1.0; # in SR
    
    # eric inputs
    JK1, JK2 = 0.01, 0.01;
    for D1 in [JK1/100]: # -abs(JK1/2)*np.array(range(1,9)):
        D2 = D1;

        # cicc quantitites
        N_SR = 2;
        factor = 99;
        ka0 = np.pi/(N_SR - 1)/factor; # a' is length defined by hopping t' btwn imps
                                # ka' = ka'0 = ka0 when t' = t so a' = a
        E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                        # measured from bottom of the band!!
        rhoJa = (1/2)*(JK1+JK2)/(np.pi*np.sqrt(tl*E_rho));

        # diagnostic
        if(verbose):
            print("\nCiccarello inputs")
            print(" - E, JK1, E/JK1 = ",E_rho, JK1, E_rho/JK1);
            print(" - ka0 = ",ka0);
            print("- rho*J*a = ", rhoJa);

        # lead eigenstates
        h1e_JK0, g2e_JK0 = wfm.utils.h_switzer(D1, D2, 0, 0, 0);
        hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets32);
        hSR_JK0 = wfm.utils.entangle(hSR_JK0, *pair);
        hSR_JK0 = np.diagflat(np.diagonal(hSR_JK0)); # force diagonal
        if(verbose): print("JK1 = JK2 = 0 hamiltonian\n",hSR_JK0);

        # physics of scattering region 
        hblocks = [np.copy(hSR_JK0)]; # array of [LL, octa, tetra, RL]
        for impi in range(2):
            if impi == 0: # on imp 1
                h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, JK1, 0);
            else: # on imp 2
                h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, 0, JK2);

            # convert to many body form
            h = fci_mod.single_to_det(h1e,g2e, species, states, dets_interest=dets32);

            # entangle the me up states into eric's me, s12, m12> = up, 2, 1> state
            h = wfm.utils.entangle(h, *pair);

            # add to list
            hblocks.append(h);

        # end loop over impi
        del h, impi, h1e, g2e; 

        # hopping
        tnn = []; # nearest neighbor hopping in SR
        tnn.append(-th*np.eye(len(source))); # hop onto imp 1
        tnn.append(-tp*np.eye(len(source))); # hop onto imp 2
        tnn.append(-th*np.eye(len(source))); # hop onto RL
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];

        # finish hblocks with RL
        hblocks.append(np.copy(hSR_JK0)); 
        hblocks = np.array(hblocks);
        for blocki in range(len(hblocks)): # set mu_LL = 0 for source channel
            hblocks[blocki] -= hSR_JK0[sourcei,sourcei]*np.eye(np.shape(hblocks[blocki])[0])
        if (verbose):
            print("\nD1 = ",D1,", JK1 = ",JK1,", H[0,0] = ",D1+JK1/2,"H[2,2] = ",D1+JK1/2+D1 - JK1*3/2);
            print("\nhblocks:\n", hblocks, "\ntnn\n", tnn); 

        # get data
        kpalims = (0.0*ka0,5.1*ka0);
        kpavals = np.linspace(*kpalims, 99);
        Vgvals = -2*tl*np.cos(ka0) + 2*tl*np.cos(kpavals);
        Tvals = [];
        for Vg in Vgvals:
            
            # add gate voltage in SR
            for blocki in range(len(hblocks)): 
                if(blocki > 0 and blocki < N_SR + 1): # if in SR
                    hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                    
            # get data
            Energy = -2*tl*np.cos(ka0);
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source));


        # plot
        fig, ax = plt.subplots();
        Tvals = np.array(Tvals);
        ax.plot(kpavals*(N_SR - 1)/np.pi, Tvals[:,2], color = "y", label = "$|i\,>$");
        ax.plot(kpavals*(N_SR - 1)/np.pi, Tvals[:,0], color = "indigo", label = "$|+>$");
        ax.legend(loc = "upper left", fontsize = "large");
        
        # format
        ax.set_xlabel("$k'(N-1)a/np.pi$", fontsize = "x-large");
        ax.set_ylabel("$T$", fontsize = "x-large");
        ax.set_xlim(min(kpavals*(N_SR - 1)/np.pi),max(kpavals*(N_SR - 1)/np.pi));
        ax.set_ylim(0,1.0);
        plt.show();
        raise(Exception);


if False: # iter over params
    for D1 in [0.1,0.2,0.3]:
        D2 = D1;

        # iter over Kondo strength
        for JK1 in [D1/3,2*D1/3,D1]:
            JK2 = JK1;

            # lead eigenstates
            h1e_JK0, g2e_JK0 = wfm.utils.h_switzer(D1, D2, 0, 0, 0);
            hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets32);
            hSR_JK0 = wfm.utils.entangle(hSR_JK0, *pair);
            if(verbose): print("JK1 = JK2 = 0 hamiltonian\n",hSR_JK0);
            hSR_JK0 = np.diagflat(np.diagonal(hSR_JK0)); # force diagonal

            # physics of scattering region -> array of [H at octo, H at tetra]
            hblocks, tblocks = [], []; # on site and hopping blocks in the SR
            for impi in range(2):
                if impi == 0: # on imp 1
                    h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, JK1, 0);
                else:
                    h1e, g2e = wfm.utils.h_switzer(D1, D2, 0, 0, JK2);

                # convert to many body form
                h = fci_mod.single_to_det(h1e,g2e, species, states, dets_interest=dets32);

                # entangle the me up states into eric's me, s12, m12> = up, 2, 1> state
                h = wfm.utils.entangle(h, *pair);
                if(verbose>3): print("\nEntangled hamiltonian\n", h);

                # add to list
                if(impi == 0):
                    hblocks.append(np.copy(hSR_JK0)); # LL eigenstates
                    tblocks.append(-th*np.eye(np.shape(h)[0])); # hop onto imp 1
                    tblocks.append(-tp*np.eye(np.shape(h)[0])); # hop onto imp 2
                hblocks.append(h);
            del h, impi, h1e, g2e;

            # finish list
            tblocks.append(-th*np.eye(np.shape(hSR_JK0)[0])); # hop onto RL
            hblocks.append(np.copy(hSR_JK0)); # RL eigenstates
            hblocks = np.array(hblocks);
            tblocks = np.array(tblocks);
            for blocki in range(len(hblocks)): # set mu_LL = 0
                hblocks[blocki] -= hSR_JK0[sourcei,sourcei]*np.eye(np.shape(hblocks[blocki])[0])
            if verbose: print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

            # iter over E
            #Elims = (-2)*tl, (-1.9)*tl;
            #Evals = np.linspace(*Elims, 40);
            ka0 = np.pi*np.sqrt(tp/tl); # val of ka s.t. ka' = pi
            kalims = 0, np.pi;
            kavals = np.linspace(*kalims, 49);
            Evals = -2*tl*np.cos(kavals);
            Tvals = [];
            for Energy in Evals:
                Tvals.append(wfm.kernel(hblocks, tblocks, tl, Energy, source));
            Tvals = np.array(Tvals); 

            # plot
            fig, ax = plt.subplots();
            ax.scatter(kavals/np.pi, Tvals[:,sourcei], marker = 's', label = "$|i\,>$");
            ax.scatter(kavals/np.pi, Tvals[:,pair[0]], marker = 's', label = "$|+>$");
            ax.scatter(kavals/np.pi, Tvals[:,pair[1]], marker = 's', label = "$|->$");

            # format
            #ax.set_title("Transmission at resonance, $J_K = 2D/3$");
            ax.set_ylabel("$T$");
            ax.set_xlabel("$E + 2t_l$");
            ax.set_title(str(D1)+", "+str(JK1)+", "+str(np.sqrt(JK1*JK1+np.power(D1-3*JK1/2,2)/4)));
            #ax.set_ylim(0.0,1.05);
            plt.legend();
            ax.minorticks_on();
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
            ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
            plt.show();






