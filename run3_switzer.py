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
plt.style.use("seaborn-dark-palette");
#np.set_printoptions(precision = 4, suppress = True);
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
#### plots in dimer regime N_SR = 2

if True: # fig 6 ie T vs rho J a

    # siam inputs
    tl = 1.0;
    th = 1.0;
    tp = 1.0;

    # eric inputs
    JK1, JK2 = 0.01, 0.01;
    for D1 in [JK1/100]: # -abs(JK1/2)*np.array(range(1,9)):
        D2 = D1;

        # lead eigenstates
        h1e_JK0, g2e_JK0 = wfm.utils.h_switzer(D1, D2, 0, 0, 0);
        hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets32);
        hSR_JK0 = wfm.utils.entangle(hSR_JK0, *pair);
        hSR_JK0 = np.diagflat(np.diagonal(hSR_JK0)); # force diagonal
        if(verbose): print("JK1 = JK2 = 0 hamiltonian\n",hSR_JK0);

        # physics of scattering region -> array of [H at octo, H at tetra]
        hblocks, tblocks = [], []; # on site and hopping blocks in the SR
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
            if(impi == 0):
                hblocks.append(np.copy(hSR_JK0)); # LL eigenstates
                tblocks.append(-th*np.eye(np.shape(h)[0])); # hop onto imp 1
                tblocks.append(-tp*np.eye(np.shape(h)[0])); # hop onto imp 2
            hblocks.append(h);

        # end loop over impi
        del h, impi, h1e, g2e; 

        # finish blocks
        tblocks.append(-th*np.eye(np.shape(hSR_JK0)[0])); # hop onto RL
        hblocks.append(np.copy(hSR_JK0)); # RL eigenstates
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);
        for blocki in range(len(hblocks)): # set mu_LL = 0 for source channel
            hblocks[blocki] -= hSR_JK0[sourcei,sourcei]*np.eye(np.shape(hblocks[blocki])[0])
        if (verbose):
            print("\nD1 = ",D1,", JK1 = ",JK1,", H[0,0] = ",D1+JK1/2,"H[2,2] = ",D1+JK1/2+D1 - JK1*3/2);
            print("\nhblocks:\n", hblocks, "\ntblocks:\n", tblocks); 

        # iter over rhoJ, getting T
        Tvals = [];
        rhoJvals = np.linspace(0.05,2.5,40);
        Erhovals = JK1*JK1/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
        for rhoJa in rhoJvals:

            # energy and K fixed by J, rhoJ
            E_rho = JK1*JK1/(rhoJa*rhoJa*np.pi*np.pi*tl); # fixed E that preserves rho_J_int
                                                    # this E is measured from bottom of band !!!
            k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
            if False:
                print("E, ka = ",E_rho, ka_rho);
                print("rhoJa = ", abs(JK1/np.pi)/np.sqrt(E_rho*tl));
            E_rho = E_rho - 2*tl; # measure from mu
            kx0 = 0*np.pi;

            # get T from this setup
            Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho, source));

        # plot
        fig, ax = plt.subplots();
        Tvals = np.array(Tvals);
        ax.plot(rhoJvals, Tvals[:,2], color = "y", label = "$|i\,>$");
        ax.plot(rhoJvals, Tvals[:,0], color = "indigo", label = "$|+>$");
        ax.legend(loc = "upper left", fontsize = "large");
        
        # inset
        rhoEvals = JK1*JK1/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
        axins = inset_axes(ax, width="50%", height="30%", loc = "upper right");
        axins.plot(rhoEvals,Tvals[:,0], color = "indigo");
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






# iter over params
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






