'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 3:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools

import sys
import time

#### top level
#np.set_printoptions(precision = 4, suppress = True);
plt.style.use("seaborn-dark-palette");
verbose = 4;

#### setup

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
#dets32 = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]]; # total spin 3/2 subspace
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# tight binding params
tl = 0.0056; # lead hopping, in Hartree
th = 0.0056; # SR hybridization
tp = 0.0056;  # hopping between imps
#epsO = -0.5; # octahedral Co onsite energy
#epsT = -1.0; # tetrahedral Co onsite energy

# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209/Ha2meV; # convert to hartree
Jz = 0.124/Ha2meV;
DO = 0.674/Ha2meV;
DT = 0.370/Ha2meV;
An = 0.031/Ha2meV;

# initialize source vector
sourcei = 16; # |down, 3/2, 3/2 >
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(len(dets));
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
if(verbose): print("\nSource:\n"+source_str);

# initialize pair
pair = (1,4); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

# lead eigenstates
h1e_JK0, g2e_JK0 = wfm.utils.h_dimer_2q((Jx, Jx, Jz, DO, DT, An, 0, 0)); 
hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets52);
hSR_JK0 = wfm.utils.entangle(hSR_JK0, 0, 1);
print("JK = 0 hamiltonian\n",hSR_JK0);
hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states);
hSR_JK0 = wfm.utils.entangle(hSR_JK0, *pair);
hLL = np.diagflat(np.diagonal(hSR_JK0)); # force diagonal
hLL += (-1)*hLL[sourcei,sourcei]*np.eye(np.shape(hLL)[0]);
hLL = np.zeros_like(hLL);
hRL = np.copy(hLL);

#########################################################
#### plots in N_SR = 2 regime

if True: # fig 6 ie T vs rho J a

    # plot at diff JK
    fig, ax = plt.subplots();
    for JK in DO*np.array([1.5,2.5,3.65,4.5,5.5]):

        # 2 site SR
        hblocks, tblocks = [hLL], [-th*np.eye(np.shape(hLL)[0])]; # on site and hopping blocks in the SR
        for Coi in range(2):

            # define all physical params
            JKO, JKT = 0, 0;
            if Coi == 0: JKO = JK; # J S dot sigma is onsite only
            else: JKT = JK;
            params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
            
            # construct second quantized ham
            h1e, g2e = wfm.utils.h_dimer_2q(params); 

            # construct h_SR (determinant basis)
            h_SR = fci_mod.single_to_det(h1e, g2e, species, states);
            h_SR = wfm.utils.entangle(h_SR, *pair);
            if(verbose > 4):
                h_SR_sub = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                print("\nUnentangled hamiltonian\n", h_SR_sub);
                h_SR_sub = wfm.utils.entangle(h_SR_sub, 0, 1);
                print("\nEntangled hamiltonian\n", h_SR_sub);

            # hopping between sites
            V_SR = -tp*np.eye(np.shape(h_SR)[0])
            
            # add to blocks list
            hblocks.append(np.copy(h_SR));
            if(Coi == 1): tblocks.append(np.copy(V_SR));

        hblocks.append(hRL);
        tblocks.append(-th*np.eye(np.shape(hLL)[0]));
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);

        # iter over rhoJ, getting T
        Tvals = [];
        rhoJvals = np.linspace(0.01,5.5,49);
        Erhovals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
        #Elims = -1.9999*tl, -1.99*tl;
        #Erhovals = np.linspace(*Elims, 49) + 2*tl; # bottom of band
        #rhoJvals = np.pi/np.sqrt(tl*Erhovals);
        for rhoi in range(len(rhoJvals)):

            # energy
            rhoJa = rhoJvals[rhoi];
            E_rho = Erhovals[rhoi];
            k_rho = np.arccos((E_rho-2*tl)/(-2*tl));
            if verbose > 4:
                print("E, E - 2t, JK1, E/JK1 = ",E_rho/tl, E_rho/tl -2, JK, E_rho/JK);
                print("ka = ",k_rho);
                print("rhoJa = ", abs(JK/np.pi)/np.sqrt(E_rho*tl));

            # T (Energy from 0)
            Tvals.append(wfm.kernel(hblocks, tblocks, tl, E_rho -2*tl, source));
            
        # plot
        Tvals = np.array(Tvals);
        #ax.plot(rhoJvals, Tvals[:,sourcei], label = "$|i\,>$");
        ax.plot(rhoJvals, Tvals[:,pair[0]], label = int(100*JK/DO)/100);
        #ax.plot(rhoJvals, Tvals[:,pair[1]], label = "$|->$");

    # end sweep over JK
    # format and show
    ax.set_xlabel("$\\rho\,J a$");
    ax.set_ylabel("$T_+$");
    ax.set_xlim(min(rhoJvals),max(rhoJvals));
    ax.set_ylim(0,0.2);
    ax.minorticks_on();
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    ax.legend(title = "$J_K /D_O$ = ",loc = "upper right");
    plt.show();










            

if False: #plot vs energy
    
    # sweep over JK
    JKreson = (4/5)*(DO - (3/4)*Jx + (3/4)*Jz);
    for JK in np.linspace(DO,5*DO,9):

        # physics of scattering region -> array of [H at octo, H at tetra]
        hblocks, tblocks = [], []; # on site and hopping blocks in the SR
        for Coi in range(2):

            # define all physical params
            JKO, JKT = 0, 0;
            if Coi == 0: JKO = JK; # J S dot sigma is onsite only
            else: JKT = JK;
            params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
            
            # construct second quantized ham
            h1e, g2e = wfm.utils.h_dimer_2q(params); 

            # construct h_SR (determinant basis)
            h_SR = fci_mod.single_to_det(h1e, g2e, species, states);
            h_SR = wfm.utils.entangle(h_SR, *pair);
            if(verbose > 4):
                h_SR_sub = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                print("\nUnentangled hamiltonian\n", h_SR_sub);
                h_SR_sub = wfm.utils.entangle(h_SR_sub, 0, 1);
                print("\nEntangled hamiltonian\n", h_SR_sub);

            # hopping between sites
            V_SR = -tp*np.eye(np.shape(h_SR)[0])
            
            # add to blocks list
            hblocks.append(np.copy(h_SR));
            if(Coi == 1): tblocks.append(np.copy(V_SR));

        if(verbose): print("shape(hblocks) = ", np.shape(hblocks));

        # get data
        Elims = -1.9999*tl, -1.99*tl;
        kavals, Tvals = wfm.Data(source, hLL, -th*np.eye(np.shape(hLL)[0]),
                        hblocks, tblocks, hRL, tl, Elims, numpts = 99, retE = True);

        # first plot is just source and entangled pair
        fig, axes = plt.subplots(2, sharex = True);
        axes[0].set_title(JK/DO);
        axes[0].plot(kavals/tl,Tvals[:,sourcei], label = "$|i>$");
        axes[0].plot(kavals/tl,Tvals[:,pair[0]], label = "$|+>$");
        axes[0].legend(loc = 'upper right');
        
        # second plot is contamination
        contamination = np.zeros_like(Tvals[:,0]);
        for contami in range(len(dets)):
            if((contami != pair[0]) and (dets[contami][0] != dets[sourcei][0])):
                contamination += Tvals[:, contami];
        contamination = contamination/(contamination+Tvals[:,pair[0]]); 
        axes[1].plot(kavals/tl, contamination, color = "grey");
        
        # format
        for ax in axes:
            ax.minorticks_on();
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
            ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
        axes[0].set_xlabel("$ka/\pi$");
        axes[0].set_ylabel("$T$");
        axes[1].set_ylabel("Contamination");
        plt.show();




