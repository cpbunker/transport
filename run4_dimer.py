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
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# tight binding params
tl = 0.0056; # lead hopping, in Hartree
th = 0.0056; # SR hybridization
tp = 0.0056;  # hopping between imps

# Ab initio params, in meV:
Ha2meV = 27.211386*1000; # 1 hartree is 27 eV
Jx = 0.209/Ha2meV; # convert to hartree
Jz = 0.124/Ha2meV;
DO = 0.674/Ha2meV;
DT = 0.370/Ha2meV;
An = 0.031/Ha2meV;
JK = DT;

# simplify for now
Jx, Jz = 0, 0;
DO, DT, An = 0,0,0;

# entangle pair
pair = (0,1); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets52[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

# lead eigenstates (JKO = JKT = 0)
h1e_JK0, g2e_JK0 = wfm.utils.h_dimer_2q((Jx, Jx, Jz, DO, DT, An, 0, 0)); 
hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets52);
hSR_JK0 = wfm.utils.entangle(hSR_JK0, 0, 1);
print("JK = 0 hamiltonian\n",hSR_JK0); # |i> decoupled when A=0
leadEs, Udiag = np.linalg.eigh(hSR_JK0);
hSR_JK0_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_JK0, Udiag));
print("diag JK = 0 hamiltonian\n",hSR_JK0_diag); # Udiag -> lead eigenstate basis

#########################################################
#### detection for N_SR = 2 regime

# peak for psi^- state
# vary kx0 by varying Vgate
if True: 
    
    # tight binding params
    tl = 1.0; # norm convention, -> a = a0/sqrt(2) = 0.37 angstrom

    # cicc quantitites
    N_SR = 200;
    ka0 = np.pi/(N_SR - 1); # a' is length defined by hopping t' btwn imps
                            # ka' = ka'0 = ka0 when t' = t so a' = a
    E_rho = 2*tl-2*tl*np.cos(ka0); # energy of ka0 wavevector, which determines rhoJa
                                    # measured from bottom of the band!!
    rhoJa = JK/(np.pi*np.sqrt(tl*E_rho));

    # diagnostic
    if(verbose):
        print("\nCiccarello inputs")
        print(" - E, J, E/J = ",E_rho, JK, E_rho/JK);
        print(" - ka0 = ",ka0);
        print("- rho*J*a = ", rhoJa);
        
    # Psi^- boundary condition
    source = np.zeros(3);
    sourcei = 1;
    source[sourcei] = 1;
    spinstate = "psimin"

    # construct LL
    hLL = np.copy(hSR_JK0);
    hLL += (-1)*hLL[sourcei,sourcei]*np.eye(np.shape(hLL)[0]); # const shift to set mu_LL = 0
    hblocks, tblocks = [hLL], [-th*np.eye(np.shape(hLL)[0])];
    if(verbose):
        print("LL hamiltonian\n", hLL);

    # construct SR
    for Coi in range(1,N_SR+1): # add SR blocks as octo, tetra impurities

        # define all physical params
        JKO, JKT = 0, 0;
        if Coi == 1: JKO = JK; # J S dot sigma is onsite only
        elif Coi == N_SR: JKT = JK;     # so do JKO, then JKT
        params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
        
        # construct second quantized ham
        h1e, g2e = wfm.utils.h_dimer_2q(params); 

        # construct h_SR (determinant basis)
        h_SR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
        h_SR = wfm.utils.entangle(h_SR, *pair);
        hblocks.append(np.copy(h_SR));

        # hopping between impurities
        if(Coi > 1): tblocks.append(-tp*np.eye(np.shape(h_SR)[0]));

    # construct RL
    hRL = np.copy(hLL);
    hblocks.append(hRL);
    tblocks.append(-th*np.eye(np.shape(hLL)[0]));
    hblocks, tblocks = np.array(hblocks), np.array(tblocks);
    if(verbose):
        print("RL hamiltonian\n", hRL);

    # get data
    kalims = (0.0*ka0,2.1*ka0);
    kavals = np.linspace(*kalims, 299);
    Vgvals = -2*tl*np.cos(ka0) + 2*tl*np.cos(kavals);
    Tvals = [];
    for Vg in Vgvals:              
        for blocki in range(len(hblocks)): # add Vg in SR
            if(blocki > 0 and blocki < N_SR + 1): # ie if in SR
                hblocks[blocki] += Vg*np.eye(np.shape(hblocks[0])[0])
                
        # get data
        Energy = -2*tl*np.cos(ka0);
        Tvals.append(wfm.kernel(hblocks, tblocks, tl, Energy, source));
    Tvals = np.array(Tvals);

    # package into one array
    if(verbose): print("shape(Tvals) = ",np.shape(Tvals));
    info = np.zeros_like(kavals);
    info[0], info[1], info[2], info[3] = tl, JK, rhoJa, ka0; # save info we need
    data = [info, kavals*(N_SR-1)];
    for Ti in range(np.shape(Tvals)[1]):
        data.append(Tvals[:,Ti]); # data has columns of kaval, corresponding T vals
    # save data
    fname = "dat/dimer/"+spinstate+"/";
    fname +="Vg_rhoJa"+str(int(rhoJa))+".npy";
    np.save(fname,np.array(data));
    if verbose: print("Saved data to "+fname);
    raise(Exception);


#########################################################
####

# initialize source vector in down, 3/2, 3/2 state
sourcei = 16; # |down, 3/2, 3/2 >
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(len(dets));
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
if(verbose): print("\nSource:\n"+source_str);

if False: # fig 6 ie T vs rho J a

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
        ax.plot(rhoJvals, Tvals[:,pair[0]], label = str(JK/JKreson)+" "+str(JK/DO));
        #ax.plot(rhoJvals, Tvals[:,pair[1]], label = "$|->$");

    # end sweep over JK
    # format and show
    ax.set_xlabel("$\\rho\,J a$", fontsize = "x-large");
    ax.set_ylabel("$T_+$", fontsize = "x-large");
    ax.set_xlim(min(rhoJvals),max(rhoJvals));
    ax.set_ylim(0,0.2);
    ax.legend(title = "$J_K /D_O$ = ",loc = "upper right", fontsize = "large");
    plt.show();










            

if True: #plot vs energy
    
    # sweep over JK
    for JK in DO*np.array([1.0]):

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
        Evals, Tvals = wfm.Data(source, hLL, -th*np.eye(np.shape(hLL)[0]),
                        hblocks, tblocks, hRL, tl, Elims, numpts = 99, retE = True);

        # first plot is just source and entangled pair
        fig, ax = plt.subplots();
        #ax.plot(Evals/tl+2,Tvals[:,sourcei], label = "$|i>$");
        ax.plot(Evals/tl+2,Tvals[:,pair[0]], label = "$|+>$");
        
        # second plot is contamination
        contamination = np.zeros_like(Tvals[:,0]);
        for contami in range(len(dets)):
            if((contami != pair[0]) and (dets[contami][0] != dets[sourcei][0])):
                contamination += Tvals[:, contami];
        #contamination = contamination/(contamination+Tvals[:,pair[0]]); 
        ax.plot(Evals/tl+2, contamination, color = "black", label = "Contam.");
        
        # format
        ax.set_xlim(min(Evals/tl+2), max(Evals/tl+2));
        ax.set_ylim(0,0.1);
        ax.set_xlabel("$E+2t_l$", fontsize = "x-large");
        ax.set_ylabel("$T$", fontsize = "x-large");
        ax.legend(loc = 'upper right', fontsize = "large");
        plt.show();




