'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 4:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#### top level
#np.set_printoptions(precision = 4, suppress = True);
plt.style.use("seaborn-dark-palette");
verbose = 5;

#### setup

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
#dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
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
An = 0

if False:
    Jx = 0/Ha2meV;
    Jz = 0/Ha2meV;
    DO = 0/Ha2meV;
    DT = 0/Ha2meV;

# initialize source vector in down, 3/2, 3/2 state
sourcei = 2; # |down, 3/2, 3/2 >
assert(sourcei >= 0 and sourcei < len(dets52));
source = np.zeros(len(dets52));
source[sourcei] = 1;
source_str = "|";
for si in dets52[sourcei]: source_str += state_strs[si];
source_str += ">";
if(verbose): print("\nSource:\n"+source_str);

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
h1e_JK0, g2e_JK0 = wfm.utils.h_dimer_2q((Jx, Jx, Jz, DO, DT, An, 0,0)); 
hSR_JK0 = fci_mod.single_to_det(h1e_JK0, g2e_JK0, species, states, dets_interest=dets52);
print("\nNon-diagonal real JK = 0 hamiltonian, in meV\n",Ha2meV*np.real(hSR_JK0)); # |i> decoupled when A=0
leadEs, Udiag = np.linalg.eigh(hSR_JK0);
print("\n eigenstates:");
for coli in range(len(leadEs)): print(np.real(Udiag.T[coli]), Ha2meV*leadEs[coli]);
hSR_JK0_diag = np.dot( np.linalg.inv(Udiag), np.dot(hSR_JK0, Udiag));
print("\nDiagonal real JK = 0 hamiltonian, in meV\n",Ha2meV*np.real(hSR_JK0_diag)); # Udiag -> lead eigenstate basis
#print("\n",Ha2meV*np.real(wfm.utils.entangle(hSR_JK0_diag, *pair)));
for i in range(len(source)): # force diag
    for j in range(len(source)):
        if(i != j):
            if(abs(hSR_JK0_diag[i,j]) >= 0 and abs(hSR_JK0_diag[i,j]) < 1e-10):
                hSR_JK0_diag[i,j] = 0;
            else: raise(Exception("Not diagonal "+str(hSR_JK0_diag[i,j])));
#########################################################
#### generation

if True: # fig 6 ie T vs rho J a

    # plot at diff DeltaK
    DeltaKvals = DO*np.array([-5,0,5]);
    for DeltaK in DeltaKvals:
        # 2 site SR
        fig, ax = plt.subplots();
        hblocks = [np.copy(hSR_JK0_diag)];
        for Coi in range(2): # iter over imps

            # define all physical params
            JKO, JKT = 0, 0;
            if Coi == 0: JKO = 5*DO+DeltaK/2; # J S dot sigma is onsite only
            else: JKT = 5*DO-DeltaK/2;
            params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
            h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

            # construct h_SR (determinant basis)
            hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                
            # transform to eigenbasis
            hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR, Udiag));
            if(verbose):
                print("\nJKO, JKT = ",JKO*Ha2meV, JKT*Ha2meV);
                print(" - ham:\n", Ha2meV*np.real(hSR));
                print(" - transformed ham:\n", Ha2meV*np.real(hSR_diag));
            
            # add to blocks list
            hblocks.append(np.copy(hSR_diag));

        # finish hblocks
        hblocks.append(hSR_JK0_diag);
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);

        # hopping
        tnn = np.array([-th*np.eye(len(source)),-tp*np.eye(len(source)),-th*np.eye(len(source))]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over rhoJ, getting T
        Tvals, Rvals = [], [];
        rhoJvals = np.linspace(0.01,1.0,99);
        Erhovals = DO*DO/(rhoJvals*rhoJvals*np.pi*np.pi*tl); # measured from bottom of band
        for rhoi in range(len(rhoJvals)):

            # energy
            rhoJa = rhoJvals[rhoi];
            Energy = Erhovals[rhoi] - 2*tl; # measure from mu
            k_rho = np.arccos(Energy/(-2*tl));
            if(False):
                print("\nCiccarello inputs");
                print("E/t, JK/t, Erho/JK1 = ",Energy/tl + 2, JK/tl, (Energy + 2*tl)/JK);
                print("ka = ",k_rho);
                print("rhoJa = ", abs(JK/np.pi)/np.sqrt((Energy+2*tl)*tl));

            # T (Energy from 0)
            Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));
            
        # plot
        Tvals, Rvals = np.array(Tvals), np.array(Rvals);
        ax.plot(rhoJvals, Tvals[:,sourcei], label = "$|i\,>$", color = "darkblue", linewidth = 2);
        ax.plot(rhoJvals, Tvals[:,pair[0]], label = "$|+>$", color = "darkgreen", linestyle = "dashed", linewidth = 2);
        ax.plot(rhoJvals, Tvals[:,pair[1]], label = "$|->$", color = "darkred", linestyle = "dotted", linewidth = 2);
        ax.plot(rhoJvals, Tvals[:,0]+Tvals[:,1]+Tvals[:,2]+Rvals[:,0]+Rvals[:,1]+Rvals[:,2], color = "red")

        # inset
        if False:
            rhoEvals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
            axins = inset_axes(ax, width="50%", height="50%");
            axins.plot(rhoEvals,Tvals[:,pair[0]], color = "darkgreen", linestyle = "dashed", linewidth = 2); # + state
            axins.set_xlim(0,tl);
            axins.set_xticks([0,tl]);
            axins.set_xlabel("$E+2t_l$", fontsize = "x-large");
            axins.set_ylim(0,0.2);
            axins.set_yticks([0,0.2]);

        # format and show
        ax.set_xlim(min(rhoJvals),max(rhoJvals));
        ax.set_xticks([0,1]);
        ax.set_xlabel("$D_O/\pi \sqrt{tE}$", fontsize = "x-large");
        ax.set_ylim(0,0.2);
        ax.set_yticks([0,0.2]);
        ax.set_ylabel("$T$", fontsize = "x-large");
        #plt.legend();
        plt.show();

    # end sweep over JK
    raise(Exception);



#cos(theta) vs DeltaK only
if False:

    # dependent var containers
    numxvals = 15;
    DeltaKvals = DO*np.linspace(-400,100,numxvals);
    rhoJa = 1
    Erho = DO*DO/(rhoJa*rhoJa*np.pi*np.pi*tl); 

    # independent var containers
    Tvals = np.zeros((len(pair),len(DeltaKvals)));
    Rvals = np.zeros_like(Tvals);

    # |+'> channel and |-'> channel
    for pairi in range(len(pair)):

        # iter over JK
        for DKi in range(len(DeltaKvals)):
            DeltaK = DeltaKvals[DKi];

            # 2 site SR
            hblocks = [np.copy(hSR_JK0_diag)];
            for Coi in range(2): # iter over imps

                # define all physical params
                JKO, JKT = 0, 0;
                if Coi == 0: JKO = DO#+DeltaK/2; # J S dot sigma is onsite only
                else: JKT = DO#-DeltaK/2;
                params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);

                # diagonalize lead states
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR, Udiag));
                if(True):
                    print("\nJKO, JKT = ",JKO*Ha2meV, JKT*Ha2meV);
                    print(" - ham:\n", Ha2meV*np.real(hSR));
                    print(" - transformed ham:\n", Ha2meV*np.real(hSR_diag));

                if(True):
                    if Coi == 0: hSR_diag += (DeltaK/2)*np.eye(len(hSR_JK0_diag));
                    elif Coi == 1: hSR_diag += (-DeltaK/2)*np.eye(len(hSR_JK0_diag));
                
                # add to blocks list
                hblocks.append(np.copy(hSR_diag));

            # finish hblocks
            hblocks.append(hSR_JK0_diag);
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            if(verbose): print("\nhblocks = \n",hblocks);

            # hopping
            tnn = np.array([-th*np.eye(len(source)),-tp*np.eye(len(source)),-th*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # T and R for desired channel only
            Tvals[pairi, DKi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erho - 2*tl, source, verbose = 0)[pair[pairi]];
            #Rvals[pairi, DKi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erhovals[rhoi] - 2*tl, source, reflect = True)[pair[pairi]];    

    # put plotting arrays in right form
    DeltaKvals = DeltaKvals/DO; # convert
    thetavals = 2*np.arctan(Tvals[pair[0]]/Tvals[pair[1]])/np.pi;
    
    # plot (To do)
    fig, ax = plt.subplots();
    ax.plot(DeltaKvals, thetavals, color = "darkblue", linewidth = 2);
                      
    # format and show
    #ax.set_xlim(min(DeltaKvals),max(DeltaKvals));
    ax.set_xlabel("$\Delta_{K} /D_O$", fontsize = "x-large");
    #ax.set_ylim(0,1);
    #ax.set_yticks([0,1]);
    ax.set_ylabel("$\\theta/\pi$", fontsize = "x-large");
    plt.show();

    # end sweep over JK
    raise(Exception);


#cos(theta) vs energy and DeltaK 
if False:

    # dependent var containers
    numxvals = 15;
    DeltaKvals = DO*np.linspace(-100,100,numxvals);
    rhoJavals = np.linspace(0.01,4.0,numxvals);
    Erhovals = DO*DO/(rhoJavals*rhoJavals*np.pi*np.pi*tl); # measured from bottom of band

    # independent var containers
    Tvals = np.zeros((len(pair),len(DeltaKvals),len(rhoJavals)));
    Rvals = np.zeros_like(Tvals);

    # |+'> channel and |-'> channel
    for pairi in range(len(pair)):

        # iter over JK
        for DKi in range(len(DeltaKvals)):
            DeltaK = DeltaKvals[DKi];

            # 2 site SR
            hblocks = [np.copy(hSR_JK0_diag)];
            for Coi in range(2): # iter over imps

                # define all physical params
                JKO, JKT = 0, 0;
                if Coi == 0: JKO = DO#+DeltaK/2; # J S dot sigma is onsite only
                else: JKT = DO#-DeltaK/2;
                params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);

                # diagonalize lead states
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR, Udiag));
                if(True):
                    print("\nJKO, JKT = ",JKO*Ha2meV, JKT*Ha2meV);
                    print(" - ham:\n", Ha2meV*np.real(hSR));
                    print(" - transformed ham:\n", Ha2meV*np.real(hSR_diag));

                if(True):
                    if Coi == 0: hSR_diag += (DeltaK/2)*np.eye(len(hSR_JK0_diag));
                    elif Coi == 1: hSR_diag += (-DeltaK/2)*np.eye(len(hSR_JK0_diag));
                
                # add to blocks list
                hblocks.append(np.copy(hSR_diag));

            # finish hblocks
            hblocks.append(hSR_JK0_diag);
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            if(verbose): print("\nhblocks = \n",hblocks);

            # hopping
            tnn = np.array([-th*np.eye(len(source)),-tp*np.eye(len(source)),-th*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # iter over rhoJ (1/k)
            for rhoi in range(len(rhoJavals)):

                # T and R for desired channel only
                Tvals[pairi, DKi, rhoi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erhovals[rhoi] - 2*tl, source, verbose = 0)[pair[pairi]];
                #Rvals[pairi, DKi, rhoi] = wfm.kernel(hblocks, tnn, tnnn, tl, Erhovals[rhoi] - 2*tl, source, reflect = True)[pair[pairi]];    

    # put plotting arrays in right form
    DeltaKvals = DeltaKvals/DO; # convert
    DeltaKvals, rhoJavals = np.meshgrid(DeltaKvals, rhoJavals);
    thetavals = 2*np.arctan(Tvals[pair[0]].T/Tvals[pair[1]].T);
    
    # plot (To do)
    fig = plt.figure();   
    #ax.plot(DeltaKvals, thetavals/np.pi, color = "darkblue", linewidth = 2);
    ax3d = fig.add_subplot(projection = "3d");
    ax3d.plot_surface(rhoJavals, DeltaKvals, thetavals, cmap = cm.coolwarm);
                      
    # format and show
    #ax.set_xlim(min(DeltaKvals),max(DeltaKvals));
    ax3d.set_xlabel("$\Delta_{K} /D_O$", fontsize = "x-large");
    #ax.set_ylim(0,1);
    #ax.set_yticks([0,1]);
    ax3d.set_ylabel("$\\theta/\pi$", fontsize = "x-large");
    plt.show();

    # end sweep over JK
    raise(Exception);   



#########################################################
#### detection for N_SR = 2 regime

# peak for psi^- state
# vary kx0 by varying Vgate
if False: 
    
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




