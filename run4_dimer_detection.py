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
for i in range(len(hSR_JK0)): # force diag
    for j in range(len(hSR_JK0)):
        if(i != j):
            if(abs(hSR_JK0_diag[i,j]) >= 0 and abs(hSR_JK0_diag[i,j]) < 1e-10):
                hSR_JK0_diag[i,j] = 0;
            else: raise(Exception("Not diagonal "+str(hSR_JK0_diag[i,j])));
            
#########################################################
#### detection

if True: 

    numpts = 9;
    phivals = np.linspace(0, 2*np.pi, numpts);
    thetavals = np.linspace(0, np.pi, numpts);
    Tvals = np.zeros((len(thetavals),len(phivals)));
    Rvals = np.zeros_like(Tvals);

    for thetai in range(len(thetavals)):
        for phii in range(len(phivals)):
            thetaval = thetavals[thetai];
            phival = phivals[phii];
            
            source = np.zeros(len(hSR_JK0));
            source[1] = np.cos(thetaval);
            source[2] = np.sin(thetaval)*np.exp(complex(0,phival));
        
            # 2 site SR
            fig, ax = plt.subplots();
            hblocks = [np.copy(hSR_JK0_diag)];
            for Coi in range(2): # iter over imps

                # define all physical params
                JKO, JKT = 0, 0;
                if Coi == 0: JKO = 5*DO; # J S dot sigma is onsite only
                else: JKT = 5*DO;
                params = Jx, Jx, Jz, DO, DT, An, JKO, JKT;
                h1e, g2e = wfm.utils.h_dimer_2q(params); # construct ham

                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);
                        
                # diagonalize lead states
                hSR_diag = np.dot(np.linalg.inv(Udiag), np.dot(hSR, Udiag));
                if(False):
                    if Coi == 0: hSR_diag += (DeltaK)*np.eye(len(hSR_JK0_diag)); print(DeltaK*Ha2meV*np.eye(len(hSR_JK0_diag)));
                    elif Coi == 1: hSR_diag += 0 #(-DeltaK/2)*np.eye(len(hSR_JK0_diag));
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
            rhoJa = 1;
            Energy = DO*DO/(rhoJa*rhoJa*np.pi*np.pi*tl) - 2*tl; # measured from bottom of band

            # T (Energy from 0)
            Tvals[thetai,phii] = np.sum(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, verbose = 0));
            Rvals[thetai, phii] = np.sum(wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, reflect = True));

    # end sweep over JK
    raise(Exception);

