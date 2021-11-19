'''
Christian Bunker
M^2QM at UF
October 2021

Steady state transport of a single electron through a one dimensional wire
Part of the wire is scattering region, where the electron spin degrees of
freedom can interact with impurity spin degrees of freedom

Impurity hamiltonians calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm

import numpy as np
import matplotlib.pyplot as plt

import sys

#### funcs

def second_q_ham(JK1, JK2):
    '''
    Generate second quantized form of the Co dimer spin hamiltonian
    '''

    # basis size
    Nb = 2+8; # e + 4 m states each imp

    # first principles params, in meV
    Jx = 0.209;
    Jy = Jx; # for more flexibility
    Jz = 0.124;
    DO = 0.674;
    DT = 0.370;
    An = 0.031;
    #An = 0; # for simplicity

    # 1 particle terms
    h1e = np.zeros((Nb, Nb), dtype = complex);

    # octo spin anisitropy
    h1e[2,2] += DO*9/4;
    h1e[3,3] += DO*1/4;
    h1e[5,5] += DO*1/4;
    h1e[5,5] += DO*9/4;

    # tetra spin anisotropy
    h1e[6,6] += DT*9/4;
    h1e[7,7] += DT*1/4;
    h1e[8,8] += DT*1/4;
    h1e[9,9] += DT*9/4;

    # 2 particle terms
    g2e = np.zeros((Nb,Nb,Nb,Nb), dtype = complex);

    # isotropic terms
    xOcoefs = np.array([np.sqrt(3),np.sqrt(3),2,2,np.sqrt(3),np.sqrt(3)])/2;
    xOops = [(2,3),(3,2),(3,4),(4,3),(4,5),(5,4)];
    xTcoefs = np.copy(xOcoefs);
    xTops = [(6,7),(7,6),(7,8),(8,7),(8,9),(9,8)];
    g2e = fci_mod.terms_to_g2e(g2e, xOops, Jx*xOcoefs, xTops, xTcoefs);

    yOcoefs = complex(0,1)*np.array([-np.sqrt(3),np.sqrt(3),-2,2,-np.sqrt(3),np.sqrt(3)])/2;
    yOops = xOops;
    yTcoefs = np.copy(yOcoefs);
    yTops = xTops;
    g2e = fci_mod.terms_to_g2e(g2e, yOops, Jy*yOcoefs, yTops, yTcoefs);

    zOcoefs = np.array([3,1,-1,3])/2;
    zOops = [(2,2),(3,3),(4,4),(5,5)];
    zTcoefs = np.copy(zOcoefs);
    zTops = [(6,6),(7,7),(8,8),(9,9)];
    g2e = fci_mod.terms_to_g2e(g2e, zOops, Jz*zOcoefs, zTops, zTcoefs);

    # anisotropic terms
    g2e = fci_mod.terms_to_g2e(g2e, xOops, An*xOcoefs, zTops, zTcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, yOops,-An*yOcoefs, zTops, zTcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, zOops, -An*zOcoefs, xTops, xTcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, zOops, An*zOcoefs, yTops, yTcoefs);

    # Kondo terms
    xeops = [(0,1),(1,0)];
    xecoefs = np.array([1,1])/2;
    g2e = fci_mod.terms_to_g2e(g2e, xeops, JK1*xecoefs, xOops, xOcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, xeops, JK2*xecoefs, xTops, xTcoefs);
    yeops = [(0,1),(1,0)];
    yecoefs = complex(0,1)*np.array([-1,1])/2;
    g2e = fci_mod.terms_to_g2e(g2e, yeops, JK1*yecoefs, yOops, yOcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, yeops, JK2*yecoefs, yTops, yTcoefs);
    zeops = [(0,0),(1,1)];
    zecoefs = np.array([1,-1])/2;
    g2e = fci_mod.terms_to_g2e(g2e, zeops, JK1*zecoefs, zOops, zOcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, zeops, JK2*zecoefs, zTops, zTcoefs);

    return h1e, g2e;


def subspace(m):

    if(m==1/2):
        # pick out m=1/2 subspace
        picks = [[0,2,9],[0,3,8],[0,4,7],[0,5,6],[1,2,8],[1,3,7],[1,4,6]]; 
        pickis = [3,6,9,12,18,21,24]; 
        pickstrs = ["|up, 3/2, -3/2>","|up, 1/2, -1/2>","|up, -1/2, 1/2>","|up, -3/2, 3/2>","|down, 3/2, -1/2>","|down, 1/2, 1/2>","|down, -1/2, 3/2>"];

    elif(m==3/2):
        # pick out m=3/2 subspace
        picks = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]];
        pickis = [2, 5, 8, 17, 20,];
        pickstrs = ["|up, 3/2, -1/2>", "|up, 1/2, 1/2>", "|up, -1/2, 3/2>","|down, 3/2, 1/2>","|down, 1/2, 3/2>"];

    elif(m==5/2):
        picks = [[0,2,7],[0,3,6],[1,2,6]];
        pickis = [1, 4, 16];
        pickstrs = ["|up, 3/2, 1/2>", "|up, 1/2, 3/2>","|down, 3/2, 3/2>"];

    else:
        raise(ValueError);

    return picks, pickis, pickstrs;


def entangle(H,bi,bj):
    '''
    Perform a change of basis on a matrix such that basis vectors bi, bj become entangled (unentangled)
    '''

    # check inputs
    assert(bi < bj);
    assert(bj < max(np.shape(H)));

    # rotation matrix
    R = np.zeros_like(H);
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            if( i != bi and i != bj):
                if(i == j):
                    R[i,j] = 1; # identity
            elif( i == bi and j == bi):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j == bj):
                R[i,j] = -1/np.sqrt(2);
            elif( i == bi and j == bj):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j== bi):
                R[i,j] = 1/np.sqrt(2);

    return np.matmul(np.matmul(R.T,H),R);


#### test code
if True:
    h_SR = np.array([[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]])/4;
    print(h_SR);
    tl = 1.0;
    Energy = -2*tl + 0.1;
    ka = np.arccos(Energy/(-2*tl));
    source = np.array([0,1,0,0]);
    Nmax = 10;
    Nvals = np.linspace(0,Nmax,min(Nmax, 20),dtype = int);
    Tvals = [];
    for N in Nvals:

        # package as block hams 
        # number of blocks depends on N
        hblocks = [np.zeros_like(h_SR)]
        tblocks = [-tl*np.eye(*np.shape(h_SR)) ];
        for Ni in range(N):
            hblocks.append(np.copy(h_SR));
            tblocks.append(-tl*np.eye(*np.shape(h_SR)) );
        hblocks.append(np.zeros_like(h_SR) );
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    xlab = "$N$"
    for bi in range(np.shape(h_SR)[0]):
        ax.plot(Nvals, Tvals[:,bi], label = bi);
    plt.legend();
    plt.show();

    # again with entangled state
    h_SR = entangle(h_SR,1,2);
    print(h_SR);
    Tvals = [];
    for N in Nvals:

        # package as block hams 
        # number of blocks depends on N
        hblocks = [np.zeros_like(h_SR)]
        tblocks = [-tl*np.eye(*np.shape(h_SR)) ];
        for Ni in range(N):
            hblocks.append(np.copy(h_SR));
            tblocks.append(-tl*np.eye(*np.shape(h_SR)) );
        hblocks.append(np.zeros_like(h_SR) );
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    xlab = "$N$"
    for bi in range(np.shape(h_SR)[0]):
        ax.plot(Nvals, Tvals[:,bi], label = bi);
    plt.legend();
    plt.show();
    assert False;

#### run code

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
option = sys.argv[1];

# tight binding params
tl = 1.0; # 2e hopping, in meV
JK = -0.0;

# construct h_SR and define source
h1e, g2e = second_q_ham(JK, JK); # second qu'd form
parts = np.array([1,1,1]); # one e, 2 separate impurities
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz

# prep system
mT = 1/2; # total z spin
mystates, myis, mystrs = subspace(mT); # subspace
h_SR = fci_mod.single_to_det(h1e, g2e, parts, states, dets_interest = mystates);
source = np.zeros(np.shape(h_SR)[0]);
source[1] = 1/np.sqrt(2);
source[2] = 1/np.sqrt(2);
print(h_SR);
mysource = r"$0.71|up, \frac{1}{2}, -\frac{1}{2} \rangle + 0.71|up, -\frac{1}{2}, \frac{1}{2} \rangle$"

#### what to iter over

if option == "E": # iter over energy

    # sweep from bottom of band up
    Evals = np.linspace(-2*tl, -2*tl + 0.6*tl, 20);

    # package h, t block matrices
    hblocks = np.array([np.zeros_like(h_SR), h_SR, np.zeros_like(h_SR)]);
    tblocks = np.array([-tl*np.eye(*np.shape(h_SR)),-tl*np.eye(*np.shape(h_SR))]);

    # get coefs at each energy
    Tvals = [];
    for Ei in range(len(Evals) ):
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Evals[Ei], source));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    xlab = "$E+2t_l $"
    for pi in range(len(myis)):
        ax.scatter(Evals + 2*tl,Tvals[:,myis[pi]], marker = 's',label = mystrs[pi]);
                                                    
    
elif option == "N": # stretch SR, switzer style

    # fix energy near bottom of band
    Energy = -2*tl + 0.5;
    ka = np.arccos(Energy/(-2*tl));

    # iter over N
    Nmax = 10;
    Nvals = np.linspace(0,Nmax,30,dtype = int);
    Tvals = [];
    for N in Nvals:

        # package as block hams 
        # number of blocks depends on N
        hblocks = [np.zeros_like(h_SR)]
        tblocks = [-tl*np.eye(*np.shape(h_SR)) ];
        for Ni in range(N):
            hblocks.append(np.copy(h_SR));
            tblocks.append(-tl*np.eye(*np.shape(h_SR)) );
        hblocks.append(np.zeros_like(h_SR) );
        hblocks = np.array(hblocks);
        tblocks = np.array(tblocks);

        # coefs
        Tvals.append(wfm.Tcoef(hblocks, tblocks, tl, Energy, source));

    # plot Tvals vs E
    Tvals = np.array(Tvals);
    fig, ax = plt.subplots();
    xlab = "$N$"
    for pi in range(len(myis)):
        ax.plot(Nvals,Tvals[:,pi], label = mystrs[pi]);


# format and plot
#ax.set_ylim(0.0,0.25);
ax.minorticks_on();
ax.grid(which='major', color='#DDDDDD', linewidth=0.8);
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
ax.set_xlabel(xlab);
ax.set_ylabel("$T$");
ax.set_title("source = "+mysource+", $J_{K} = $"+str(JK));
plt.legend();
plt.show();




