'''
Christian Bunker
M^2QM at UF
February 2023

Play around with absorbing and emitting boundary conditions
'''

from transport import bardeen, fci_mod

import numpy as np

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 3;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def print_H_j(H) -> None:
    if(len(np.shape(H)) != 4): raise ValueError;
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def print_H_alpha(H) -> None:
    if(len(np.shape(H)) != 4): raise ValueError;
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

def h_kondo(J, s2) -> np.ndarray:
    '''
    Kondo interaction between spin 1/2 and spin s2
    '''
    n_loc_dof = int(2*(2*s2+1));
    h = np.zeros((n_loc_dof,n_loc_dof),dtype=complex);
    if(s2 == 0.5):
        h[0,0] = 1;
        h[1,1] = -1;
        h[2,2] = -1;
        h[3,3] = 1;
        h[1,2] = 2;
        h[2,1] = 2;
        h *= J/4;
    else: raise NotImplementedError;
    return h;

def h_tb(t, V, N) -> np.ndarray:
    if(not isinstance(V, np.ndarray)): raise TypeError;
    spatial_orbs = N;
    n_loc_dof = len(V);
    h_4d = np.zeros((spatial_orbs,spatial_orbs,n_loc_dof,n_loc_dof),dtype=complex);
    for spacei in range(spatial_orbs):
        h_4d[spacei,spacei] += V;
        if(spacei < spatial_orbs-1):
            h_4d[spacei+1,spacei] += -t;
            h_4d[spacei,spacei+1] += -t;
    return h_4d;


#################################################################
#### visualize eigenspectrum of different HL's

# redo 1d spinless bardeen theory with absorbing/emitting bcs
# for no reflection, absorbing bcs should be smooth
# look at xgz paper for better spin-flip bcs
# spin rotation region + infinite wall

if True: # spinless case

    # setup
    NL = 7;
    tL = 1.0*np.eye(1);
    VL = 0.0*np.eye(len(tL));

    # construct well and add spin parts
    E_cut = -1.8;
    E_cont = -1.933; # energy of the continuum state
    alpha_cont = 0; # spin of the continuum state
    HL = h_tb(tL,VL,NL);
    x0, V0 = NL//2, (2.0+E_cont)/0.5;
    for xi in range(NL):
        if(xi > x0):
            HL[xi,xi] += V0*np.ones_like(tL);
    # couple to continuum
    if False:
        HL = bardeen.couple_to_cont(HL,E_cont,alpha_cont);
    else:
        kaval = np.lib.scimath.arccos((E_cont)/(-2));
        kapaval = np.lib.scimath.arccos((E_cont-V0)/(-2));
        HL[0,0] += -tL*np.exp(complex(0,-kaval));
        HL[-1,-1] += -tL*np.exp(complex(0,kapaval));
    print_H_alpha(HL);

    # plot HL eigenfunctions
    bardeen.plot_wfs(HL,E_cont,alpha_cont,E_cut);
    
if False: # spinful case

    # setup
    NL = 17;
    tL = np.eye(2);
    VL = 0.0*np.eye(len(tL));
    Jval = -0.5;

    # construct well and add spin parts
    E_cut = -1.8;
    E_cont = -1.933;
    alpha_cont = 0;
    HL = h_tb(tL,VL,NL);

    # mix in spin
    spinpart = h_kondo(Jval,0.5);
    HL[NL//2,NL//2] += spinpart[1:3,1:3];
    HL = bardeen.couple_to_cont(HL,E_cont,alpha_cont);
    print_H_alpha(HL);

    # plot HL eigenfunctions
    bardeen.plot_wfs(HL,E_cont,alpha_cont,E_cut);

        

