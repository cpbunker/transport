'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering from a rectangular potential barrier
benchmarked to exact solution
solved in time-dependent QM using bardeen theory method in transport/bardeen
'''

from transport import bardeen

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

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

# colormap
cm_reds = matplotlib.cm.get_cmap("seismic");
def get_color(colori,numcolors):
    denominator = 2*numcolors
    assert(colori >=0 and colori < numcolors);
    if colori <= numcolors // 2: # get a blue
        return cm_reds((1+colori)/denominator);
    else:
        return cm_reds((denominator-(numcolors-(colori+1)))/denominator);

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

#################################################################
#### benchmarking T in spinless 1D case

# tight binding params
n_loc_dof = 1; 
tL = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tL;
tR = 1.0*tL;
Vinfty = 0.5*tL;
VL = 0.0*tL;
VR = 0.0*tL;

# Ms and overlaps vs NL
if True:

    numplots = n_loc_dof;
    fig, axes = plt.subplots(nrows=numplots, ncols=2, sharex = True);
    if numplots == 1: axes = np.array([axes]);
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tL;
    VC = 0.1*tL;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] = VC;
    for j in range(NC-1):
        HC[j,j+1] = -tC;
        HC[j+1,j] = -tC;
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);

    Ninfty = 20;
    NL = 500;
    NR = 1*NL;
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    Evals, Mels, overlaps = bardeen.kernel_mels(tinfty, tL, tinfty, tR, tinfty,
                                  Vinfty, VL, Vinfty, VR, Vinfty,
                                  Ninfty, NL, NR, HC, HCprime,
                                  E_cutoff=VC[0,0], verbose=1);
    # truncate number of ms
    num_ms = 4; # len(Mels) // 4
    inds = np.linspace(0,len(Mels)-1,num_ms,dtype = int);
    Mels = np.array([Mels[i] for i in inds]);
    overlaps = np.array([overlaps[i] for i in inds]);
    print(inds);
    print("Output shapes:");
    for arr in [Evals, Mels, overlaps]: print(np.shape(arr));

    # only one loc dof, and transmission is diagonal
    alphai = 0;
    for betai in range(n_loc_dof):

        # m is color
        xvals = np.real(Evals[betai])+2*tL[betai,betai];
        for m in range(num_ms):
            axes[betai,0].plot(xvals, Mels[alphai,m,betai], color=get_color(m,num_ms));
            axes[betai,1].plot(xvals, overlaps[alphai,m,betai], color=get_color(m,num_ms));

        # format
        axes[betai,0].set_ylabel('$\langle'+str(alphai)+'|H_{sys}-H_L|'+str(betai)+'\\rangle$',fontsize=myfontsize);
        axes[betai,1].set_ylabel('$\langle'+str(alphai)+'|'+str(betai)+'\\rangle$',fontsize=myfontsize)

# format and show
axes[-1,0].set_xscale('log', subs = []);
axes[-1,0].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
axes[-1,1].set_xscale('log', subs = []);
axes[-1,1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
plt.tight_layout();
plt.show();




