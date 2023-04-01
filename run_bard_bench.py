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
    assert(colori >=0 and colori <= numcolors);
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
    fig, axes = plt.subplots(nrows=2, ncols=n_loc_dof, sharex = True);
    if n_loc_dof == 1: axes = axes.reshape((2,n_loc_dof));
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tL;
    VC = 1.5*tL;
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
    NL = 200;
    NR = 1*NL;
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    Evals, Mels, overlaps = bardeen.kernel_mels(tinfty, tL, tinfty, tR, tinfty,
                                  Vinfty, VL, Vinfty, VR, Vinfty,
                                  Ninfty, NL, NR, HC, HCprime,
                                  E_cutoff=2*VC[0,0], verbose=1);
    print("Output shapes:");
    for arr in [Evals, Mels, overlaps]: print(np.shape(arr));

    # truncate number of ms
    num_ms = 6; # len(Mels) // 4
    minds = np.linspace(0,np.shape(Mels)[1]-1,num_ms,dtype = int);

    # only one loc dof, and transmission is diagonal
    alphai = 0;
    for betai in range(n_loc_dof):

        # m is color
        xvals = np.real(Evals[betai])+2*tL[betai,betai];
        for mi in range(num_ms):
            axes[0,betai].plot(xvals, Mels[alphai,minds[mi],betai], marker='o', color=get_color(minds[mi],minds[-1]),label="$\\varepsilon_m = $"+str(int(1000*np.real(2+Evals[alphai,minds[mi]]))/1000));
            axes[1,betai].plot(xvals, overlaps[alphai,minds[mi],betai], marker='o', color=get_color(minds[mi],minds[-1]));

        # format
        axes[0,betai].set_ylabel('$\langle k_n |H_{sys}-H_L| k_m \\rangle$',fontsize=myfontsize);
        axes[1,betai].set_ylabel('$\langle k_n | k_m \\rangle$',fontsize=myfontsize);
        axes[0,betai].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[1,betai].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[0,betai].axvline(VC[betai,betai],color='black',linestyle='dashed');
        axes[1,betai].axvline(VC[betai,betai],color='black',linestyle='dashed');
        #axes[-1,betai].set_xscale('log', subs = []);
        #axes[-1,betai].set_xlim(0,VC[betai,betai]);
        axes[-1,betai].set_xlabel('$(\\varepsilon_n + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
        axes[0,betai].legend();

plt.tight_layout();
plt.show();




