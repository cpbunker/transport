'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering from a potential barrier
benchmarked to exact solution
solved in time-independent QM using wfm method in transport/wfm
'''

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","^","s","d","*","X","P"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# tight binding params
tl = 1.0;
Vb = 0.0; # barrier height
VR = 0.0; # right well height
Vinfty = 0.1; # far right potential
NC = 11; # barrier width
NR = 0; # right well width
Ninfty = 5;
n_loc_dof = 1;

# build hamiltonian
# left lead mu is 0
hblocks = [0*np.eye(n_loc_dof)];
# add barrier region
for _ in range(NC): hblocks.append(Vb*np.eye(n_loc_dof));
# add right well region
for _ in range(NR): hblocks.append(VR*np.eye(n_loc_dof));
# classically forbidden region
for _ in range(Ninfty): hblocks.append(Vinfty*np.eye(n_loc_dof));
# 0 at end
for _ in range(Ninfty): hblocks.append(Vinfty*np.eye(n_loc_dof));
hblocks = np.array(hblocks, dtype = float);

# hopping
tnn = [-tl*np.eye(n_loc_dof)];
for _ in range(NC): tnn.append(-tl*np.eye(n_loc_dof));
for _ in range(NR): tnn.append(-tl*np.eye(n_loc_dof));
for _ in range(2*Ninfty-1): tnn.append(-tl*np.eye(n_loc_dof));
tnn = np.array(tnn);
tnnn = np.zeros_like(tnn)[:-1];
if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);
plt.plot(hblocks[:,0,0]);
plt.title('$V_j$');
plt.show();  

# source
source = np.zeros(np.shape(hblocks[0])[0]);
source[0] = 1;

# sweep over range of energies
# def range
logElims = -2,0
Evals = np.logspace(*logElims,myxvals, dtype=complex);

# test main wfm kernel
Rvals = np.empty((len(Evals),len(source)), dtype = float);
Tvals = np.empty((len(Evals),len(source)), dtype = float); 
for Evali in range(len(Evals)):
    # energy
    Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
    Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

    if(Evali < 1): # verbose
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False, verbose = verbose);
    else: # not verbose
         Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
    Rvals[Evali] = Rdum;
    Tvals[Evali] = Tdum;


# plot Tvals vs E
numplots = 1;
fig, axes = plt.subplots(numplots, sharex = True);
if numplots == 1: axes = [axes];
fig.set_size_inches(7/2,3*numplots/2);
axes[0].plot(Evals, np.real(Tvals[:,0]), color=mycolors[0], marker=mymarkers[0], markevery=mymarkevery, linewidth=mylinewidth); 
axes[0].plot(Evals, np.real(Rvals[:,0]), color=mycolors[1], marker=mymarkers[1], markevery=mymarkevery, linewidth=mylinewidth); 

# ideal
kavals = np.arccos((Evals-2*tl-hblocks[0][0,0])/(-2*tl));
kappavals = np.arccos((Evals-2*tl-hblocks[-1][0,0])/(-2*tl));
ideal_Rvals = np.power((kavals-kappavals)/(kavals+kappavals),2);

# ideal comparison
axes[0].plot(np.real(Evals),np.real(ideal_Rvals), color = accentcolors[0], linewidth = mylinewidth);
#axes[0].set_ylim(-0.1,1.1);
axes[0].set_ylabel('$T$');
        
# format and show
axes[-1].set_xscale('log', subs = []);
axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize); 
plt.tight_layout();
plt.show();
   







