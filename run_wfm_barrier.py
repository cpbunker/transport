'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering from a rectangular potential barrier
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
mymarkers = ["+","o","^","s","d","*","X"];
mymarkevery = (10, 10);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
plt.rcParams.update({"font.family": "serif"})
#plt.rcParams.update({"text.usetex": True}) 

# tight binding params
tl = 1.0;
Vb = 0.1; # barrier height
NC = 11; # barrier width

# blocks and inter block hopping
hblocks = [[[0]]];
for _ in range(NC): hblocks.append([[Vb]]);
hblocks.append([[0]]);
hblocks = np.array(hblocks, dtype = float);
tnn = [[[-tl]]];
for _ in range(NC): tnn.append([[-tl]]);
tnn = np.array(tnn);
tnnn = np.zeros_like(tnn)[:-1];
if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn); 

# source
source = np.zeros(np.shape(hblocks[0])[0]);
source[0] = 1;

# sweep over range of energies
# def range
logKlims = -3,0
Kvals = np.logspace(*logKlims,myxvals, dtype=complex);

# test main wfm kernel
Rvals = np.empty((len(Kvals),len(source)), dtype = float);
Tvals = np.empty((len(Kvals),len(source)), dtype = float); 
for Kvali in range(len(Kvals)):
    # energy
    Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
    Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper

    if(Kvali < 5): # verbose
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, 
                         False, False, all_debug = True, verbose = verbose);
    else: # not verbose
         Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, 
                         False, False, all_debug = False, verbose = 0);
    Rvals[Kvali] = Rdum;
    Tvals[Kvali] = Tdum;

# plot Tvals vs E
numplots = 1;
fig, axes = plt.subplots(numplots, sharex = True);
if numplots == 1: axes = [axes];
fig.set_size_inches(7,3*numplots);
axes[0].plot(np.real(Kvals), np.real(Tvals[:,0]), color=mycolors[0], marker=mymarkers[1], markevery=mymarkevery, linewidth=mylinewidth); 

# ideal
kavals = np.arccos((Kvals-2*tl-hblocks[0][0,0])/(-2*tl));
kappavals = np.arccosh((Kvals-2*tl-hblocks[1][0,0])/(-2*tl));
ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
ideal_exp = np.exp(-2*NC*kappavals);
ideal_Tvals = ideal_prefactor*ideal_exp;
ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
ideal_Tvals *= ideal_correction

# ideal comparison
axes[0].plot(np.real(Kvals),np.real(ideal_Tvals), color = 'black', marker=mymarkers[0], markevery=mymarkevery, linewidth = mylinewidth);
y_offset = 0.07;
yticks = [0.0, 1.0]
axes[0].set_ylim(yticks[0]-y_offset, yticks[1]+y_offset);
for tick in yticks: axes[0].axhline(tick, linestyle="dashed", color="gray");
axes[0].set_ylabel('$T$');
        
# format and show
axes[-1].set_xscale('log', subs = []);
axes[-1].set_xlim(10**(logKlims[0]), 10**(logKlims[1]));
axes[-1].set_xticks([10**(logKlims[0]), 10**(logKlims[1])]);
axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize); 
plt.tight_layout();
plt.show();
   







