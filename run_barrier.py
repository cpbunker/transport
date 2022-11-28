'''
Christian Bunker
M^2QM at UF
November 2022

Time independent scattering formalism using GF's
Exact solution for a potential barrier
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = cm.get_cmap('Set1');
mymarkers = ["o","^","s","d","*","X","P"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# tight binding params
tl = 1.0;
Vb = 0.1; # barrier height
NC = 5; # barrier width

# blocks and inter block hopping
hblocks = [[[0]]];
for _ in range(2*NC+1): hblocks.append([[Vb]]);
hblocks.append([[0]]);
hblocks = np.array(hblocks, dtype = float);
tnn = [[[-tl]]];
for _ in range(2*NC+1): tnn.append([[-tl]]);
tnn = np.array(tnn);
tnnn = np.zeros_like(tnn)[:-1];
if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn); 

# source
source = np.zeros(np.shape(hblocks[0])[0]);
source[0] = 1;

# sweep over range of energies
# def range
logElims = -3,0
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
numplots = 2;
fig, axes = plt.subplots(numplots, sharex = True);
if numplots == 1: axes = [axes];
fig.set_size_inches(7/2,3*numplots/2);
axes[1].plot(Evals, np.real(Tvals[:,0]), color=mycolors(1), marker=mymarkers[1], markevery=mymarkevery, linewidth=mylinewidth); 

# approximations to ideal
kavals = np.arccos((Evals-2*tl-hblocks[0][0,0])/(-2*tl));
kappavals = np.arccosh((Evals-2*tl-hblocks[1][0,0])/(-2*tl));
ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
ideal_exp = np.exp(-2*(2*NC+1)*kappavals);
ideal_Tvals = ideal_prefactor*ideal_exp;
ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
ideal_Tvals *= ideal_correction
axes[0].plot(Evals,np.real(ideal_Tvals), color = 'black', linewidth = mylinewidth);
axes[0].plot(Evals,np.real(ideal_Tvals/ideal_correction), color=mycolors(0), linewidth=mylinewidth, marker=mymarkers[0], markevery=mymarkevery);
axes[0].set_ylim(0,0.2);
axes[0].set_ylabel('$T_{approx}$');

# ideal comparison
axes[1].plot(Evals,np.real(ideal_Tvals), color = 'black', linewidth = mylinewidth);
axes[1].set_ylim(0,1);
axes[1].set_ylabel('$T_{numerical}$');
        
# format and show
axes[-1].set_xscale('log', subs = []);
axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize); 
plt.tight_layout();
plt.savefig('figs/barrier.pdf');
   







