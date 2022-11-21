'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a step potential

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkcyan", "darkmagenta","darkgray"];
mymarkers = ["o","^","s","d","*","X","P"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
tl = 1.0;
Vb = 0.1; # barrier height
NC = 1; # barrier width

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
logElims = -6,0
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
axes[0].plot(Evals, Tvals[:,0], color=mycolors[0], marker = mymarkers[0], linewidth = mylinewidth, markevery = mymarkevery); 
axes[0].set_ylim(0,1);

#axes[1].plot(Evals, Rvals[:,0], color=mycolors[1], marker = mymarkers[1], linewidth = mylinewidth, markevery = mymarkevery); 
#totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
#axes[1].plot(Evals, totals, color="red");
#axes[1].set_ylim(0,1);

# ideal comparison
kavals = np.arccos((Evals-2*tl-hblocks[0][0,0])/(-2*tl));
kappavals = np.arccosh((Evals-2*tl-hblocks[1][0,0])/(-2*tl));
ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
ideal_exp = np.exp(-2*NC*kappavals);
ideal_Tvals = ideal_prefactor*ideal_exp;
ideal_Tvals *= np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
axes[0].plot(Evals,ideal_Tvals);
#axes[1].plot(Evals,kappavals);
#axes[1].plot(Evals,np.sqrt(Vb - Evals));
        
# format and show
axes[-1].set_xscale('log', subs = []);
axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
plt.show();
   







