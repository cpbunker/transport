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

import sys

if(__name__=="__main__"):

    # top level
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;
    case = sys.argv[1];

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

    # set up figure
    numplots = 2;
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7,3*numplots);

# tight binding params
tl = 1.0;
Vb = 0.1; # barrier height
NC = 4; # barrier width
assert(NC%2 == 0); # for compatibility with diatomic unit cell

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
imag_pt_E = float(sys.argv[2])
conv_tol = float(sys.argv[3]); # convergence tolerance for iterative gf scheme

# test wmf kernel - closed form surface green's function
Tvals = np.full((len(Kvals),len(source)), np.nan, dtype = float); 
for Kvali in range(len(Kvals)):
    # energy
    Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
    Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper

    if(Kvali < 5): # verbose
        # np here we pass imag part of E = 0.0 since it's not needed, and
        # pass conv_tol = np.nan to tell code to use closed form surface greens func
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn,tl,Energy,0.0,np.nan,source, 
                         False, False, all_debug = True, verbose = verbose);
    else: # not verbose
         Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn,tl,Energy,0.0,np.nan,source,  
                         False, False, all_debug = False, verbose = 0);
    Tvals[Kvali] = Tdum;
axes[0].plot(np.real(Kvals), np.real(Tvals[:,0]), label="closed-form $u=${:.2f}, $v=${:.2f}".format(0.0,-1.0), color=mycolors[0], marker=mymarkers[1], markevery=mymarkevery, linewidth=mylinewidth); 

# ideal single-channel transmission through barrier
kavals = np.arccos((Kvals-2*tl-hblocks[0][0,0])/(-2*tl));
kappavals = np.arccosh((Kvals-2*tl-hblocks[1][0,0])/(-2*tl));
ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
ideal_exp = np.exp(-2*NC*kappavals);
ideal_Tvals = ideal_prefactor*ideal_exp;
ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
ideal_Tvals *= ideal_correction

#################################################################
# test wmf kernel - iterative scheme for surface green's function
# this requires constructing diatomic Hamiltonian!!

# the model for the diatomic unit cell model is the Rice-Mele model
# parameterized by staggered on-site potential +u,-u and staggered hopping vow
uval = float(case);
vval = -1*tl;
# w is just tl;

# construct diatomic Hamiltonian
del hblocks, tnn, tnnn, source
h00 = np.array([[uval, vval], [vval, -uval]]);
h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
print("\n\nRice-Mele, v = {:.2f}, u = {:.2f}".format(vval,uval));
print("h00 =\n",h00);
print("h01 =\n",h01);
h_dia = [1*h00]; # LL part of hblocks
for _ in range(NC//2): h_dia.append(h00+Vb*np.eye(len(h00))); # include barrier
h_dia.append(1*h00); # RL part of hblocks
h_dia = np.array(h_dia, dtype = float);
tnn_dia = [1*h01]; # *upper diagonal* hopping blocks
for _ in range(NC//2): tnn_dia.append(1*h01);
# we assume that h01 is same for LL, SR, RL
tnn_dia = np.array(tnn_dia);
tnnn_dia = np.zeros_like(tnn_dia)[:-1];
if verbose: print("\nh_dia:\n",h_dia,"\ntnn_dia:\n", tnn_dia,"\ntnnn_dia:\n",tnnn_dia);

# source for diatomic system
source_dia = np.zeros(np.shape(h00)[0]);
# indices that tell where in the unit cell the incoming wave comes in and the 
# transmitted wave is read out. It is more logical to have these on the boundaries
# of the chains but I have not observed an accuracy difference
# you can think of changing these as akin to adding more left lead or right lead in
# the part that is considered the SR - it changes where you match waves, but should not
# change the actual results
dia_in, dia_out = 0,1;
source_dia[dia_in] = 1;

# get Evals predicted by the iterative scheme
Tvals_iter = np.full((len(Kvals),len(source_dia)), np.nan, dtype = float); 
for Kvali in range(len(Kvals)):
    # energy
    Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
    Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper

    if(Kvali < 5): # verbose
        # passing conv_tol assures we use iterative scheme
        Rdum, Tdum = wfm.kernel(h_dia, tnn_dia, tnnn_dia,tl,Energy,imag_pt_E, 
                       conv_tol,source_dia, False, False, all_debug=True, verbose=verbose);
    else: # not verbose
         Rdum, Tdum = wfm.kernel(h_dia, tnn_dia, tnnn_dia,tl,Energy,imag_pt_E,
                        conv_tol,source_dia, False, False, all_debug=False, verbose=0);
    Tvals_iter[Kvali] = Tdum;
    # NB Tvals[:,dia_out] gives transmission *into the right lead*
axes[0].plot(np.real(Kvals), np.real(Tvals_iter[:,dia_out]), label="iterative $u=${:.2f}, $v=${:.2f}".format(uval,vval), color=mycolors[1], marker=mymarkers[2], markevery=mymarkevery, linewidth=mylinewidth); 

# plot ideal for comparison
axes[0].plot(np.real(Kvals),np.real(ideal_Tvals), label="exact", color = 'black', marker=mymarkers[0], markevery=mymarkevery, linewidth = mylinewidth);
y_offset = 0.07;
yticks = [0.0, 1.0]
axes[0].set_ylim(yticks[0]-y_offset, yticks[1]+y_offset);
for tick in yticks: axes[0].axhline(tick, linestyle="dashed", color="gray");
axes[0].set_ylabel('$T$');

# plot error w/r/t ideal
axes[1].plot(np.real(Kvals),abs(np.real((ideal_Tvals-Tvals[:,0])/ideal_Tvals)), color = mycolors[0], marker=mymarkers[1], markevery=mymarkevery, linewidth = mylinewidth);
axes[1].plot(np.real(Kvals),abs(np.real((ideal_Tvals-Tvals_iter[:,dia_out])/ideal_Tvals)), color = mycolors[1], marker=mymarkers[2], markevery=mymarkevery, linewidth = mylinewidth);
axes[1].set_ylabel("Relative error");
        
# format
axes[0].set_title("Barrier height$=${:.2f}, barrier width$=${:.0f}, $\eta =${:.0e}, tol$=${:.0e}".format(Vb, NC, imag_pt_E, conv_tol))
axes[0].legend();
axes[-1].set_xscale('log', subs = []);
axes[-1].set_xlim(10**(logKlims[0]), 10**(logKlims[1]));
axes[-1].set_xticks([10**(logKlims[0]), 10**(logKlims[1])]);
axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);

# show
plt.tight_layout();
plt.show();
   







