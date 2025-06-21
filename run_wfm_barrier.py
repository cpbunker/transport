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
    if(case not in ["VB","CB"]): raise NotImplementedError;

    # fig standardizing
    myxvals = 199;
    myfontsize = 14;
    mylinewidth = 1.0;
    from transport.wfm import UniversalColors, UniversalAccents, ColorsMarkers, AccentsMarkers, UniversalMarkevery, UniversalPanels;
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
NC = 14; # barrier width
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
logKlims = -4,0
Kvals = np.logspace(*logKlims,myxvals, dtype=complex);

# ideal single-channel transmission through barrier
#kavals = np.arccos(((Kvals+np.min(-band_edges))**2-uval**2 -vval**2 -tl**2)/(-2*vval*tl));
#kappavals = np.arccos(((Kvals+np.min(-band_edges))**2-uval**2 -vval**2 -tl**2)/(-2*vval*tl));
kavals = np.arccos((Kvals-2*tl-hblocks[0][0,0])/(-2*tl)); # <--- unshifted !!
kappavals = np.arccosh((Kvals-2*tl-hblocks[1][0,0])/(-2*tl));
ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
ideal_exp = np.exp(-2*NC*kappavals);
ideal_Tvals = ideal_prefactor*ideal_exp;
ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
ideal_Tvals *= ideal_correction

# closed form surface green's function
conv_closed = "g_closed"
Tvals_clos = np.full((len(Kvals),len(source)), np.nan, dtype = float); 
for Kvali in range(len(Kvals)):
    # energy
    Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
    Energy = Kval - 2*tl; # <--- unshifted !!

    if(Kvali < 5): # verbose
        # pass converger=g_closed tell code to use closed form surface greens func
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn,tl,Energy,conv_closed,source, 
                         False, False, all_debug = True, verbose = verbose);
    else: # not verbose
         Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn,tl,Energy,conv_closed,source,  
                         False, False, all_debug = False, verbose = 0);
    Tvals_clos[Kvali] = Tdum;

# plot the closed form surface green's function
axes[0].plot(np.real(Kvals), np.real(Tvals_clos[:,0]),
label=conv_closed+", $E_{min}=$"+"{:.2f}".format(-2), 
color=UniversalAccents[0], marker=AccentsMarkers[0], markevery=UniversalMarkevery, linewidth=mylinewidth); 
# and plot its error w/r/t ideal
axes[1].plot(np.real(Kvals),abs(np.real((ideal_Tvals-Tvals_clos[:,0])/ideal_Tvals)), 
  color =UniversalAccents[0], marker=AccentsMarkers[0], markevery=UniversalMarkevery, linewidth = mylinewidth); 

#################################################################
# test various schemes for surface green's function of Rice-Mele
# this requires constructing diatomic Hamiltonian!!

# the model for the diatomic unit cell model is the Rice-Mele model
# parameterized by staggered on-site potential +u,-u and staggered hopping vow
vval = float(sys.argv[2]);
uval = float(sys.argv[3]); 
# w is always -tl;
band_edges = np.array([np.sqrt(uval*uval+(-tl+vval)*(-tl+vval)),
                       np.sqrt(uval*uval+(-tl-vval)*(-tl-vval))]);
RiceMele_shift = np.min(-band_edges) + 2*tl; # new band bottom - old band bottom
if(case=="CB"): RiceMele_shift = np.min(band_edges) + 2*tl; # new band = conduction band!
RiceMele_Energies = Kvals - 2*tl + RiceMele_shift; # value in the RM band
RiceMele_numbers = np.arccos(1/(2*vval*(-tl))*(RiceMele_Energies**2 - uval**2 - vval**2 - tl**2));

# construct diatomic Hamiltonian
del hblocks, tnn, tnnn, source;
# NC = NC*2 # <------ destroys agreement
h00 = np.array([[uval, vval], [vval, -uval]]);
h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
print("\n\nRice-Mele, v = {:.2f}, u = {:.2f}".format(vval,uval));
print("h00 =\n",h00);
print("h01 =\n",h01);
h_dia = [1*h00]; # LL part of hblocks
for _ in range(NC//2): h_dia.append(h00+Vb*np.eye(len(h00))); # include barrier
h_dia.append(1*h00); # RL part of hblocks
tnn_dia = [1*h01]; # *upper diagonal* hopping blocks
for _ in range(NC//2): tnn_dia.append(1*h01);
# we assume that h01 is same for LL, SR, RL
for further_cell in range(200): # to verify that evaluating further into lead does not change the physics
    h_dia.append(1*h00)
    tnn_dia.append(1*h01)
h_dia = np.array(h_dia, dtype = float);
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

# run over the different methods for handling the diatomic Rice-Mele unit cell
# and getting the surface green's function
# these methods are: call g_RiceMele, use iterative green's func
imag_pt_E = float(sys.argv[4])       # for iterative gf scheme, E needs small >0 imag pt
iterative_tol = 1e-3  # and we need tolerance at which we stop iterating
myconverger_values = ["g_RiceMele"] #, (imag_pt_E, iterative_tol)];
for myconvergeri in range(len(myconverger_values)):
    myconverger = myconverger_values[myconvergeri];

    # get Tvals corresponding to this diatomic method
    Tvals_iter = np.full((len(Kvals),len(source_dia)), np.nan, dtype = float); 
    for Kvali in range(len(Kvals)):
        # energy
        Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
        Energy = Kval-2*tl+RiceMele_shift; # energy that is `Kval` above either VB or CB
        if(Kvali < 5): # verbose
            # passing conv_tol assures we use iterative scheme
            Rdum, Tdum = wfm.kernel(h_dia, tnn_dia, tnnn_dia,tl,Energy,myconverger, 
                       source_dia, False, False, all_debug=True, verbose=verbose);
        else: # not verbose
            Rdum, Tdum = wfm.kernel(h_dia, tnn_dia, tnnn_dia,tl,Energy,myconverger,
                        source_dia, False, False, all_debug=True, verbose=0);
        Tvals_iter[Kvali] = Tdum;

    # plot the Tvals
    if(not isinstance(myconverger, str)): 
        myconverger_lab = "$\eta =${:.0e}, tol$=${:.0e}".format(*myconverger);
    else:
        myconverger_lab = myconverger[:];

    # NB Tvals[:,dia_out] gives transmission *into the right lead*
    axes[0].plot(np.real(Kvals), np.real(Tvals_iter[:,dia_out]), 
      label=myconverger_lab, color=UniversalColors[myconvergeri], marker=ColorsMarkers[myconvergeri], 
      markevery=UniversalMarkevery, linewidth=mylinewidth);
    axes[1].plot(np.real(Kvals),abs(np.real((ideal_Tvals-Tvals_iter[:,dia_out])/ideal_Tvals)),
      color=UniversalColors[myconvergeri], markevery=UniversalMarkevery, linewidth=mylinewidth); 

### end loop over different gf methods

# plot ideal for comparison
axes[0].plot(np.real(Kvals),np.real(ideal_Tvals), label="exact, $E_{min}=$"+"{:.2f}".format(-2), 
  color = UniversalAccents[1], marker=AccentsMarkers[1], markevery=UniversalMarkevery, linewidth = mylinewidth);
       
# format
title_str = "Barrier height$=${:.2f}, barrier width$=${:.0f}".format(Vb, NC)
title_str += ", $u=${:.2f}, $v=${:.2f}, $w=${:.2f}".format(uval, vval, -tl)
axes[0].set_title(title_str)
axes[0].legend();
axes[0].set_ylabel('$T$');
axes[1].set_ylabel("Relative error");
y_offset = 0.07;
yticks = [0.0, 1.0]
axes[0].set_ylim(yticks[0]-y_offset, yticks[1]+y_offset);
for tick in yticks: axes[0].axhline(tick, linestyle="dashed", color="gray");
axes[-1].set_xscale('log', subs = []);
axes[-1].set_xlim(10**(logKlims[0]), 10**(logKlims[1]));
axes[-1].set_xticks([10**(logKlims[0]), 10**(logKlims[1])]);
RiceMele_shift_str = "$-E_{min}^{(VB)}, E_{min}^{(VB)}=$"+"{:.2f}".format(np.min(-band_edges))
if(case=="CB"): RiceMele_shift_str="$-E_{min}^{(CB)},  E_{min}^{(CB)}=$"+"{:.2f}".format(np.min(band_edges))
RiceMele_shift_str += ",  $ka/\pi \in $[{:.2f},{:.2f}]".format(np.real(RiceMele_numbers[0]/np.pi), np.real(RiceMele_numbers[-1]/np.pi))
axes[-1].set_xlabel("$E$"+RiceMele_shift_str,fontsize = myfontsize);

# show
plt.tight_layout();
plt.show();
   







