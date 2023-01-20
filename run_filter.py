'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 4:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from code import fci_mod, wfm
from code.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
import sys

#### top level
#np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### setup

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
#dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets52 = [[0,2,7],[0,3,6],[1,2,6]]; # total spin 5/2 subspace

# initialize source vector in down, 3/2, 3/2 state
sourcei = 2; # |down, 3/2, 3/2 >
assert(sourcei >= 0 and sourcei < len(dets52));
source = np.zeros(len(dets52));
source[sourcei] = 1;
source_str = "|";
for si in dets52[sourcei]: source_str += state_strs[si];
source_str += ">";
if(verbose): print("\nSource:\n"+source_str);

# entangle pair
pair = (0,1); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >
if(verbose):
    print("\nEntangled pair:");
    pair_strs = [];
    for pi in pair:
        pair_str = "|";
        for si in dets52[pi]: pair_str += state_strs[si];
        pair_str += ">";
        print(pair_str);
        pair_strs.append(pair_str);

tl = 1.0;
tp = 1.0;
JK = 0.1;
J12 = JK/10;
J12x, J12y, J12z = J12, J12, J12;
            
#########################################################
#### supress Ti by making DeltaE < 0 and shifting chem pot in RL

if True: # generate data
    Esplit = -0.005; # closer to 0 typically better # NB for manganese = 0.003
    Dval = -Esplit/2;
    chem_LL = 0.0;
    chem_RL = -Esplit;

    # iter over zeeman splittings
    zeemanvals = [0.05]; # [0, 0.01, 0.02, 0.05,0.08];
    for zeemani in range(len(zeemanvals)):
        zeeman_LL = zeemanvals[zeemani]; # energy cost of deloc'd E flipping up upon reflection
        zeeman_RL = 0.0; # energy cost of deloc'd E flipping up upon reflection

        # iter over E, getting T
        logElims = -4,-1;
        Evals = np.logspace(*logElims,myxvals);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            
            # optical distances, N = 2 fixed
            N0 = 1; # N0 = N - 1

            # construct hblocks
            hblocks = [];
            impis = [1,2];
            for j in range(4): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = J12x, J12y, J12z, Dval, Dval, 0, JK1, JK2;
                h1e, g2e = wfm.utils.h_cobalt_2q(params); # construct ham
                # construct h_SR (determinant basis)
                hSR = fci_mod.single_to_det(h1e, g2e, species, states, dets_interest = dets52);            
                # transform to eigenbasis
                hSR_diag = wfm.utils.entangle(hSR, *pair);
                hblocks.append(np.copy(hSR_diag));
                if(verbose > 3 and Eval == Evals[0]):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", np.real(hSR));
                    print(" - transformed ham:\n", np.real(hSR_diag));
                    print(" - DeltaE = ",Esplit)

            # finish hblocks
            hblocks = np.array(hblocks);
            # chem potential shift
            hblocks[0] += chem_LL*np.eye(*np.shape(hblocks[0]));
            hblocks[-1] += chem_RL*np.eye(*np.shape(hblocks[-1]));
            # zeeman for deloc'd e in leads
            hblocks[0] += zeeman_LL*np.diag([1,1,0]); # because |+> and |-> have e up
            hblocks[-1] += zeeman_RL*np.diag([1,1,0]);
            # const shift st hLL[sourcei,sourcei] = 0
            E_shift = hblocks[0,sourcei,sourcei]; 
            for hb in hblocks:
                hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            # verify Esplit
            print("Delta E / t = ", (hblocks[-1][0,0] - hblocks[-1][2,2])/tl); # RL is what matters

            # hopping
            tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;
            
        # save data to .npy
        data = np.zeros((2+2*len(source),len(Evals)));
        data[0,0] = tl;
        data[0,1] = JK;
        data[1,:] = Evals;
        data[2:2+len(source),:] = Tvals.T;
        data[2+len(source):2+2*len(source),:] = Rvals.T;
        fname = "data/model32_filter/Esplit"+str(int(Esplit*1000)/1000)+"_zeeman"+str(int(zeemanvals[zeemani]*1000)/1000);
        print("Saving data to "+fname);
        np.save(fname, data);

# load data
def load_data(fname):
    print("Loading data from "+fname);
    data = np.load(fname);
    tl = data[0,0];
    Jeff = data[0,1];
    myxvals = data[1];
    myTvals = data[2:5];
    myRvals = data[5:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
    return myxvals, myRvals, myTvals, mytotals;

# p2
def p2(Ti,Tp,theta):
    assert isinstance(Ti,float) and isinstance(Tp,float); # vectorized in thetas only
    if Tp == 0.0: Tp = 1e-10;
    return Ti*Tp/(Tp*np.cos(theta/2)*np.cos(theta/2)+Ti*np.sin(theta/2)*np.sin(theta/2));

# figure of merit
def FOM(Ti,Tp, grid=100000):
    thetavals = np.linspace(0,np.pi,grid);
    p2vals = p2(Ti,Tp,thetavals);
    fom = np.trapz(p2vals, thetavals)/np.pi;
    return fom;

if True: # plot data
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datafs[fi]);
        logElims = -4,-1;

        # plot T
        axes[0].plot(xvals, Tvals[sourcei], linestyle = 'solid', color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
        axes[0].plot(xvals, Tvals[pair[0]], linestyle = 'dashed', color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 

        # plot R
        axes[1].plot(xvals, Rvals[sourcei], linestyle = 'solid', color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
        axes[1].plot(xvals, Rvals[pair[0]], linestyle = 'dashed', color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery, linewidth = mylinewidth); 
        axes[1].plot(xvals, totals, color='red');
    # format
    #axes[0].set_ylim(0,0.16);
    axes[0].set_ylabel('$T_{\sigma}$', fontsize = myfontsize);
    #axes[1].set_ylim(0.15,0.25);
    axes[1].set_ylabel('$R_{\sigma}$', fontsize = myfontsize);

    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    #plt.savefig('figs/model32.pdf');
    plt.show();


