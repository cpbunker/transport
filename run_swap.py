'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 4:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import fci_mod, wfm
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
import sys

#### top level
#np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["black","darkblue","darkgreen","darkred", "darkcyan", "darkmagenta","darkgray"];
mymarkers = ["o","^","s","d","*","X","P"];
mycolors, mymarkers = np.append(mycolors,mycolors), np.append(mymarkers, mymarkers);
def mymarkevery(fname,yvalues):
    if True:
        return (40,40);
    else:
        return [np.argmax(yvalues)];
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

#### setup

# entangle pair
pair = (0,1); # |up, 1/2, 3/2 > and |up, 3/2, 1/2 >
sourcei = 2;
source = np.array([0.0,1.0,0.0]);

tl = 1.0;
tp = 1.0;
JK = -0.5*tl/100;
J12 = tl/100;

# constructing the hamiltonian
def reduced_ham(params, S):
    D1, D2, J12, JK1, JK2 = params;
    assert(D1 == D2);
    ham = np.array([[S*S*D1+(S-1)*(S-1)*D2+S*(S-1)*J12+(JK1/2)*S+(JK2/2)*(S-1), S*J12, np.sqrt(2*S)*(JK2/2) ], # up, s, s-1
                    [S*J12, (S-1)*(S-1)*D1+S*S*D2+S*(S-1)*J12+(JK1/2)*S + (JK2/2)*(S-1), np.sqrt(2*S)*(JK1/2) ], # up, s-1, s
                    [np.sqrt(2*S)*(JK2/2), np.sqrt(2*S)*(JK1/2),S*S*D1+S*S*D2+S*S*J12+(-JK1/2)*S +(-JK2/2)*S]], # down, s, s
                   dtype = complex);

    return ham;
            
#########################################################
#### effects of Ki and Delta E

if True: # T+ at different Delta E by changing D
    myspinS = 0.5;
    Dval = 0.0; # order of D: 0.1 meV for Mn to 1 meV for MnPc
    Nbarriervals = np.linspace(0,50,6);
    for Nvali in Nbarriervals:
        Nval = Nbarriervals[Nvali];

        # optical distances, N = 2 fixed
        N0 = 1; # N0 = N - 1

        # construct hblocks from spin ham
        hblocks = [];
        impis = [1,2];
        for j in range(4): # LL, imp 1, imp 2, RL
            # define all physical params
            JK1, JK2 = 0, 0;
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = Dval, Dval, J12, JK1, JK2;
            # construct h_SR (determinant basis)
            hSR = reduced_ham(params,S=myspinS);           
            hblocks.append(np.copy(hSR));
            if(verbose > 3 ):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", hSR);

        # add large barrier at end
        for _ in range(Nval):
            hblocks.append(np.zeros_like(hblocks[0]));
        hblocks[-1] += 2*tl*np.eye(len(source));

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
            
        # hopping
        tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over E, getting T
        logElims = -6,-0
        Evals = np.logspace(*logElims,myxvals, dtype = complex);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

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
        fname = "data/temp/"+str(myspinS);
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
    return myxvals, myRvals, myTvals, mytotals;

#### plot T+ and p2 vs Ki at different Delta E
if True:
    num_plots = 3;
    fig, axes = plt.subplots(num_plots, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    datafs = sys.argv[1:];
    for fi in range(len(datafs)):
        xvals, Rvals, Tvals, totals = load_data(datafs[fi]);
        logElims = np.log10(xvals[0]), np.log10(xvals[-1]);

        # plot T_sigma
        for sigmai in range(len(source)):
            axes[sigmai].plot(xvals, Rvals[sigmai], color=mycolors[fi], marker=mymarkers[fi], markevery=mymarkevery(datafs[fi],Rvals[sigmai]), linewidth = mylinewidth);        
            axes[sigmai].set_ylabel('$R_'+str(sigmai)+'$', fontsize = myfontsize);
        axes[-1].plot(xvals, totals, color='red');
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    #plt.savefig();
    plt.show();

