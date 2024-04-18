'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 2:
Scattering of single electron off of two localized spin-1/2 impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
'''

#from code import wfm
#from code.wfm import utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import sys

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
pair = (1,2); # pair[0] is the + state after entanglement
sourcei = 4;

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
tl = 1.0;

# choose boundary condition
source = np.zeros(8); 
source[sourcei] = 1; # down up up

##################################################################################
#### entanglement generation (cicc Fig 6)

if False: # compare T vs rhoJa for N not fixed

    # iter over E, getting T
    logElims = -4,-1
    Evals = np.logspace(*logElims,myxvals);
    Rvals = np.empty((len(Evals),len(source)), dtype = float);
    Tvals = np.empty((len(Evals),len(source)), dtype = float);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
        
        # location of impurities, fixed by kx0 = pi
        k_rho = np.arccos(Energy/(-2*tl)); # = ka since \varepsilon_0ss = 0
        kx0 = 2.0*np.pi;
        N0 = max(1,int(kx0/(k_rho))); #N0 = (N-1)
        print(">>> N0 = ",N0);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = wfm.utils.h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
        Rvals[Evali] = Rdum;
        Tvals[Evali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = Evals;
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/Nx/"+str(int(kx0*100)/100);
    print("Saving data to "+fname);
    np.save(fname, data);


if False: # compare T vs rhoJa for N=2 fixed
    Jval = -0.5*tl/100;
    Esplit = 0.0;
    Delta = -Esplit;

    # iter over E, getting T
    logElims = -6,-4;
    Evals = np.logspace(*logElims,myxvals, dtype = complex);
    Rvals = np.empty((len(Evals),len(source)), dtype = float);
    Tvals = np.empty((len(Evals),len(source)), dtype = float);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
        
        # optical distances, N = 2 fixed
        N0 = 1; # N0 = N - 1

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [N0+1];
        hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        # Zeeman splitting effects. NB s=1/2 so 2s-1=0
        hzeeman = np.zeros_like(hblocks[0]);
        hzeeman[sourcei, sourcei] = Delta;
        for hbi in range(len(hblocks)): hblocks[hbi] += np.copy(hzeeman);
        # shift so hblocks[0,i,i] = 0
        Eshift = hblocks[0,sourcei, sourcei];
        for hbi in range(len(hblocks)): hblocks[hbi] += -Eshift*np.eye(len(hblocks[0]));
        if(verbose > 3 and Eval == Evals[0]): print(hblocks);

        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
        Rvals[Evali] = Rdum;
        Tvals[Evali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jval;
    data[1,:] = Evals;
    data[2:10,:] = Tvals.T; # 8 spin dofs
    data[10:,:] = Rvals.T;
    fname = "data/model0.5/N2/"+str(int(Jval*1000)/1000);
    print("Saving data to "+fname);
    np.save(fname, data);


########################################################################
#### plot data

# load data
def load_data(fname):
    print("Loading data from "+fname);
    data = np.load(fname);
    tl = data[0,0];
    Jeff = data[0,1];
    myxvals = data[1];
    myTvals = data[2:10];
    myRvals = data[10:];
    mytotals = np.sum(myTvals, axis = 0) + np.sum(myRvals, axis = 0);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    print("- shape Rvals = ", np.shape(myRvals));
    return myxvals, myRvals, myTvals, mytotals;

# colormap
cm_reds = matplotlib.cm.get_cmap("seismic");
def get_color(colori,numcolors):
    denominator = 2*numcolors
    assert(colori >=0 and colori < numcolors);
    if colori <= numcolors // 2: # get a blue
        return cm_reds((1+colori)/denominator);
    else:
        return cm_reds((denominator-(numcolors-(colori+1)))/denominator);

# p2
def p2(Ti,Tp,theta):
    assert isinstance(Ti,float) and isinstance(Tp,float); # vectorized in thetas only
    return Ti*Tp/(Tp*np.cos(theta/2)*np.cos(theta/2)+Ti*np.sin(theta/2)*np.sin(theta/2));

# figure of merit
def FOM(Ti,Tp, grid=100000):
    thetavals = np.linspace(0,np.pi,grid);
    p2vals = p2(Ti,Tp,thetavals);
    fom = np.trapz(p2vals, thetavals)/np.pi;
    return fom;

#### plot T+ like cicc figure
if True:
    num_plots = 2;
    height_mult = 1;
    fig, axes = plt.subplots(num_plots, 1, gridspec_kw={'height_ratios':[1,height_mult]}, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*(1+height_mult)/2);
    dataf = sys.argv[1];
    xvals, Rvals, Tvals, totals = load_data(dataf);
    logElims = np.log10(xvals[0]), np.log10(xvals[-1]);

    # plot Ti, T+, T-
    lower_y = 0.08;
    sigmas = [sourcei,pair[0],pair[1]];
    for sigmai in range(len(sigmas)):
        factor = 1;
        if sigmas[sigmai] == pair[1]: factor = 10**5; # blow up T-
        axes[0].plot(xvals, factor*Tvals[sigmas[sigmai]],color = mycolors[sigmai],marker = mymarkers[sigmai+1],markevery=mymarkevery,linewidth = mylinewidth);
    print(">>> T+ max = ",np.max(Tvals[pair[0]])," at Ki = ",xvals[np.argmax(Tvals[pair[0]])]);
    print(">>> T- max = ",np.max(Tvals[pair[1]])," at Ki = ",xvals[np.argmax(Tvals[pair[1]])]);
    axes[0].set_ylim(-lower_y,1.0);
    axes[0].set_ylabel(r'$T_\alpha$', fontsize = myfontsize);
    
    # plot p2 at diff theta
    numtheta = 9;
    thetavals = np.linspace(0,np.pi,numtheta);
    thetais = [0,1,2,8];
    endthetavals = [];
    for thetai in thetais:
        yvals = [];
        for xi in range(len(xvals)):
            yvals.append(p2(Tvals[sourcei,xi],Tvals[pair[0],xi],thetavals[thetai]));
        axes[1].plot(xvals, yvals,color = get_color(thetai,numtheta),linewidth = mylinewidth);
        endthetavals.append(np.copy(yvals)[-1]);
        print(thetavals[thetai]);
    print(endthetavals);

    # plot analytical FOM
    axes[1].plot(xvals, np.sqrt(Tvals[sourcei]*Tvals[pair[0]]), color = accentcolors[0], marker=mymarkers[0],markevery=mymarkevery, linewidth = mylinewidth)
    print(">>> p2 max = ",np.max(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))," at Ki = ",xvals[np.argmax(np.sqrt(Tvals[sourcei]*Tvals[pair[0]]))]);

    # label LHS with p2 values
    ax1ylim = (0,1.0);
    axes[1].set_ylim(*ax1ylim);
    axes[1].set_ylabel('$p^2(\\tilde{\\theta})$', fontsize = myfontsize);
    # label thetavals with RHS yticks
    if True:
        axRHS = axes[1].twinx();
        axRHS.tick_params(axis='y');
        axRHS.set_ylim(*ax1ylim);
        axRHS.set_yticks(endthetavals);
        axRHS.set_yticklabels(['0','$\pi/8$','$\pi/4$','$\pi$']);


    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    #for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/double/model12.pdf');
    #plt.show();
    

    
