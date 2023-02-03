'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) 
benchmarked to exact solution
solved in time-dependent QM using bardeen theory method in transport/bardeen
'''

from transport import bardeen

import numpy as np
import matplotlib.pyplot as plt

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

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def print_H_alpha(H):
    assert(len(np.shape(H)) == 4);
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

def h_kondo(J,s2):
    '''
    Kondo interaction between spin 1/2 and spin s2
    '''
    n_loc_dof = int(2*(2*s2+1));
    h = np.zeros((n_loc_dof,n_loc_dof),dtype=complex);
    if(s2 == 0.5):
        h[0,0] = 1;
        h[1,1] = -1;
        h[2,2] = -1;
        h[3,3] = 1;
        h[1,2] = 2;
        h[2,1] = 2;
        h *= J/4;
    else: raise NotImplementedError;
    return h;

#################################################################
#### all possible T_{\alpha -> \beta}

if True:
    # benchmark w/out spin flips
    barrier = False;

    # iter over J
    Jvals = np.array([-0.5,-1.0,-5.0]);
    nplots = len(Jvals);
    fig, axes = plt.subplots(nrows = nplots, sharex = True);
    fig.set_size_inches(7/2,nplots*3/2);
    alphas = [1,2];
    if barrier:
        alphas=[0];
        Jvals *= (-1);
    alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];    # plotting
    alpha_strs = alpha_strs[alphas[0]:alphas[-1]+1];
    myalpha = 0;
    
    # tight binding params
    n_loc_dof = len(alphas); # spin up and down for each
    tL = 1.0*np.eye(n_loc_dof);
    tinfty = 1.0*tL;
    tR = 1.0*tL;
    Vinfty = 0.5*tL;
    VL = 0.0*tL;
    VR = 0.0*tL;

    # iter over J
    for Jvali in range(len(Jvals)):
        Jval = Jvals[Jvali];

        # central region
        tC = 1.0*tL;
        VC = abs(Jval/4)*tL;
        NC = 5;
        if barrier: NC = 11;
        my_kondo = h_kondo(Jval,0.5)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # exchange interaction
                    if(NCi == NC //2): # exchange right at barrier-well boundary
                        HC[NCi,NCj] += my_kondo;
                    else:
                        HC[NCi,NCj] += 0.0;
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HC[NCi,NCj] += -tC;

        # central region prime
        tCprime = tC;
        HCprime = np.zeros_like(HC);
        kondo_replace = np.diagflat(np.diagonal(my_kondo));
        #kondo_replace = my_kondo;
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # exchange interaction
                    if(NCi == NC //2):#  replace exchange
                        HCprime[NCi,NCj] += kondo_replace;
                    else: 
                        HCprime[NCi,NCj] += 0.0;
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HCprime[NCi,NCj] += -tC;

        # print
        print("HC =");
        print_H_alpha(HC);
        print("HC - HCprime =");
        print_H_alpha(HC-HCprime);

        # bardeen results for spin flip scattering
        Ninfty = 20;
        NL = 200;
        NR = 1*NL;
        ##
        #### Notes
        ##
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        # I am setting VLprime = VRprime = Vinfty for best results according
        # tests performed in run_barrier_bardeen 
        # returns two arrays of size (n_loc_dof, n_left_bound)
        Evals, Tvals = bardeen.kernel(tinfty,tL,tinfty, tR, tinfty,
                                      Vinfty, VL, Vinfty, VR, Vinfty,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=0.1,verbose=1);

        # benchmark
        Tvals_bench = bardeen.benchmark(tL, tR, VL, VR, HC, Evals, verbose=0);

        # plot based on initial state
        xvals = np.real(Evals[myalpha])+2*tL[myalpha,myalpha];
        axes[Jvali].scatter(xvals, Tvals[myalpha], marker=mymarkers[0], color=mycolors[0]);

        # % error
        axright = axes[Jvali].twinx();
        axes[Jvali].scatter(xvals, Tvals_bench[myalpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
        axright.plot(xvals,100*abs((Tvals[myalpha]-Tvals_bench[myalpha])/Tvals_bench[myalpha]),color=accentcolors[1]); 

        #format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        #axright.set_ylim(0,50);
        fig.suptitle("$"+alpha_strs[myalpha]+"\\rightarrow $");
        axes[Jvali].set_title("$J = "+str(Jval)+"$");
        my_ylim = (0,0.5);
        if barrier: my_ylim = (0,0.1);
        #axes[Jvali].set_ylim(*my_ylim);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    plt.tight_layout();
    plt.show();





