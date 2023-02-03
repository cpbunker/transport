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
mymarkers = ["o","^","s","d","*","X","P"];
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

    # spin dofs
    alphas = [1,2];
    alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];    # plotting
    nplots_x = len(alphas);
    nplots_y = len(alphas);
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);

    # tight binding params
    n_loc_dof = len(alphas); # spin up and down for each
    tL = 1.0*np.eye(n_loc_dof);
    tinfty = 1.0*tL;
    tR = 1.0*tL;
    Vinfty = 0.5*tL;
    VL = 0.0*tL;
    VR = 0.0*tL;
    Jval = -0.7;

    # central region
    tC = 1.0*tL;
    VC = -Jval/4;
    NC = 3;
    if barrier: NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): # exchange interaction
                if(barrier or NCi == NC //2): # spinless barrier on central site
                    HC[NCi,NCj] += VC*np.eye(n_loc_dof);
                else: # kondo exchange right at barrier-well boundary
                    HC[NCi,NCj] += h_kondo(Jval,0.5)[1:3,1:3];
            elif(abs(NCi -NCj) == 1): # nn hopping
                HC[NCi,NCj] = -tC*np.eye(n_loc_dof);

    # central region prime
    tCprime = tC;
    HCprime = np.zeros_like(HC);
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): # exchange interaction
                if(barrier or NCi == NC // 2): # spinless barrier on central site
                    HCprime[NCi,NCj] += VC*np.eye(n_loc_dof);
                else: 
                    pass;
            elif(abs(NCi -NCj) == 1): # nn hopping
                HCprime[NCi,NCj] += -tC*np.eye(n_loc_dof);

    # print
    print("HC - HCprime =");
    print_H_alpha(HC-HCprime);

    # bardeen results for spin flip scattering
    Ninfty = 20;
    NL = 200;
    NR = 1*NL;
    ##
    #### Notes
    ##
    # - bardeen.kernel syntax:
    #       tinfty, tL, tLprime, tR, tRprime,
    #       Vinfty, VL, VLprime, VR, VRprime,
    #       Ninfty, NL, NR, HC,HCprime,
    # - I am setting VLprime = VRprime = Vinfty for best results according
    # run_barrier_bardeen tests
    Evals, Tvals = bardeen.kernel(tinfty,tL,tinfty, tR, tinfty,
                                  Vinfty, VL, Vinfty, VR, Vinfty,
                                Ninfty, NL, NR, HC, HCprime,
                                  E_cutoff=HC[0,0,1,1],verbose=1);

    # benchmark
    Tvals_bench = bardeen.benchmark(tL, tR, VL, VR, HC, Evals, verbose=0);


    # initial and final states
    alphas = [1,2];
    for alphai in range(len(alphas)):
        for betai in range(len(alphas)):
            alpha, beta = alphas[alphai], alphas[betai];

            # truncate to bound states and plot
            yvals = np.diagonal(Tvals[betai,:,alphai,:]);
            yvals_bench = np.diagonal(Tvals_bench[betai,:,alphai,:]);
            xvals = np.real(Evals[alphai])+2*tL[alphai,alphai];
            axes[alphai,betai].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

            # compare
            axes[alphai,betai].plot(xvals, yvals_bench, color=accentcolors[0], linewidth=mylinewidth);

            #format
            axes[alphai,betai].set_title("$"+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+"$");
            my_ylim = (0,0.5);
            if barrier: my_ylim = (0,0.1);
            axes[alphai,betai].set_ylim(*my_ylim);
            axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\,  J = '+str(Jval)+'$',fontsize=myfontsize);

    # format and show
    axes[-1,-1].set_xscale('log', subs = []);
    plt.tight_layout();
    plt.show();





