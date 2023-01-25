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

def get_ideal_T(alpha,beta,Es,mytL,myVL,myNC,myJ,myVC = None):
    '''
    Get analytical T for spin-spin scattering, Menezes paper
    '''
    if alpha not in [0,1,2,3]: raise ValueError;
    if beta not in [0,1,2,3]: raise ValueError;
    assert np.min(Es) >= 0; # energies must start at zero
    assert myVL == 0.0;
    #assert myNC == 1;

    if myVC != None: # hacky code to get barrier T's
        mytC = mytL;
        kavals = np.arccos((Es-2*mytL-myVL)/(-2*mytL));
        kappavals = np.arccosh((Es-2*mytC-myVC)/(-2*mytC));
        print("Es:\n", Es[:8]);
        print("kavals:\n", kavals[:8]);
        print("kappavals:\n", kappavals[:8]);

        ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
        ideal_exp = np.exp(-2*myNC*kappavals);
        ideal_T = ideal_prefactor*ideal_exp;
        ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
        ideal_T *= ideal_correction;

        if((alpha==1 and beta==1) or (alpha==2 and beta==2)):
            return np.real(ideal_T);
        else:
            return np.zeros_like(ideal_T);

    kas = np.arccos((Es-2*mytL)/(-2*mytL));
    jprimes = myJ/(4*mytL*kas);
    ideal_Tf = jprimes*jprimes/(1+(5/2)*jprimes*jprimes+(9/16)*np.power(jprimes,4));
    ideal_Tnf = (1+jprimes*jprimes/4)/(1+(5/2)*jprimes*jprimes+(9/16)*np.power(jprimes,4));
    
    if((alpha==1 and beta==1) or (alpha==2 and beta==2)):
        return ideal_Tnf;
    elif((alpha == 1 and beta == 2) or (alpha == 2 and beta == 1)):
        return ideal_Tf;
    else:
        return np.zeros_like(ideal_Tf);

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

    else: raise ValueError("Unsupported s2");

    return h;

#################################################################
#### all possible T_{\alpha -> \beta}

if True:
    # benchmark w/out spin flips
    barrier = True;

    # spin dofs
    alphas = [0,1,2,3];
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
    Jval = -0.4;

    # central region
    tC = 1.0*tL;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): # exchange interaction
                HC[NCi,NCj] = h_kondo(Jval,0.5);
            elif(abs(NCi -NCj) == 1): # nn hopping
                HC[NCi,NCj] = -tC*np.eye(n_loc_dof);
    print_H_alpha(HC);

    # central region prime
    HCprime = np.zeros_like(HC);
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): # exchange interaction
                HCprime[NCi,NCj] = np.diagflat(np.diagonal(HC[NCi,NCj]));
            elif(abs(NCi -NCj) == 1): # nn hopping
                HCprime[NCi,NCj] = -tC*np.eye(n_loc_dof);
    print_H_alpha(HCprime);

    if barrier: # get rid of off diag parts for benchmarking
        HC = np.copy(HCprime);

    # bardeen results for spin flip scattering
    Ninfty = 50;
    NL = 100;
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
                                Ninfty, NL, NR, HC, HCprime,cutoff=HC[0,0,1,1],verbose=verbose);

    # initial and final states
    alphas = [1,2];
    for alpha in alphas:
        for beta in alphas:

            # truncate to bound states and plot
            yvals = np.diagonal(Tvals[beta,:,alpha,:]);
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[alpha,beta].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

            # compare
            VC_barrier = None;
            if barrier: VC_barrier = -Jval/4;
            ideal_Tvals_alpha = get_ideal_T(alpha, beta, xvals, tL[alpha,alpha],VL[alpha,alpha],NC,Jval,myVC = VC_barrier);
            axes[alpha,beta].plot(xvals,np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
            #axes[NLi].set_ylim(0,1.1*max(Tvals[alpha]));

            # error
            if False: #( barrier and (alpha == 1 and beta == 1)):
                axright = axes[alpha,beta].twinx();
                axright.plot(xvals,100*abs((yvals-np.real(ideal_Tvals_alpha))/ideal_Tvals_alpha),color=accentcolors[1]);
                axright.set_ylim(0,50);

            #format
            axes[alpha,beta].set_title("$"+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+"$")
            #axes[alpha,-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);

    # format and show
    axes[-1,-1].set_xscale('log', subs = []);
    plt.tight_layout();
    plt.show();





