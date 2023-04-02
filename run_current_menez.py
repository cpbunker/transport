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
myxvals = 49;
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
#### I vs Vb

if True:

    # alpha -> beta
    alphas = [1,2];
    alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];    # plotting
    nplots_x = len(alphas);
    nplots_y = len(alphas);
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);
    
    # tight binding params
    n_loc_dof = len(alphas); # spin up and down for each
    tL = 100.0*np.eye(n_loc_dof); # all units meV
    tinfty = 1.0*tL;
    tR = 1.0*tL;
    Vinfty = 2.0*tL;
    VL = 0.0*tL;
    VR = 0.0*tL;
    Jval = -4*tL[0,0];

    # central region
    tC = 1.0*tL; # 100 meV
    #VC = abs(Jval/4)*tL;
    NC = 1;
    my_kondo = h_kondo(Jval,0.5)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];
    #my_kondo = np.diagflat(np.diagonal(my_kondo));
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): 
                if(NCi == NC //2): # exchange in middle of barrier
                    HC[NCi,NCj] += my_kondo;
                else: # buffer zone
                    HC[NCi,NCj] += 0.0; 
            elif(abs(NCi -NCj) == 1): # nn hopping
                HC[NCi,NCj] += -tC;

    # central region prime
    tCprime = tC;
    HCprime = np.zeros_like(HC);
    kondo_replace = np.diagflat(np.diagonal(my_kondo));
    assert(bardeen.is_alpha_conserving(kondo_replace, n_loc_dof));
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): 
                if(NCi == NC //2):#  replace exchange
                    HCprime[NCi,NCj] += kondo_replace;
                else:  # buffer zone
                    HCprime[NCi,NCj] += 0.0; 
            elif(abs(NCi -NCj) == 1): # nn hopping
                HCprime[NCi,NCj] += -tC;

    # print
    print("HC =");
    print_H_alpha(HC);
    print("HC - HCprime =");
    print_H_alpha(HC-HCprime);

    # matrix elements for spin flip scattering
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
    Evals, Mvals = bardeen.kernel(tinfty,tL,tinfty, tR, tinfty,
                              Vinfty, VL, Vinfty, VR, Vinfty,
                              Ninfty, NL, NR, HC, HCprime,
                              E_cutoff=-Jval/4,verbose=1);

    # current at finite temperature, vs bias
    kBT = (0.01*26); # 1% of room temp (26 meV) in units of t (100 meV)
    muR = (0.5-2)*tL[0,0] #1000/100; # 1 eV, in units of t (100 meV)
    Vbvals = np.linspace(tL[0,0]*(-0.2),tL[0,0]*(0.2),myxvals);
    #print(muR); assert False;
    Ivals = np.empty((n_loc_dof,n_loc_dof,myxvals));
    for Vbi in range(myxvals):
        Ivals[:,:,Vbi] = bardeen.current(Evals, Mvals, muR, Vbvals[Vbi], kBT);

    # initial and final states
    for alphai in range(len(alphas)):
        for betai in range(len(alphas)):
            alpha, beta = alphas[alphai], alphas[betai];

            # plot based on initial state
            axes[betai,alphai].plot(muR+Vbvals, Ivals[betai,alphai], color=mycolors[0], linewidth=mylinewidth);
            axes[betai,alphai].plot(muR+Vbvals, np.gradient(Ivals[betai,alphai],muR+Vbvals), color = mycolors[1], linewidth=mylinewidth);
            axes[betai,alphai].axvline(muR,color=accentcolors[0],linestyle='dashed');
            #format

    # show
    plt.tight_layout();
    plt.show();





