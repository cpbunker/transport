'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering from a rectangular potential barrier, with
different well heights on either side (so that scattering is inelastic)
solved in time-dependent QM using bardeen theory method in transport/bardeen
'''

from transport import bardeen

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 3;
save = True;

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

#################################################################
#### spinless 1D IETS

# tight binding params
n_loc_dof = 1; 
tLR = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tLR;
VL = 0.0*tLR;
Vinfty = 0.5*tLR;
NLR = 200;
Ninfty = 20;
NC = 11;

# cutoffs
Ecut = 0.1;
error_lims = (0,50);

# T vs VR
if True:

    VRvals = np.array([-0.005*tLR,-0.001*tLR,0.001*tLR,0.005*tLR]);
    numplots = len(VRvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1.0*tLR;
    VC = 0.4*tLR;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print_H_j(HC);

    # bardeen results for heights of barrier covering well
    for VRvali in range(len(VRvals)):
        VLprime, VRprime = 1.0*Vinfty, 1.0*Vinfty;
        # VLprime = VRvals[VRi]+Vinfty; # left cover needs to be higher from view of right well
        assert(not np.any(np.ones((len(VC)),)[np.diagonal(VC) < np.diagonal(VRvals[VRvali])]));
        
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        # returns two arrays of size (n_loc_dof, n_left_bound)
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR,
                                      Vinfty, VL, VLprime, VRvals[VRvali], VRprime,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=Ecut,interval = 1e-2, verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VL, VRvals[VRvali], NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VL, VRvals[VRvali], HC, Evals,verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

         # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[VRvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[VRvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright = axes[VRvali].twinx();
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 
            axright.set_ylim(*error_lims);
            
        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axes[VRvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[VRvali].set_title("$V_R' = "+str(VRvals[VRvali][0,0])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

        # save
        folder = "data/bardeen/run_bard_step/vsVR/"
        if(save):
            print("Saving data to "+folder);
            np.save(folder+"VR_{:.3f}_Tvals.npy".format(VRvals[VRvali][0,0]), Tvals);
            np.save(folder+"VR_{:.3f}_Tvals_bench.npy".format(VRvals[VRvali][0,0]), Tvals_bench);
            np.savetxt(folder+"VR_{:.3f}_info.txt".format(VRvals[VRvali][0,0]),
                       np.array([tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VRvals[VRvali][0,0], VRprime[0,0], Ninfty, NLR, NLR]),
                       header = "tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VRvals[VRvali][0,0], VRprime[0,0], Ninfty, NLR, NLR");
            np.savetxt(folder+"VR_{:.3f}_HC.txt".format(VRvals[VRvali][0,0]),HC[:,:,0,0], fmt="%.4f");

        #### end loop over VRvals

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();





