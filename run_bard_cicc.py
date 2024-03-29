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
#np.set_printoptions(precision = 4, suppress = True);
verbose = 3;
save_figs = False;

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

#################################################################
#### matrix elements are the right barrier being removed and Kondo term
#### being added to the central region

# constructing the hamiltonian
def diag_ham(params, S) -> np.ndarray:
    '''
    '''
    for el in params:
        if( not isinstance(el, float)): raise TypeError;
        
    D1, D2, J12, JK1, JK2 = params;
    D = (D1+D2)/2;
    DeltaD = D1-D2;
    h = np.eye(3)*(2*S*S*D + (S*S-S)*J12);
    h += np.array([[ (1-2*S)*D + S*J12, (S-1/2)*DeltaD, 0],
                   [ (S-1/2)*DeltaD, (1-2*S)*D - S*J12, 0],
                   [ 0, 0, S*J12]]);
    h += (JK1/2)*np.array([[S-1/2,1/2, np.sqrt(S)],
                           [1/2,S-1/2,-np.sqrt(S)],
                           [np.sqrt(S),-np.sqrt(S),-S]]);
    h += (JK2/2)*np.array([[S-1/2,-1/2,np.sqrt(S)],
                           [-1/2,S-1/2,np.sqrt(S)],
                           [np.sqrt(S),np.sqrt(S),-S]]);
    return h;

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

# tight binding params
n_loc_dof = 3; 
tLR = 1.0*np.eye(n_loc_dof); # normalized to one, physically ~ 100 meV
tinfty = 1.0*tLR;
VLR = 0.0*tLR;
Vinfty = 0.5*tLR;
NLR = 200;
Ninfty = 20;

# cutoffs
Ecut = 0.1;
error_lims = (0,20);

###############################################
# matrix elements are the right barrier being removed only
# the Kondo term is included in HL and HR
# therefore physical basis != eigenbasis of Sz

# T vs tvac
if True:
    
    # alpha -> beta
    alphas = [0,1,2];
    alpha_strs = ["\downarrow,s,s","\\uparrow,s,s-1","\\uparrow,s-1,s"];
    
    # plotting
    plot_alpha = True;
    if(plot_alpha):
        indvals = np.array([0.8*tLR]);
        nplots_x = len(alphas);
        nplots_y = len(alphas);
    else:
        indvals = np.array([1.0*tLR,0.8*tLR,0.2*tLR]);
        nplots_x = 1
        nplots_y = len(indvals);
        alpha_initial, alpha_final = 0,0;
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    if(nplots_y == 1 and nplots_x == 1): axes=[axes]
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);
        
    # iter over tC
    for indvali in range(len(indvals)):

        # impurity physics
        JK  =-1.0 *tLR[0,0]/100; # first number is value in meV
        J12 = 0.0 *tLR[0,0]/100;
        Dval= 0.2 *tLR[0,0]/100; # < ---- TODO: get to work when nonzero
        myspinS = 1;

        # central region
        tvac = indvals[indvali]; # hopping between leads and central
        tC = 1.0*tLR; # hopping in central
        VC = 0.4*tLR;
        NC = 3; # < ---- TODO: get to work when > 3

        # modify all potential terms with the noninteracting spin-spin background
        background_diag = diag_ham((Dval, Dval, J12, 0.0, 0.0), myspinS);
        VLR += background_diag;
        Vinfty += background_diag;
        VC += background_diag;

        # fill in the central region
        barrieris = np.array(range(1,NC-1)); # NC-2 site barrier
        impis = [1,NC-2]; # preserve LR symmetry
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=float);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # diagonal in space

                    # add spatial barrier
                    if(NCi in barrieris):
                        HC[NCi,NCj] += VC;
                    else:
                        HC[NCi,NCj] += VLR;

                    # add spin terms
                    if(NCi in impis):
                        JK1, JK2 = 0.0, 0.0; 
                        if(NCi == impis[0]): JK1 = JK;
                        if(NCi == impis[1]): JK2 = JK;
                        # NB D and J12 already accounted for by background term
                        imp_ham_diag = diag_ham((0.0,0.0,0.0,JK1,JK2), myspinS);
                        HC[NCi,NCj] += imp_ham_diag;
                        
                elif(abs(NCi -NCj) == 1): # nn hopping
                    if(NCi in barrieris and NCj in barrieris):
                        HC[NCi,NCj] += -tC;
                    else:
                        HC[NCi,NCj] += -tvac;
        # define alpha basis
        alpha_mat = np.zeros_like(HC[0,0]);
        for impi in impis:
            alpha_mat += np.copy(HC[impi,impi]);
        print("HC =");
        print_H_alpha(HC);
        print("alpha_mat =\n",alpha_mat);

        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime, change_basis
        # where change_basis are coefs that take us from alpha to \tilde{\alpha}
        Evals, Mvals = bardeen.kernel_well_super(tinfty,tLR, tLR, 
                                  Vinfty, VLR, Vinfty, VLR, Vinfty,
                                  Ninfty, NLR, NLR, HC, HC, alpha_mat,                                                
                                  E_cutoff=Ecut,eigval_tol=1e-9,verbose=1);
        # symmetrize
        Evals[0], Evals[1] = (Evals[0]+Evals[1])/2, (Evals[0]+Evals[1])/2;
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # initial and final states
        if plot_alpha:
            for alphai in range(len(alphas)):
                for betai in range(len(alphas)):
                    alpha, beta = alphas[alphai], alphas[betai];
                    # plot based on initial state
                    xvals = np.real(Evals[alphai])+2*tLR[alphai,alphai];
                    axes[alphai,betai].scatter(xvals, Tvals[betai,:,alphai], marker=mymarkers[0],color=mycolors[0]);
                    # % error
                    axright = axes[alphai,betai].twinx();
                    axes[alphai,betai].scatter(xvals, Tvals_bench[betai,:,alphai], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
                    axright.plot(xvals,100*abs((Tvals[betai,:,alphai]-Tvals_bench[betai,:,alphai])/Tvals_bench[betai,:,alphai]),color=accentcolors[1]);            
                    #format
                    if(betai==len(alphas)-1): axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
                    #axright.set_ylim(*error_lims);
                    axes[alphai,betai].set_title("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
                    axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
                    axes[-1,betai].set_xscale('log', subs = []);
                    axes[alphai,0].set_ylabel("$T$");

        else:
            # plot based on initial state
            xvals = np.real(Evals[alpha_initial])+2*tLR[alpha_initial,alpha_initial];
            axes[indvali].scatter(xvals, Tvals[alpha_final,:,alpha_initial], marker=mymarkers[0],color=mycolors[0]);
            # % error
            axright = axes[indvali].twinx();
            axes[indvali].scatter(xvals, Tvals_bench[alpha_final,:,alpha_initial], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha_final,:,alpha_initial]-Tvals_bench[alpha_final,:,alpha_initial])/Tvals_bench[alpha_final,:,alpha_initial]),color=accentcolors[1]);            
            #format
            axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            #axright.set_ylim(*error_lims);
            axes[indvali].set_title("$t_{vac} = "+str(tvac[0,0])+"$", x=0.4, y = 0.7, fontsize=myfontsize);
            axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            axes[indvali].set_ylabel("$T("+alpha_strs[alpha_initial]+"\\rightarrow"+alpha_strs[alpha_final]+")$");
            axes[indvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # show
    fig.suptitle("$N_C = "+str(NC)+",\, J = "+str(JK)+"$");
    plt.tight_layout();
    fname = "figs/bard_cicc/bard_cicc";
    if(not plot_alpha):
        if( (alpha_initial, alpha_final) == (0,0) ): fname +="_nsf.pdf";
        elif( (alpha_initial, alpha_final) == (0,1) ): fname +="_sf.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

