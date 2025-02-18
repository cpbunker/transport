from transport import wfm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

import sys

if(__name__=="__main__"):

    # top level
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;
    case = sys.argv[1];

    # fig standardizing
    myxvals = 99; # number of pts on the x axis
    myfontsize = 14;
    mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
    accentcolors = ["black","red"];
    mymarkers = ["+","o","^","s","d","*","X"];
    mymarkevery = (40, 40);
    mylinewidth = 1.0;
    plt.rcParams.update({"font.family": "serif"})
    #plt.rcParams.update({"text.usetex": True}) 
    
    # tight binding parameters
    tl = 1.0;
    Evals = np.linspace(-2*tl+1e-4, 2*tl-1e-4, 199, dtype=complex);
    imag_pt_E = 1e-2; 
    taus = np.array([-tl, 1*tl]); # in-cell hopping either same, or off by (-1)
    # in-cell energies are eps_d, eps_d + \delta \epsilon
    # eps_d is like eps_0_ii in the tau = -tl case, we can freely set it to zero
    eps_d = 0.0;
    deltas = tl*np.array([0.0, 1.5]); # needs to be zero in the tau = -tl case
    n_iterations = 91;

    # set up figure
    mustrings = ["d","p"];
    n_mu_dof = len(mustrings); # runs over d orbitals and p orbitals
    n_tau_dof = len(taus); # display results for tau = -t and tau = +t
    
if(case in ["g_iter"]):       

    # set up figure
    fig, axes = plt.subplots(n_tau_dof*n_mu_dof, n_mu_dof, sharex = True);
    fig.set_size_inches(n_mu_dof*3.6, n_tau_dof*n_mu_dof*1.8);

    # run over tau values
    for taui in range(len(taus)):

        # right lead surface Green's function
        # exists as a matrix acting on mu dofs
        gsurf = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
    
        # from tight binding parameters, construct matrices that act on mu dofs
        h00 = np.array([[eps_d, taus[taui]], [taus[taui], eps_d + deltas[taui]]]);
        h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
        print("\nCase tau = {:.1f}, delta = {:.1f}".format(taus[taui],deltas[taui]));
        print("h00 =\n",h00);
        print("h01 =\n",h01);

        # iter over right lead surface Green's function iterations
        # i.e. \textbf{g}_{00}^{(i)}, the i^th recursive solution to the self-consistent equation
        last_iter = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        for ith_iter in range(n_iterations):
            # energy values
            for Evali in range(len(Evals)):
                gsurf[:,:,Evali] = wfm.g_iter(h00, h01, Evals[Evali], 
                    ith_iter, last_iter[:,:,Evali], imE=imag_pt_E); 
                last_iter[:,:,Evali] = 1*gsurf[:,:,Evali]; # for passing to next iteration
            
            # plot results for this iteration
            if(ith_iter == n_iterations-1):
                for mu in range(n_mu_dof):
                    for mup in range(n_mu_dof):
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.real(gsurf[mu,mup]), 
                          color=mycolors[0], linestyle="solid", label=str(ith_iter));
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.imag(gsurf[mu,mup]), 
                          color=mycolors[0], linestyle="dashed");
       

    # plot the closed form result
    gsurf_closed = np.zeros_like(Evals);
    for Evali in range(len(Evals)): 
        gsurf_closed[Evali] = wfm.g_closed(eps_d, -tl, Evals[Evali]); # NB the off-diag element is -tl, not tl
    axes[0,0].plot(np.real(Evals),np.real(gsurf_closed),color=accentcolors[0], linestyle="solid",label="$\infty$");
    axes[0,0].plot(np.real(Evals),np.imag(gsurf_closed),color=accentcolors[0], linestyle="dashed");
    axes[0,1].plot(np.real(Evals), np.nan*np.ones_like(np.real(Evals)), color=accentcolors[0],      linestyle="solid",label="$\infty$"); # for legend

    # legend
    axes[0,1].legend(title="# iterations");

    # format
    for rowi in range(np.shape(axes)[0]):
        for coli in range(np.shape(axes)[1]):
            axes[rowi, coli].set_ylim(-2.0,2.0);
            axes[rowi, coli].set_ylabel("$\langle "+mustrings[rowi%2]+"| \mathbf{g}_{00}|"+mustrings[coli%2]+" \\rangle$");
    for ax in axes[-1]: ax.set_xlabel("$E$");
    for taui in range(len(taus)): 
        axes[taui*n_mu_dof,0].set_title("$\\tau =${:.1f}$t_l, \delta \\varepsilon=${:.1f}$t_l, \eta =${:.0e}"
         .format(taus[taui],deltas[taui], imag_pt_E));

    # show
    plt.tight_layout();
    plt.show();
    
elif(case in ["dos"]): 

    # set up figure
    fig, axes = plt.subplots(n_tau_dof, 1, sharex = True);
    fig.set_size_inches(3.6, n_tau_dof*3.6);

    # run over tau values
    for taui in range(len(taus)):

        # right lead surface Green's function
        # exists as a matrix acting on mu dofs
        gsurf = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
    
        # from tight binding parameters, construct matrices that act on mu dofs
        h00 = np.array([[eps_d, taus[taui]], [taus[taui], eps_d + deltas[taui]]]);
        h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
        print("\nCase tau = {:.1f}, delta = {:.1f}".format(taus[taui],deltas[taui]));
        print("h00 =\n",h00);
        print("h01 =\n",h01);

        # iter over right lead surface Green's function iterations
        # i.e. \textbf{g}_{00}^{(i)}, the i^th recursive solution to the self-consistent equation
        last_iter = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        for ith_iter in range(n_iterations):
            # energy values
            for Evali in range(len(Evals)):
                gsurf[:,:,Evali] = wfm.g_iter(h00, h01, Evals[Evali], 
                    ith_iter, last_iter[:,:,Evali], imE=imag_pt_E); 
                last_iter[:,:,Evali] = 1*gsurf[:,:,Evali]; # for passing to next iteration
    
        # get density of states
        dos = (-1/np.pi)*np.imag(gsurf); # real-valued
        axes[taui].plot(np.real(Evals), dos[0,0], color=mycolors[0], label=str(n_iterations-1));
        
    # plot exact soln
    exact_dos_case1 = 1/(np.pi*tl)*np.sqrt(1-np.power((np.real(Evals)-eps_d)/(-2*tl),2));
    axes[0].plot(np.real(Evals), exact_dos_case1, color=accentcolors[0], label="$\infty$");
    
    # legend
    axes[0].legend(title="# iterations");

    # format
    axes[-1].set_xlabel("$E$");
    for taui in range(len(taus)): 
        axes[taui].set_ylabel("DOS");
        axes[taui].set_title("$\\tau =${:.1f}$t_l, \delta \\varepsilon=${:.1f}$t_l, \eta =${:.0e}"
         .format(taus[taui],deltas[taui], imag_pt_E));

    # show
    plt.tight_layout();
    plt.show();   

else: raise NotImplementedError("case = "+case);
