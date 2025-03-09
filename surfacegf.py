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
    mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkmagenta","darkgray", "darkcyan"];
    accentcolors = ["black","red"];
    mymarkers = ["+","o","^","s","d","*","X"];
    mymarkevery = (40, 40);
    mylinewidth = 1.0;
    plt.rcParams.update({"font.family": "serif"})
    #plt.rcParams.update({"text.usetex": True}) 
    
    # tight binding parameters
    tl = 1.0;
    # in-cell energies are u_0 + u, u_0 - u;
    u0 = 0.0;
    uvals = tl*np.array([0.0, 0.0]); # needs to be zero in the tau = -tl case
    n_iterations = int(sys.argv[2]);
    logKlims = (-4,-1)
    Evals = np.linspace(-2*tl, 2*tl, 399, dtype=complex);
    imag_pt_E = float(sys.argv[3]);
    taus = np.array([-tl, -1.095*tl]);
    is_topological = False;
    if(is_topological): 
        taus = np.array([-tl, tl]); 
        uvals = tl*np.array([0.0, 0.25]); 
    plot_dia = True;

    # set up figure
    mustrings = ["A","B"];
    n_mu_dof = len(mustrings); # runs over d orbitals and p orbitals
    n_tau_dof = len(taus); # display results for tau = -t and tau = +t
    
if(case in ["giter"]):  
    band_widen = 0;
    Evals = np.linspace(-tl*(2+band_widen), tl*(2+band_widen), 199, dtype=complex);     

    # set up figure
    nrow = n_tau_dof*n_mu_dof;
    ncol = 1;
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True);
    if(ncol==1): axes = np.array([[ax] for ax in axes]);

    fig.set_size_inches(n_mu_dof*4.0, n_tau_dof*n_mu_dof*1.8);

    # run over tau values
    for taui in range(len(taus)):

        # from tight binding parameters, construct matrices that act on mu dofs
        h00 = np.array([[u0+uvals[taui], taus[taui]], [taus[taui], u0-uvals[taui]]]);
        h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
        houtgoing = 1; # tells code it is right lead
        print("\n\nRice-Mele v = {:.2f}, u = {:.2f}".format(taus[taui],uvals[taui]));
        print("h00 =\n",h00);
        print("h01 =\n",h01);

        # monatomic cell closed-form result
        gsurf_exact = np.zeros_like(Evals); # NB lack of mu dofs
        for Evali in range(len(Evals)): 
            # NB the off-diag element is -tl, not tl
            gsurf_exact[Evali]=wfm.g_closed(h00[:1,:1],h01[1:,:1],Evals[Evali],houtgoing)[0,0];
        if(taui == 0): # plot monatomic
            axes[taui*n_mu_dof,0].plot(np.real(Evals),np.real(gsurf_exact), color=accentcolors[0], linestyle="solid",label="$\infty$ (monatomic cell)");
            axes[taui*n_mu_dof,0].plot(np.real(Evals),np.imag(gsurf_exact), color=accentcolors[0], linestyle="dashed");
            axes[taui*n_mu_dof+1,0].plot(np.real(Evals), np.nan*np.ones_like(np.real(Evals)), color=accentcolors[0],linestyle="solid",label="$\infty$ (monatomic cell)"); #legend
        del gsurf_exact;

        # *diatomic model* (Rice-Mele) closed-form result
        gsurf_dia = np.zeros_like(Evals); # NB lack of mu dofs
        for Evali in range(len(Evals)):
            gsurf_dia[Evali] = wfm.g_RM(h00,h01, Evals[Evali], houtgoing)[0,0]; # get AA element
        if(plot_dia): # plot diatomic
            axes[taui*n_mu_dof,0].plot(np.real(Evals),np.real(gsurf_dia),color=accentcolors[1], linestyle="solid",label="$\infty$ (diatomic cell)");
            axes[taui*n_mu_dof,0].plot(np.real(Evals),np.imag(gsurf_dia),color=accentcolors[1], linestyle="dashed");
            axes[taui*n_mu_dof+1,0].plot(np.real(Evals), np.nan*np.ones_like(np.real(Evals)), color=accentcolors[1], linestyle="solid",label="$\infty$ (diatomic cell)"); # for legend
            
        # show where the bottom of the valence band is
        assert(u0==0);
        band_edge_pm =[np.sqrt(uvals[taui]*uvals[taui]+np.power(-tl+taus[taui],2)),
                           np.sqrt(uvals[taui]*uvals[taui]+np.power(-tl-taus[taui],2))];
        axes[taui*n_mu_dof,0].axvline(np.min(band_edge_pm), color="gray",linestyle="dashed");


        #
        # **iterative **
        #
        # right lead surface Green's function
        # exists as a matrix acting on mu dofs
        gsurf = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);

        # iter over right lead surface Green's function iterations
        # i.e. \textbf{g}_{00}^{(i)}, the i^th recursive solution to the self-consistent equation
        last_iter = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        last_change = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        # NB this is the right lead, so we need outgoing=1
        for ith_iter in range(n_iterations):
            # energy values
            for Evali in range(len(Evals)):
                gsurf[:,:,Evali] = wfm.g_ith(h00, h01, Evals[Evali], houtgoing, 
                    imag_pt_E, ith_iter, last_iter[:,:,Evali]); 
                last_change[:,:,Evali] = abs(gsurf[:,:,Evali]-last_iter[:,:,Evali]);
                last_iter[:,:,Evali] = 1*gsurf[:,:,Evali]; # for passing to next iteration
            
            # plot results for this iteration
            if(ith_iter == n_iterations-1):
                for mu in range(n_mu_dof):
                    for mup in [0]: # range(n_mu_dof):
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.real(gsurf[mu,mup]), 
                          color=mycolors[0], linestyle="solid", label=str(ith_iter));
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.imag(gsurf[mu,mup]), 
                          color=mycolors[0], linestyle="dashed");
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.real(last_change[mu,mup]),
                          color=mycolors[1], linestyle="solid", label="$\Delta_{"+str(ith_iter)+"}$");      

    # legend
    axes[1,0].legend(title="# iterations");

    # format
    for rowi in range(np.shape(axes)[0]):
        for coli in range(np.shape(axes)[1]):
            #axes[rowi, coli].set_ylim(-2.0,2.0);
            axes[rowi, coli].axhline(0.0, color="gray", linestyle="dotted");
            axes[rowi, coli].set_ylabel("$\langle "+mustrings[rowi%2]+"| \mathbf{g}_{00}|"+mustrings[coli%2]+" \\rangle$");
    for ax in axes[-1]: 
        ax.set_xlabel("$E$");
        ax.set_xlim(min(np.real(Evals)), max(np.real(Evals)));
    for taui in range(len(taus)): 
        axes[taui*n_mu_dof,0].set_title("$v =${:.2f}$, u=${:.2f}$, \eta =${:.0e}"
         .format(taus[taui],uvals[taui], imag_pt_E)+", Re[$g$]=solid, Im[$g$]=dashed");

    # show
    plt.tight_layout();
    plt.show();
    
elif(case in ["sdos"]): 

    # set up figure
    nrow, ncol = n_tau_dof, 2;
    fig, axes = plt.subplots(nrow, ncol, sharex=True);
    fig.set_size_inches(ncol*2*3.6, nrow*3.6);

    # run over tau values
    for taui in range(len(taus)):

        # right lead surface Green's function
        # exists as a matrix acting on mu dofs
        gsurf = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
    
        # from tight binding parameters, construct matrices that act on mu dofs
        h00 = np.array([[u0+uvals[taui], taus[taui]], [taus[taui],u0-uvals[taui]]]);
        h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
        houtgoing = 1
        print("\n\nRice-Mele v = {:.2f}, u = {:.2f}".format(taus[taui],uvals[taui]));
        print("h00 =\n",h00);
        print("h01 =\n",h01);

        if(taui == 0): # plot closed-form solution, monatomic case
            sdos_closed = np.zeros_like(Evals,dtype=float); # NB lack of mu dofs
            for Evali in range(len(Evals)): 
                sdos_closed[Evali] =(-1/np.pi)*np.imag(wfm.g_closed(h00[:1,:1],h01[1:,:1], Evals[Evali], houtgoing))[0,0];
            axes[taui,0].plot(np.real(Evals),sdos_closed,color=accentcolors[0], label="$\infty$ (monatomic cell)");
            axes[taui,1].plot(np.real(Evals),2*tl*np.sqrt(1-np.power(np.real(Evals/(2*tl)),2)), color=accentcolors[0]);

        # plot closed-form solution, diatomic case
        sdos_dia = np.zeros_like(Evals, dtype=float); 
        for Evali in range(len(Evals)):
            sdos_dia[Evali] =(-1/np.pi)*np.imag(wfm.g_RM(h00,h01, Evals[Evali], houtgoing)[0,0]); 
        axes[taui,0].plot(np.real(Evals),sdos_dia,color=accentcolors[1], linestyle="solid",label="$\infty$ (diatomic cell)");
        # diatomic velocities
        axes[taui,1].plot(np.real(Evals),sdos_dia*2*np.pi*tl*tl,color=accentcolors[1], linestyle="solid");

        # velocities direct from derivative of diatomic E(k)
        velocities_deriv = np.zeros_like(sdos_dia);
        for Evali in range(len(Evals)):
            velocities_deriv[Evali] = wfm.velocity_RM(h00,h01,Evals[Evali]);
        axes[taui,1].plot(np.real(Evals),velocities_deriv,color=mycolors[-1], linestyle="solid", marker="s"); 

        # show where the bottom of the valence band is
        assert(u0==0);
        band_edge_pm =[np.sqrt(uvals[taui]*uvals[taui]+np.power(-tl+taus[taui],2)),
                           np.sqrt(uvals[taui]*uvals[taui]+np.power(-tl-taus[taui],2))];
        for ax in axes[taui]: 
            ax.axvline(np.min(band_edge_pm), color="gray",linestyle="dashed");

        #
        # **iterative **
        #
        # i.e. \textbf{g}_{00}^{(i)}, the i^th recursive solution to the self-consistent equation
        last_iter = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        last_change = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        # NB this is the right lead, so we need outgoing=1
        for ith_iter in range(n_iterations):
            # energy values
            for Evali in range(len(Evals)):
                gsurf[:,:,Evali] = wfm.g_ith(h00, h01, Evals[Evali], houtgoing,
                    imag_pt_E, ith_iter, last_iter[:,:,Evali]); 
                last_change[:,:,Evali] = gsurf[:,:,Evali]-last_iter[:,:,Evali];
                last_iter[:,:,Evali] = 1*gsurf[:,:,Evali]; # for passing to next iteration
    
        # get density of states
        dos = (-1/np.pi)*np.imag(gsurf)[0,0]; # real-valued function of E
        axes[taui,0].plot(np.real(Evals), dos, color=mycolors[0], label=str(n_iterations-1));

        # get change in density of states at last iteration
        dos_change = (-1/np.pi)*np.imag(last_change)[0,0]; # real-valued function of E
        axes[taui,0].plot(np.real(Evals), dos_change, color=mycolors[1], label="$\Delta_{"+str(n_iterations-1)+"}$");

        # get velocities
        velocities_dos = dos*(-np.pi)*tl*tl*(-2); # -2Im[\Sigma]
        axes[taui,1].plot(np.real(Evals), velocities_dos, color=mycolors[0]);
        
    # legend
    axes[0,0].legend(title="# iterations");

    # format
    for coli in range(ncol):
        axes[-1,coli].set_xlabel("$E$", fontsize=myfontsize);
        axes[-1,coli].set_xlim(min(np.real(Evals)), max(np.real(Evals)));
    for taui in range(len(taus)): 
        axes[taui,0].set_ylabel("Right Lead Surface DOS", fontsize=myfontsize);
        axes[taui,1].set_ylabel("Right Lead Velocities", fontsize=myfontsize);
        axes[taui,0].set_title("$v =${:.2f}$, u=${:.2f}$, \eta =${:.0e}"
         .format(taus[taui],uvals[taui], imag_pt_E), fontsize=myfontsize);

    # show
    plt.tight_layout();
    plt.show(); 

elif(case in ["sdos_conv","sdos_log","sdos_adapt"]):

    # control convergence 
    # code will determine n_iterations automatically for each E
    conv_tol = float(sys.argv[4]); # convergence tolerance for iterative gf scheme

    # set up figure
    taus = taus[:1] # skip other tau, delta values
    subrows = 4; 
    fig, axes = plt.subplots(subrows, 1, sharex=True);
    fig.set_size_inches(2*3.6, subrows*1.8);

    # logarithmic energies
    if(case in ["sdos_log", "sdos_adapt"]):
        Kvals = np.logspace(*logKlims,myxvals, dtype=complex);
        Evals = Kvals - 2*tl;
    else:
        Kvals = Evals + 2*tl;
    # whether to scale imag pt of E with increasing energies
    if(case in ["sdos_adapt"]): imEvals = imag_pt_E*Kvals;
    else: imEvals = imag_pt_E*np.ones((len(Kvals),),dtype=float);

    # run over tau values
    for taui in range(len(taus)):

        # right lead surface Green's function
        # exists as a complex matrix acting on mu dofs
        gsurf = np.zeros((n_mu_dof, n_mu_dof, len(Kvals)), dtype=complex);

        # right lead surface density of states
        # real-valued function of E, but we store each iteration to compare
        dos = np.zeros((n_iterations, len(Kvals)), dtype=float);
    
        # from tight binding parameters, construct matrices that act on mu dofs
        h00 = np.array([[u0+uvals[taui], taus[taui]], [taus[taui], u0-uvals[taui]]]);
        h01 = np.array([[0.0, 0.0],[-tl, 0.0]]);
        print("\n\nRice-Mele v = {:.2f}, u = {:.2f}".format(taus[taui],uvals[taui]));
        print("h00 =\n",h00);
        print("h01 =\n",h01);

        # iter over right lead surface Green's function iterations
        # i.e. \textbf{g}_{00}^{(i)}, the i^th recursive solution to the self-consistent equation
        last_iter = np.zeros((n_mu_dof, n_mu_dof, len(Evals)), dtype=complex);
        # NB this is the right lead, so we need outgoing=1
        outgoing = 1;
        for ith_iter in range(n_iterations):
            # energy values
            for Evali in range(len(Evals)):
                gsurf[:,:,Evali] = wfm.g_ith(h00, h01, Evals[Evali], outgoing,
                    imEvals[Evali], ith_iter, last_iter[:,:,Evali]); 
                last_iter[:,:,Evali] = 1*gsurf[:,:,Evali]; # for passing to next iteration

            # get density of states at this iteration
            dos[ith_iter,:] = (-1/np.pi)*np.imag(gsurf)[0,0]; # real-valued func of E
    
            # plot
            ith_to_plot = [6*n_iterations//8,7*n_iterations//8,n_iterations-1];
            if(ith_iter in ith_to_plot):
                ith_color = ith_to_plot.index(ith_iter);

                # SDOS itself
                axes[0].plot(np.real(Kvals), dos[ith_iter], color=mycolors[ith_color], label=str(ith_iter));

                # \Delta SDOS
                axes[1].plot(np.real(Kvals), abs((dos[ith_iter]-dos[ith_iter-1])/dos[ith_iter]), color=mycolors[ith_color],label=str(ith_iter));
                # avg this metric over iterations
                # assert False

        # instead of getting surface gf after given number of iterations, 
        # force it to comply with  convergence tolerance
        dos_tol = np.zeros((len(Kvals),), dtype=float); 
        dos_tol_change = np.zeros((len(Kvals),), dtype=float);
        dos_tol_nmax = np.zeros((len(Evals),), dtype=int); 
        for Evali in range(len(Evals)):
            g_iter_output = wfm.g_iter(h00, h01, Evals[Evali], outgoing, 
                              imEvals[Evali], conv_tol, full = True);
            dos_tol[Evali] = (-1/np.pi)*np.imag(g_iter_output[0][0,0]);
            dos_tol_change[Evali] = g_iter_output[1]; # already dos
            dos_tol_nmax[Evali] = 1*g_iter_output[2];
        axes[0].plot(np.real(Kvals), dos_tol, color=mycolors[-1], label="tol");
        axes[1].plot(np.real(Kvals), dos_tol_change, color=mycolors[-1], label="tol");
        axes[2].plot(np.real(Kvals), dos_tol_nmax, color=mycolors[-1], label="tol");
        
    # plot closed-form soln
    dos_closed = np.zeros((len(Kvals),), dtype=float); # NB lack of mu dofs
    eye_like = np.eye(1);
    for Evali in range(len(Evals)): 
        dos_closed[Evali] = (-1/np.pi)*np.imag(wfm.g_closed(u0*eye_like, -tl*eye_like,Evals[Evali],1))[0,0];
    axes[0].plot(np.real(Kvals), dos_closed, color=accentcolors[0], label="$\infty$");
    axes[3].plot(np.real(Kvals), abs((dos_closed-dos_tol)/dos_closed), color=mycolors[-1]);
    
    # legend
    axes[0].legend(title="# iterations");

    # format
    if(case in ["sdos_log", "sdos_adapt"]):
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlim(10**(logKlims[0]), 10**(logKlims[1]));
        axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    else:
        axes[-1].set_xlabel("$K_i$", fontsize=myfontsize);
        axes[-1].set_xlim(min(np.real(Kvals)), max(np.real(Kvals)));

    axes[0].set_ylabel("SDOS", fontsize=myfontsize);
    axes[0].set_title("$v =${:.2f}$, u=${:.2f}$, \eta =${:.0e}"
      .format(taus[taui],uvals[taui], imag_pt_E)
      +", tol={:.0e}".format(conv_tol), fontsize=myfontsize);
    axes[1].set_ylabel("$\Delta$SDOS", fontsize=myfontsize);
    axes[2].set_ylabel("$n_{iterations}$", fontsize=myfontsize);
    axes[3].set_ylabel("Error", fontsize=myfontsize);
    axes[3].set_ylim(0.0,0.1);

    # show
    plt.tight_layout();
    plt.show();    

else: raise NotImplementedError("case = "+case);
