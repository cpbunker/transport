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
    myxvals = 200; # number of pts on the x axis
    myfontsize = 14;
    from transport.wfm import UniversalColors, UniversalAccents, ColorsMarkers, AccentsMarkers, UniversalMarkevery, UniversalPanels;
    mylinewidth = 1.0;
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"text.usetex": True}) 
    
    # tight binding parameters
    u0val = 0.0; # in-cell energies are u_0 + u, u_0 - u;
    vval = -1.0; # intradimer hopping, fixed reference param akin to tl
    wvals = np.array([vval, float(sys.argv[2])]);
    uvals = np.array([0.0,  float(sys.argv[3])]);

    # set up figure
    mustrings = ["A","B"];
    n_mu_dof = len(mustrings); # runs over d orbitals and p orbitals
    
    # iterative parameters
    n_iterations = 21 # 201 #raise Exception("change all niter to conv_tol");
    imag_pt_E = 1e-2;
    
if(case in ["dispersion","dispersion_pdf"]): 

    # set up figure
    nrow, ncol = 2, 2;
    
    if("_pdf" in case):
        fig, axes = plt.subplots(nrow, ncol, sharey=True, gridspec_kw = {"height_ratios":[1.0, 0.001]});
        fig.set_size_inches(ncol*3.5, 3);
    else:
        fig, axes = plt.subplots(nrow, ncol, sharey=True);
        fig.set_size_inches(ncol*3.5, nrow*3);
    dispax, dosax, sdosax, veloax = axes[0,0], axes[0,1], axes[1,0], axes[1,1];
    
    # format figure
    for rowi in range(nrow):
        axes[rowi,0].set_ylabel("$E$", fontsize=myfontsize);
    dispax.set_xlabel("$ka/\pi$", fontsize=myfontsize);
    dispax.set_xlim(0.0,1.0);
    dosax.set_xlabel("$\\rho$", fontsize=myfontsize);
    dosax.set_xlim(0.0,10.0);
    sdosax.set_xlabel("Right Lead Surface DOS", fontsize=myfontsize);
    veloax.set_xlabel("Right Lead Velocities", fontsize=myfontsize);
    
    # we will compare different Rice-mele parameter values
    wvals = np.linspace(wvals[0], wvals[-1], 5);
    wcolors = matplotlib.colormaps['bwr'](np.linspace(0,1,1+len(wvals)));
    uvals = float(sys.argv[3])*np.ones_like(wvals);
    dispax.set_title("$u = {:.2f}, v = {:.2f}$".format(uvals[0], vval));

    # run over different RM params
    for wi in range(len(wvals)):

        # from tight binding parameters, construct diatomic matrices
        h00 = np.array([[u0val+uvals[wi], vval], [vval,u0val-uvals[wi]]]);
        h01 = np.array([[0.0, 0.0],[wvals[wi], 0.0]]);
        houtgoing = 1
        title_or_label = wfm.string_RiceMele(h00, h01, energies=False, tex=True)
        print("\n\nRice-Mele "+title_or_label);
        title_or_label = "$w = {:.2f}$".format(wvals[wi]); assert(len(wvals)==5);
        print("h00 =\n",h00);
        print("h01 =\n",h01);
        band_edges = wfm.bandedges_RiceMele(h00, h01);
        Evals = np.linspace(np.min(band_edges), np.max(band_edges), myxvals, dtype=complex);
        kvals = np.linspace(-np.pi,np.pi, myxvals, dtype=float);
        Evals_forkvals = wfm.dispersion_RiceMele(h00, h01, kvals);    

        #### diatomic case -- closed form solution
        ####
        
        # dispersion
        for bandi in range(np.shape(Evals_forkvals)[0]):
            if(bandi==0): label_fordispax = title_or_label[:];
            else: label_fordispax = "_";
            dispax_line2d = dispax.plot(kvals/np.pi, Evals_forkvals[bandi],color=wcolors[wi],label=label_fordispax);

        # continuum density of states
        Evals_gradient = np.array([np.gradient(Evals_forkvals[0],kvals),
                                   np.gradient(Evals_forkvals[1],kvals)]); # handles both bands at once
        for bandi in range(np.shape(Evals_forkvals)[0]):
            dosax.plot(2/np.pi*abs(1/Evals_gradient[bandi]),Evals_forkvals[bandi],color=wcolors[wi]);
        
        # surface density of states
        sdos_dia = np.zeros_like(Evals, dtype=float); 
        for Evali in range(len(Evals)):
            sdos_dia[Evali] =(-1/np.pi)*np.imag(wfm.g_RiceMele(h00,h01,Evals[Evali],houtgoing)[0,0]); 
        sdosax.plot(sdos_dia,np.real(Evals),color=wcolors[wi],label=title_or_label);

        # velocities from self energy
        velocities_Sigma = np.zeros_like(sdos_dia);
        for Evali in range(len(Evals)):
            SigmaRmat = np.matmul(h01, np.matmul( wfm.g_RiceMele(h00,h01,Evals[Evali],houtgoing), np.conj(h01).T));
            velocities_Sigma[Evali] = -2*np.imag(SigmaRmat)[1,1]; # <- get B,B component
        veloax.plot(velocities_Sigma,np.real(Evals),color=wcolors[wi]); 

        if(False): # monatomic case -- closed form solution
            sdos_closed = np.zeros_like(Evals,dtype=float); # NB lack of mu dofs
            for Evali in range(len(Evals)): 
                sdos_closed[Evali] =(-1/np.pi)*np.imag(wfm.g_closed(h00[:1,:1],h01[1:,:1], Evals[Evali], houtgoing))[0,0];
            sdosax.plot(sdos_closed,np.real(Evals),label="$\infty$ (monatomic cell)",color=UniversalAccents[0], marker=AccentsMarkers[0],markevery=UniversalMarkevery);
            # monatomic velocities
            velocities_closed = 2*abs(vval)*np.sqrt(1-np.power(np.real(Evals/(2*abs(vval))),2))
            veloax.plot(velocities_closed, np.real(Evals), color=UniversalAccents[0], marker=AccentsMarkers[0],markevery=UniversalMarkevery);
            
    # truncate axes
    if("_pdf" in case):
        fig.delaxes(sdosax);
        fig.delaxes(veloax);
        
    # show
    dispax.legend();
    plt.tight_layout();
    
    # save to pdf, if asked
    savename = "/home/cpbunker/Desktop/FIGS_Cicc_with_DMRG/dispersion_RiceMele.pdf";
    if("_pdf" in case):
        print("Saving to "+savename);
        plt.savefig(savename);
    else:
        plt.show();

elif(case in ["sdos_conv","sdos_log","sdos_adapt"]):

    # control convergence 
    # code will determine n_iterations automatically for each E
    conv_tol = 1e-3; # convergence tolerance for iterative gf scheme

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

elif(case in ["giter"]):  

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

        # energy values lie in a band determined by Rice-Mele params
        assert(u0==0);
        band_edges = np.array([np.sqrt(uvals[taui]**2+(-tl+taus[taui])**2),
                               np.sqrt(uvals[taui]**2+(-tl-taus[taui])**2)]);
        Evals = np.linspace(np.min(-band_edges), np.max(band_edges), 199, dtype=complex);     

        # show where the bottom of the valence band is
        for edge in [np.min(-band_edges),np.max(-band_edges), np.min(band_edges), np.max(band_edges)]:
            axes[taui*n_mu_dof,0].axvline(edge, color="gray",linestyle="dashed");


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
            gsurf_dia[Evali] = wfm.g_RiceMele(h00,h01, Evals[Evali], houtgoing)[0,0]; # get AA element
        if(plot_dia): # plot diatomic
            axes[taui*n_mu_dof,0].plot(np.real(Evals),np.real(gsurf_dia),color=accentcolors[1], linestyle="solid",label="$\infty$ (diatomic cell)");
            axes[taui*n_mu_dof,0].plot(np.real(Evals),np.imag(gsurf_dia),color=accentcolors[1], linestyle="dashed");
            axes[taui*n_mu_dof+1,0].plot(np.real(Evals), np.nan*np.ones_like(np.real(Evals)), color=accentcolors[1], linestyle="solid",label="$\infty$ (diatomic cell)"); # for legend

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
                          color=UniversalColors[0], linestyle="solid", label=str(ith_iter));
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.imag(gsurf[mu,mup]), 
                          color=UniversalColors[0], linestyle="dashed");
                        axes[taui*n_tau_dof+mu,mup].plot(np.real(Evals), np.real(last_change[mu,mup]),
                          color=mycolors[1], linestyle="solid", label="$\Delta_{"+str(ith_iter)+"}$");      

    # legend
    axes[1,0].legend(title="# iterations");

    # format
    for rowi in range(np.shape(axes)[0]):
        for coli in range(np.shape(axes)[1]):
            axes[rowi, coli].set_ylim(-2.0,2.0);
            axes[rowi, coli].axhline(0.0, color="gray", linestyle="dotted");
            axes[rowi, coli].set_ylabel("$\langle "+mustrings[rowi%2]+"| \mathbf{g}_{00}|"+mustrings[coli%2]+" \\rangle$");
    for ax in axes[-1]: 
        ax.set_xlabel("$E$");
        #ax.set_xlim(min(np.real(Evals)), max(np.real(Evals)));
    for taui in range(len(taus)): 
        axes[taui*n_mu_dof,0].set_title("$v =${:.2f}$, u=${:.2f}$, \eta =${:.0e}"
         .format(taus[taui],uvals[taui], imag_pt_E)+", Re[$g$]=solid, Im[$g$]=dashed");

    # show
    plt.tight_layout();
    plt.show();

else: raise NotImplementedError("case = "+case);
