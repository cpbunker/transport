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

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.legend_handler
import sys
    
#################################################################################
#### utils

def h_cicc_reduced(Jsd, i1, i2, Nunits, unitcell, S) -> np.ndarray:
    '''
    construct the SR hamiltonian over j, mu, sigma dofs
    '''
    assert(isinstance(i1, list) and isinstance(i2, list));
    assert(i1[0] == 1); # SR starts at 1
    assert(i1[-1] < i2[0]);
    N = i2[-1]; # SR ends at N
    
    # get S dot S in the reduced subspace
    # ie Eq (40) in my PRA paper  with JK1=JK2=Jsd only nonzero parameter                        
    h_deltaj1 = (Jsd/2)*np.array([[S-1/2,1/2, np.sqrt(S)],
                           [1/2,S-1/2,-np.sqrt(S)],
                           [np.sqrt(S),-np.sqrt(S),-S]]);
    h_deltaj2 = (Jsd/2)*np.array([[S-1/2,-1/2,np.sqrt(S)],
                           [-1/2,S-1/2,np.sqrt(S)],
                           [np.sqrt(S),np.sqrt(S),-S]]);

    # expand from spin space into unit cell space
    n_spin_dof = len(h_deltaj1);
    h_deltaj1_unit = np.zeros((unitcell*n_spin_dof, unitcell*n_spin_dof),dtype=float);
    h_deltaj1_unit[:n_spin_dof,:n_spin_dof] = h_deltaj1[:,:];
    # ^ only puts S dot S on first orbital of unit cell, mu=A
    h_deltaj2_unit = np.zeros((unitcell*n_spin_dof, unitcell*n_spin_dof),dtype=float);
    h_deltaj2_unit[:n_spin_dof,:n_spin_dof] = h_deltaj2[:,:];
    # insert these local interactions on certain unit cells only
    h_cicc =[];
    for sitei in range(Nunits): # iter over all sites
        if(sitei in i1 and sitei not in i2):
            h_cicc.append(h_deltaj1_unit);
        elif(sitei in i2 and sitei not in i1):
            h_cicc.append(h_deltaj2_unit);
        elif(sitei not in i1 and sitei not in i2):
            h_cicc.append(np.zeros_like(h_deltaj1_unit) );
        else:
            raise Exception;
    h_cicc = np.array(h_cicc, dtype = complex);
    return h_cicc;
    
if(__name__=="__main__"):
    # top level
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;
    case = sys.argv[1];

    # fig standardizing
    myxvals = 99; # number of pts on the x axis
    myfontsize = 14;
    mylinewidth = 1.0;
    from transport.wfm import UniversalColors, UniversalAccents, ColorsMarkers, AccentsMarkers, UniversalMarkevery, UniversalPanels;
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"text.usetex": True})
else: pass;

#################################################################
#### **DIATOMIC UNIT CELL**
#### Rice-Mele model

if(case in ["CB_rhos", "cartoon_rhos", "VB_rhos"]): # entanglement *preservation* vs N, different colors for rho value
    my_unit_cell = 2; # since diatomic

    # Rice-Mele tight binding
    vval = -1.0; # sets energy scale
    wval = float(sys.argv[3]);
    uval = float(sys.argv[4]); 
    Jval = -0.05;
    myspinS = 0.5;
    if(Jval == -0.005): myxvals = 30;

    # Rice-Mele matrices
    n_spin_dof = 3; # spin dofs
    diag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
    diag_base_RM_spin[:n_spin_dof,:n_spin_dof] = uval*np.eye(n_spin_dof);
    diag_base_RM_spin[n_spin_dof:,n_spin_dof:] = -uval*np.eye(n_spin_dof);
    diag_base_RM_spin[:n_spin_dof,n_spin_dof:] = vval*np.eye(n_spin_dof);
    diag_base_RM_spin[n_spin_dof:,:n_spin_dof] = vval*np.eye(n_spin_dof);
    offdiag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
    offdiag_base_RM_spin[n_spin_dof:,:n_spin_dof] = wval*np.eye(n_spin_dof);
    diag_base_nospin = diag_base_RM_spin[::n_spin_dof,::n_spin_dof];
    offdiag_base_nospin = offdiag_base_RM_spin[::n_spin_dof,::n_spin_dof];
    assert(abs(np.sum(np.diagonal(diag_base_nospin))/len(diag_base_nospin)) < 1e-10); # u0 = 0
    band_edges = wfm.bandedges_RiceMele(diag_base_nospin, offdiag_base_nospin)[-2:];
    
    # output Rice-Mele
    title_RiceMele = wfm.string_RiceMele(diag_base_nospin, offdiag_base_nospin, energies=False, tex=True)
    print("\n\nRice-Mele "+title_RiceMele);
    print("h00 =\n",diag_base_nospin);
    print("h01 =\n",offdiag_base_nospin);
                                   
    # channels
    pair = (0,1); # pair[0] = |+> channel, pair[1] = |-> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement
                                          # in this case, elec up, MSQs in triplet, singlet
                                          # source must impinge on A orbital 
    # rhoJa = fixed throughout, thus fixing energy and wavenumber
    if(sys.argv[2]=="vsN"): vsN = True;
    else: vsN = False;
    rhoJvals = np.array([0.5,1.0]);
    if(case in ["cartoon_rhos"]): 
        rhoJvals = np.array([1.5,0.5]); 
        myfontsize = 1.5*myfontsize
                                    
    # return values
    # shaped by fixed rhoJval (color), MSQ-MSQ distance (x axis), spin dofs 
    # Transmission coefficients. Note:
        # we compute only source channel -> source channel scattering, leave the rest as NaNs
        # we evaluate T at SR boundary, namely the B site
    Tvals = np.full((len(rhoJvals),myxvals,n_spin_dof), np.nan, dtype=float);
   
    # Tsummed measures source channel -> all channels transmission
    Tsummed = np.full((len(rhoJvals),myxvals,n_spin_dof), np.nan, dtype=float);
    TpRsummed = np.full((len(rhoJvals),myxvals,n_spin_dof), np.nan, dtype=float); 
    
    # d = number of lattice constants between MSQ 1 and MSQ 2
    if(vsN): kdalims = 0.01*np.pi, 1.01*np.pi; 
    else: kdalims = 0.01*np.pi, 2.1*np.pi; 
    widelimsflag = False;
    try: 
        if(sys.argv[5]=="widelims"): widelimsflag = True;
    except: 
        print(">>> Not flagged to widen kda limits");
    if(widelimsflag): kdalims = 0.01*np.pi, 100*np.pi; 
    print("kda limits = ", kdalims);
    kdavals = np.full((len(rhoJvals),myxvals), np.nan, dtype=float);  
    Distvals = np.full((len(rhoJvals),myxvals), np.nan, dtype=int);  
    fixed_knumbers = np.full((len(rhoJvals),), np.nan, dtype = float);  
    fixed_rhoJs = np.full((len(rhoJvals),), np.nan, dtype = float);   
    fixed_Energies = np.full((len(rhoJvals),), np.nan, dtype = complex);    
    
    # iter over rhoJavals
    for colori, target_rhoJ in enumerate(rhoJvals):

        # graphical dispersion for fixed energy
        fig, (dispax, dosax) = plt.subplots(ncols=2, sharey = True); 
        Ks_for_solution = np.logspace(-6,1,499); # creates a discrete set of energy points, 
                                             # logarithmic w/r/t desired band edge
        if(case in ["CB_rhos"]): 
            discrete_band = np.min(band_edges)+Ks_for_solution;
            discrete_band = discrete_band[discrete_band < np.max(band_edges)]; # stay w/in conduction band
        elif(case in ["VB_rhos","cartoon_rhos"]): 
            discrete_band = np.min(-band_edges)+Ks_for_solution;
            discrete_band = discrete_band[discrete_band < np.max(-band_edges)]; # stay w/in valence band
        else: raise NotImplementedError("case = "+case);
        dispks = np.linspace(-np.pi, np.pi,len(Ks_for_solution));
        disp = wfm.dispersion_RiceMele(diag_base_nospin, offdiag_base_nospin, dispks);
        # plot and format the dispersion
        for dispvals in disp: dispax.plot(dispks/np.pi, dispvals,color="cornflowerblue");
    
        # highlight the parts of the band we are considering
        discrete_ks = np.arccos(1/(2*vval*wval)*(discrete_band**2 - uval**2 - vval**2 - wval**2))
        dispax.scatter(discrete_ks/np.pi, discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]); 
    
        # graphical density of states for fixed energy
        discrete_dos = 2/np.pi*abs(1/np.gradient(discrete_band, discrete_ks));
    
        dosax.scatter(discrete_dos,discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]);
        dosline_from_rhoJ = target_rhoJ/abs(Jval);
       # solve graphically for fixed E *specifically in VB/CB* that gives desired rhoJa
        fixed_Energies[colori] = complex(discrete_band[np.argmin(abs(discrete_dos-dosline_from_rhoJ))],0);
        # ^ grabs one of the discrete energy points in this_band, based on having closest to desired rho(E)
        # v grabs corresponding k and rho(E) values
        fixed_knumbers[colori] = discrete_ks[np.argmin(abs(discrete_dos-dosline_from_rhoJ))];
        fixed_rhoJs[colori] = discrete_dos[np.argmin(abs(discrete_dos-dosline_from_rhoJ))]*abs(Jval);
        # NB we use this, the closest discrete rhoJ, rather than command-line rhoJ
        del target_rhoJ;
        print("\nfixed_rhoJ = {:.6f}".format(fixed_rhoJs[colori]));
        print("fixed_Energy = {:.6f}".format(np.real(fixed_Energies[colori])));
        print("fixed_knumber = {:.6f}".format(fixed_knumbers[colori]));
        dosax.axvline(dosline_from_rhoJ, color=UniversalAccents[1], linestyle = "dashed");
        dosax.axhline(np.real(fixed_Energies[colori]), color=UniversalAccents[1], linestyle="dashed");
        dispax.axvline(fixed_knumbers[colori]/np.pi, color=UniversalAccents[1], linestyle = "dashed");
        dispax.axhline(np.real(fixed_Energies[colori]), color=UniversalAccents[1], linestyle="dashed");
      
        # plotting
        if(case in ["VB_rhos","cartoon_rhos"]): 
            RiceMele_band = "-";
            RiceMele_shift_str = "$, E_{min}^{(VB)}="+"{:.2f}$".format(np.min(-band_edges))
        elif(case in ["CB_rhos"]): 
            RiceMele_band = "+";
            RiceMele_shift_str="$,  E_{min}^{(CB)}="+"{:.2f}$".format(np.min(band_edges))
        RiceMele_shift_str += ",  $k_i a/\pi \in [{:.2f},{:.2f}]$".format(discrete_ks[0]/np.pi, discrete_ks[-1]/np.pi);
        dispax.set_ylabel("$E_\pm( k_i)$"+RiceMele_shift_str, fontsize = myfontsize);
        dosax.set_xlabel("$\\rho, \\rho J_{sd} a ="+"{:.2f}".format(fixed_rhoJs[colori])+", J_{sd} ="+"{:.2f}$".format( Jval), fontsize = myfontsize);
        dosax.set_xlim(0,10);
        dispax.set_xlabel("$k_i a/\pi$", fontsize = myfontsize);
        dispax.set_title(title_RiceMele, fontsize = myfontsize);
    
        # show
        plt.tight_layout();
        stopflag = False;
        try: 
            if(sys.argv[5]=="stop"):
                # if code gets here, we show the E vs k, rho plot locating the fixed numbers and stop
                plt.show();
                stopflag = True;
        except:
            # usually we go here and don't plot
            plt.close();
            print(">>> Not flagged to stop");
        assert(not stopflag); 
        
        ####
        #### finally done determining energy, wavenumber for this color set (rhoJa fixed value)
    
        # determine the number of lattice constants across this range
        kdavals[colori,:] = np.linspace(*kdalims, myxvals);
        Distvals[colori,:] = np.rint(kdavals[colori]/fixed_knumbers[colori]).astype(int);
        
        # truncate to remove 0's, then extend back to length myxvals
        kdavals_trunc = kdavals[colori, Distvals[colori] > 0];
        kdavals[colori,:] = np.append(np.full((myxvals-len(kdavals_trunc),),kdavals_trunc[0]),kdavals_trunc); # extend
        Distvals_trunc = Distvals[colori, Distvals[colori] > 0];
        Distvals[colori,:] = np.append(np.full((myxvals-len(Distvals_trunc),),Distvals_trunc[0]),Distvals_trunc); # extend
        print("Nd values covered ({:.0f} total) =\n".format(len(Distvals[colori])),Distvals[colori]);
    
        # iter over Distvals to compute T
        for Distvali in range(len(Distvals[colori])):
        
            # construct hams
            i1, i2 = [1], [Distvals[colori, Distvali]+1];
            hblocks_noRM = h_cicc_reduced(Jval, i1, i2, i2[-1]+2, my_unit_cell, myspinS); 
            # ^ the +2 is for each lead site
            assert(np.shape(hblocks_noRM)[1] == my_unit_cell*n_spin_dof);
            hblocks = 1*hblocks_noRM;
            tnn = np.zeros_like(hblocks);
            # add Rice Mele terms
            for blocki in range(len(hblocks)):
                hblocks[blocki] += diag_base_RM_spin;
                tnn[blocki] += offdiag_base_RM_spin;
            tnn = tnn[:-1];
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            if(Distvali==0): 
                print("hblocks =\n");
                blockstoprint = 3;
                for blocki in range(blockstoprint):
                    print("\n\n");
                    for chunki in range(my_unit_cell):
                        print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,chunki, chunki))
                        print(np.real(hblocks[blocki][chunki*n_spin_dof:(chunki+1)*n_spin_dof,chunki*n_spin_dof:(chunki+1)*n_spin_dof]));
                    print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,0, 1))
                    print(np.real(hblocks[blocki][0*n_spin_dof:(0+1)*n_spin_dof,1*n_spin_dof:(1+1)*n_spin_dof]));
                    print("t(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,1, 0))
                    print(np.real(tnn[blocki][n_spin_dof:,:n_spin_dof]));
                print("J = {:.4f}".format(Jval));
                print("rhoJ = {:.4f}".format(fixed_rhoJs[colori]));
                print("max N = {:.0f}\n".format(np.max(Distvals[colori])+2));

            for sigmai in range(len(sigmas)):# sourcei is one of the entangled pairs always 
                source = np.zeros(my_unit_cell*n_spin_dof); # <- has site flavor dofs so the vector
                                      # outputs of wfm.kernel will as well.
                                      # You must remove the site flavor dofs manually!!

                source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet, impinging on A site

                # get  T coefs
                Rdum_mu, Tdum_mu = wfm.kernel(hblocks, tnn, tnnn, abs(vval), fixed_Energies[colori], "g_RiceMele", 
                          source, False, False, all_debug = True, verbose=0);
                Tdum = Tdum_mu[n_spin_dof:]; # extract only at boundary (B site for T)
                Rdum = Rdum_mu[:n_spin_dof]; # extract only at boundary (A site for R)
                Tvals[colori, Distvali,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
                Tsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum);
                TpRsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);
               
        ####
        #### end loop over MSQ-MSQ distance
    
    ####
    #### end loop over rhoJa values (colors)
    del rhoJvals;
    
    # figure setup
    width_ratios = [0.7,0.3];
    numrows = 2;
    colorfig, axes_arr = plt.subplots(numrows, len(width_ratios), sharex = "col",
                       gridspec_kw = dict(width_ratios=width_ratios));
    colorfig.set_size_inches(6*np.sum(width_ratios),2*numrows); #aspect ratio of run_wfm_single plots= 6*sum,4
    color_gridspec = axes_arr[0,0].get_gridspec();
    axes_arr[0,0].remove(); # remove both the 1st col axes where we will then create one combined
    axes_arr[1,0].remove();
    # combine into colorax
    colorax = colorfig.add_subplot(color_gridspec[:,0]) # gridspec grabs all 1st column rows

    # create and format axis for dispersion
    #axes_arr[1,1].remove();
    legend_ax = axes_arr[1,1];
    legend_ax.set_xticks(np.arange(2));
    legend_ax.set_yticks(np.arange(2));
    legend_ax.axis("off");
    dispax = axes_arr[0,1];
    dispax.axis("off")
    dispax.axvline(0,color="black");
    dispax.axhline(0,color="black");
    dispax.text(0.5,1.1,"$E_\pm(k_\sigma)$",fontsize=myfontsize,transform=dispax.transAxes);
    dispax.text(1.1,0.5,"$k_\sigma$",fontsize=myfontsize,transform=dispax.transAxes);
    dispax.plot(dispks, disp[0], color="red");
    dispax.plot(dispks, disp[1], color="red");
    
    # figure formatting;
    #colorax.set_ylabel("$T$", fontsize=myfontsize);
    # labels
    if(vsN):
        colorax.set_xlabel("$N$",fontsize=myfontsize);
        colorax.set_xlim(np.min(np.min(Distvals,axis=1))+1, np.min(np.max(Distvals, axis=1))+1);
    else:
        colorax.set_xlabel("$N_d k_i a / \pi$",fontsize=myfontsize);
        colorax.set_xlim(kdalims[0]/np.pi, kdalims[1]/np.pi);
        colorax.set_xticks( np.arange(int(np.rint(max(kdalims)/np.pi+1))));
        
    if(case in ["cartoon_rhos"]): # cartoon labels
        colorax.axis("off");
        colorax.set_ylim(0.0,1.2)
        if(vsN):
            middle_klim = np.median(Distvals[0]);
            colorax.axvline(middle_klim,color="black",linewidth=2);           
        else:
            middle_klim = 1.0;
            colorax.axvline(middle_klim,color="black",linewidth=2);
            colorax.set_xlim(middle_klim*0.5, middle_klim*1.5);
        colorax.axhline(0,color="black",linewidth=4);
        colorax.text(0.0,-0.1,"MSQ-MSQ separation",fontsize=myfontsize,transform=colorax.transAxes);
    else: # non-cartoon labels
        colorax.set_title(title_RiceMele+", $J_{sd} = "+"{:.2f}$".format(Jval), fontsize=myfontsize)
        colorax.set_ylim(0.0, 1.0);
        
    # efficiency
    efficiency_colors_N = (Tsummed[:,:,sigmas[1]]-Tsummed[:,:,sigmas[0]])/(Tsummed[:,:,sigmas[1]] + Tsummed[:,:,sigmas[0]]);
        
    # plot transmission coefficients vs N (1+MSQ-MSQ distance)
    # to_plot = |T0>,|S>, eta
    yvals_identifiers = ["$T(|S\\rangle )$", "$T(|T_0 \\rangle)$", "$\eta$"];
    yvals_to_plot = [Tsummed[:,:,sigmas[1]], Tsummed[:,:,sigmas[0]], efficiency_colors_N]; 
    yvals_styles = ["solid","dotted","dashdot"];
    if(vsN): yvals_identifiers = yvals_identifiers[:-1]; yvals_to_plot = yvals_to_plot[:-1]; yvals_styles = yvals_styles[:-1]; # dont plot eta in this case
    lines_to_legend_tuples = []; # append solid, dashed tuple for each color
    lines_to_legend_labels = [];
    for colori in range(len(fixed_rhoJs)):
    
        # x axis
        indep_vals = Distvals[colori]*fixed_knumbers[colori]/np.pi;
        if(vsN): indep_vals = Distvals[colori]+1;
        
        # plot
        this_line_tuple = [];
        for stylei, yvals in enumerate(yvals_to_plot):
            # handle label--only label once per colori  
            if(stylei==0):
                lines_to_legend_labels.append("$\\rho (k_i) J_{sd} a = "+"{:.1f}, k_i a/\pi = {:.2f}$".format(fixed_rhoJs[colori], fixed_knumbers[colori]/np.pi));
                
            # handle identifiers: one for each style
            if(colori==0):
                legend_ax.plot([np.nan], [np.nan], color="black",linestyle=yvals_styles[stylei], label = yvals_identifiers[stylei]);
                
            # plot line
            line_fromstyle, = colorax.plot(indep_vals, yvals[colori], color=UniversalColors[colori], linestyle=yvals_styles[stylei]);
            
            # handle line object for passing to legend
            this_line_tuple.append(line_fromstyle);
        lines_to_legend_tuples.append(tuple(this_line_tuple)); 

    # truncate tuples if not desiring to combine solid-dotted-dash lines in main legend
    if(True):
        for tupi in range(len(lines_to_legend_tuples)): 
            lines_to_legend_tuples[tupi] = (lines_to_legend_tuples[tupi][0],); 

    # show
    if(True and (case not in ["cartoon_rhos"])):
        color_legend = colorax.legend(lines_to_legend_tuples, lines_to_legend_labels,
           #bbox_to_anchor =(0.00,1.02,1.00,0.102),loc="lower left",mode="expand",borderaxespad=0.0, 
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple( ndivide=None)},fontsize=myfontsize);
    legend_ax.legend(fontsize = myfontsize);
    plt.tight_layout();
    folder = "/home/cpbunker/Desktop/FIGS_Cicc_WFM/"
    if(case in ["cartoon_rhos"]):
        dispax.remove();
        fname = folder+"cartoon_rhos.pdf";
    else:
        if(vsN): fname = folder+'vsN.pdf';
        else: 
            if(Jval == -0.005): fname = folder+'disparity_periodic_Jsmall.pdf';
            else: fname = folder+'disparity_periodic.pdf';
    print("Saving plot to "+fname);
    plt.savefig(fname);

if(case in ["CB_ks", "cartoon_ks", "VB_ks"]): # entanglement *preservation* vs k
    my_unit_cell = 2; # since diatomic

    # Rice-Mele tight binding
    vval = -1.0; # sets energy scale
    wval = float(sys.argv[3]);
    uval = float(sys.argv[4]); 
    Jval = -0.05;
    myspinS = 0.5;
    if(case in ["cartoon_ks"]): 
        Distvals = np.array([20]);
        myfontsize = 1.5*myfontsize;
    else: 
        Distvals = np.array([20,40,80])  # diff colors for MSQ-MSQ separation

    # Rice-Mele matrices
    n_spin_dof = 3; # spin dofs
    diag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
    diag_base_RM_spin[:n_spin_dof,:n_spin_dof] = uval*np.eye(n_spin_dof);
    diag_base_RM_spin[n_spin_dof:,n_spin_dof:] = -uval*np.eye(n_spin_dof);
    diag_base_RM_spin[:n_spin_dof,n_spin_dof:] = vval*np.eye(n_spin_dof);
    diag_base_RM_spin[n_spin_dof:,:n_spin_dof] = vval*np.eye(n_spin_dof);
    offdiag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
    offdiag_base_RM_spin[n_spin_dof:,:n_spin_dof] = wval*np.eye(n_spin_dof);
    diag_base_nospin = diag_base_RM_spin[::n_spin_dof,::n_spin_dof];
    offdiag_base_nospin = offdiag_base_RM_spin[::n_spin_dof,::n_spin_dof];
    assert(abs(np.sum(np.diagonal(diag_base_nospin))/len(diag_base_nospin)) < 1e-10); # u0 = 0
    band_edges = wfm.bandedges_RiceMele(diag_base_nospin, offdiag_base_nospin)[-2:];
    
    # output Rice-Mele
    title_RiceMele = wfm.string_RiceMele(diag_base_nospin, offdiag_base_nospin, energies=False, tex=True)
    print("\n\nRice-Mele "+title_RiceMele);
    print("h00 =\n",diag_base_nospin);
    print("h01 =\n",offdiag_base_nospin);
                                   
    # channels
    pair = (0,1); # pair[0] = |+> channel, pair[1] = |-> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement
                                          # in this case, elec up, MSQs in triplet, singlet
                                          # source must impinge on A orbital 

                                    
    # return values
    # shaped by fixed NMSQ-MSQ distance (color), wavenumber (k axis), spin dofs 
    # Transmission coefficients. Note:
        # we compute only source channel -> source channel scattering, leave the rest as NaNs
        # we evaluate T at SR boundary, namely the B site
    Tvals = np.full((len(Distvals),myxvals,n_spin_dof), np.nan, dtype=float);
   
    # Tsummed measures source channel -> all channels transmission
    Tsummed = np.full((len(Distvals),myxvals,n_spin_dof), np.nan, dtype=float);
    TpRsummed = np.full((len(Distvals),myxvals,n_spin_dof), np.nan, dtype=float); 

    # k = wavenumber
    # d = number of lattice constants between MSQ 1 and MSQ 2
    kdalims = 0.1*np.pi, 1.01*np.pi; 
    if(sys.argv[2] == "vskda"): vskda=True;
    else: vskda = False;
    knumbers = np.full((len(Distvals),myxvals), np.nan, dtype=float);
    kdavals = np.full((len(Distvals),myxvals), np.nan, dtype=float);
    rhoJvals = np.full((len(Distvals),myxvals), np.nan, dtype=float);

    # iter over MSQ-MSQ distance
    for colori in range(len(Distvals)):

        # determine range of wavenumbers from Dist value
        if(vskda):
            discrete_ks = np.linspace(kdalims[0]/Distvals[colori],kdalims[1]/Distvals[colori], myxvals);
        else:
            #klims = (1e-3*np.pi, 1e-1*np.pi); # logarithmic!
            #discrete_ks = np.logspace(np.log10(klims[0]), np.log10(klims[1]), myxvals);
            klims = (1e-4*np.pi, 0.06*np.pi); # linear
            discrete_ks = np.linspace(klims[0], klims[1], myxvals);


        # graphical dispersion for this run
        fig, (dispax, dosax) = plt.subplots(ncols=2, sharey = True);
        if(case in ["CB_ks"]): 
            discrete_band = wfm.dispersion_RiceMele(diag_base_nospin, offdiag_base_nospin, discrete_ks)[1];
        elif(case in ["VB_ks","cartoon_ks"]): 
            discrete_band = wfm.dispersion_RiceMele(diag_base_nospin, offdiag_base_nospin, discrete_ks)[0];
        else: raise NotImplementedError("case = "+case);
   
        # highlight the parts of the band we are considering
        print(np.shape(discrete_ks))
        print(np.shape(discrete_band))
        dispax.scatter(discrete_ks, discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]); 
    
        # graphical density of states for fixed energy
        discrete_dos = 2/np.pi*abs(1/np.gradient(discrete_band, discrete_ks));
        dosax.scatter(discrete_dos,discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]);

        # plotting
        if(case in ["VB_ks","cartoon_ks"]): 
            RiceMele_band = "-";
            RiceMele_shift_str = "$, E_{min}^{(VB)}="+"{:.2f}$".format(np.min(-band_edges))
        elif(case in ["CB_ks"]): 
            RiceMele_band = "+";
            RiceMele_shift_str="$,  E_{min}^{(CB)}="+"{:.2f}$".format(np.min(band_edges))
        dispax.set_ylabel("$E_\pm( k_i)$"+RiceMele_shift_str, fontsize = myfontsize);
        dosax.set_xlabel("$\\rho, J_{sd} ="+"{:.2f}$".format( Jval), fontsize = myfontsize);
        dispax.set_xlabel("$k_i$", fontsize = myfontsize);
        dispax.set_title(title_RiceMele, fontsize = myfontsize);
    
        # show
        plt.tight_layout();
        stopflag = False;
        try: 
            if(sys.argv[5]=="stop"): 
                plt.show();
                stopflag = True;
        except: print(">>> Not flagged to stop");
        assert(not stopflag); 
        plt.close();
        
        # determine kda
        knumbers[colori,:] = discrete_ks[:];
        kdavals[colori,:] = discrete_ks*Distvals[colori];
        rhoJvals[colori, :] = discrete_dos*abs(Jval);
        print("Nd = {:.0f}".format(Distvals[colori]));
    
        # iter over Distvals to compute T
        for ki in range(len(discrete_ks)):
        
            # construct hams
            i1, i2 = [1], [Distvals[colori]+1];
            hblocks_noRM = h_cicc_reduced(Jval, i1, i2, i2[-1]+2, my_unit_cell, myspinS); 
            # ^ the +2 is for each lead site
            assert(np.shape(hblocks_noRM)[1] == my_unit_cell*n_spin_dof);
            hblocks = 1*hblocks_noRM;
            tnn = np.zeros_like(hblocks);
            # add Rice Mele terms
            for blocki in range(len(hblocks)):
                hblocks[blocki] += diag_base_RM_spin;
                tnn[blocki] += offdiag_base_RM_spin;
            tnn = tnn[:-1];
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            if(ki==0): 
                print("hblocks =\n");
                blockstoprint = 3;
                for blocki in range(blockstoprint):
                    print("\n\n");
                    for chunki in range(my_unit_cell):
                        print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,chunki, chunki))
                        print(np.real(hblocks[blocki][chunki*n_spin_dof:(chunki+1)*n_spin_dof,chunki*n_spin_dof:(chunki+1)*n_spin_dof]));
                    print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,0, 1))
                    print(np.real(hblocks[blocki][0*n_spin_dof:(0+1)*n_spin_dof,1*n_spin_dof:(1+1)*n_spin_dof]));
                    print("t(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,1, 0))
                    print(np.real(tnn[blocki][n_spin_dof:,:n_spin_dof]));
                print("J = {:.4f}".format(Jval));
                print("rhoJ min, median, max = {:.2f}, {:.2f}, {:.2f}".format(np.min(rhoJvals[colori]), np.median(rhoJvals[colori]), np.max(rhoJvals[colori])));
                print("Nd = {:.0f}\n".format(Distvals[colori]));

            for sigmai in range(len(sigmas)):# sourcei is one of the entangled pairs always 
                source = np.zeros(my_unit_cell*n_spin_dof); # <- has site flavor dofs so the vector
                                      # outputs of wfm.kernel will as well.
                                      # You must remove the site flavor dofs manually!!

                source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet, impinging on A site

                # get  T coefs
                Rdum_mu, Tdum_mu = wfm.kernel(hblocks, tnn, tnnn, abs(vval), discrete_band[ki], "g_RiceMele", 
                          source, False, False, all_debug = True, verbose=0);
                Tdum = Tdum_mu[n_spin_dof:]; # extract only at boundary (B site for T)
                Rdum = Rdum_mu[:n_spin_dof]; # extract only at boundary (A site for R)
                Tvals[colori, ki,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
                Tsummed[colori, ki,sigmas[sigmai]] = np.sum(Tdum);
                TpRsummed[colori, ki,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);
               
        ####
        #### end loop over electron wavenumber
    
    ####
    #### end loop over colors
    
    # figure setup
    width_ratios = [0.7,0.3];
    numrows = 2;
    colorfig, axes_arr = plt.subplots(numrows, len(width_ratios), sharex = "col",
                       gridspec_kw = dict(width_ratios=width_ratios));
    colorfig.set_size_inches(6*np.sum(width_ratios),2*numrows); #aspect ratio of run_wfm_single plots= 6*sum,4
    color_gridspec = axes_arr[0,0].get_gridspec();
    axes_arr[0,0].remove(); # remove both the 1st col axes where we will then create one combined
    axes_arr[1,0].remove();
    # combine into colorax
    colorax = colorfig.add_subplot(color_gridspec[:,0]) # gridspec grabs all 1st column rows

    # create and format axis for dispersion
    #axes_arr[1,1].remove();
    legend_ax = axes_arr[1,1];
    legend_ax.set_xticks(np.arange(2));
    legend_ax.set_yticks(np.arange(2));
    legend_ax.axis("off");
    dispax = axes_arr[0,1];
    dispax.axis("off")
    dispax.axvline(0,color="black");
    dispax.axhline(0,color="black");
    dispax.text(0.5,1.1,"$E_\pm(k_\sigma)$",fontsize=myfontsize,transform=dispax.transAxes);
    dispax.text(1.1,0.5,"$k_\sigma$",fontsize=myfontsize,transform=dispax.transAxes);
    ks_for_disp = np.linspace(-np.pi, np.pi, 99);
    bands_for_disp = wfm.dispersion_RiceMele(diag_base_nospin, offdiag_base_nospin, ks_for_disp);
    dispax.plot(ks_for_disp, bands_for_disp[0], color="red");
    dispax.plot(ks_for_disp, bands_for_disp[1], color="red");
    if(not vskda): dispax.scatter(discrete_ks, discrete_band);
    
    # figure formatting
    if(case not in ["cartoon_ks"]): colorax.set_title(title_RiceMele+", $J_{sd} = "+"{:.2f}$".format(Jval), fontsize=myfontsize);
    #colorax.set_ylabel("$T$", fontsize=myfontsize);
    colorax.set_ylim(0.0, 1.0);
    # labels
    if(vskda):
        colorax.set_xlabel("$N_d k_i a / \pi$",fontsize=myfontsize);
        colorax.set_xticks( np.arange(int(np.rint(max(kdalims)+1))));
    else:
        colorax.set_xlabel("$k_i a / \pi$",fontsize=myfontsize);
        #colorax.set_xscale("log");
    if(case in ["cartoon_ks"]): # cartoon labels
        colorax.axis("off");
        colorax.axvline(0,color="black",linewidth=4);
        colorax.axhline(0,color="black",linewidth=4);
        colorax.text(0,-0.1,"Electron wavenumber",fontsize=myfontsize,transform=colorax.transAxes);

    # efficiency
    efficiency_colors_k = (Tsummed[:,:,sigmas[1]]-Tsummed[:,:,sigmas[0]])/(Tsummed[:,:,sigmas[1]] + Tsummed[:,:,sigmas[0]]);
        
    # plot transmission coefficients vs N (1+MSQ-MSQ distance)
    # to_plot = |T0>,|S>, eta
    yvals_identifiers = ["$T(|S\\rangle )$", "$T(|T_0 \\rangle)$", "$\eta$"];
    yvals_to_plot = [Tsummed[:,:,sigmas[1]], Tsummed[:,:,sigmas[0]], efficiency_colors_k]; # |T0> then |S>
    yvals_styles = ["solid","dotted","dashdot"];
    lines_to_legend_tuples = []; # append solid, dashed tuple for each color
    lines_to_legend_labels = [];
    for colori in range(len(Distvals)):
    
        # x axis
        # NB we will be monotonically changing rho Jsd a as we sweep k
        # but large rho Jsd a only helps **at resonance**
        if(vskda):
            indep_vals = kdavals[colori]/np.pi;
        else:
            indep_vals = knumbers[colori]/np.pi
            #indep_vals = rhoJvals[colori]
            #colorax.set_xlabel("$\\rho(k_i) J_{sd} a$", fontsize=myfontsize);
        colorax.set_xlim(np.min(indep_vals), np.max(indep_vals))
        
        # plot
        this_line_tuple = [];
        for stylei, yvals in enumerate(yvals_to_plot):
            # handle label--only label once per colori  
            if(stylei==0):
                lines_to_legend_labels.append("$ N_d = {:.0f}$".format(Distvals[colori]));

             # handle identifiers: one for each style
            if(colori==0):
                legend_ax.plot([np.nan], [np.nan], color="black",linestyle=yvals_styles[stylei], label = yvals_identifiers[stylei]);
                
            # plot line
            line_fromstyle, = colorax.plot(indep_vals, yvals[colori], color=UniversalColors[colori], linestyle=yvals_styles[stylei]);
            #colorax.axvline(1/Distvals[colori], color=UniversalColors[colori],alpha=0.4)
            
            # handle line object for passing to legend
            this_line_tuple.append(line_fromstyle);
        lines_to_legend_tuples.append(tuple(this_line_tuple)); 

    # truncate tuples if not desiring to combine solid-dotted-dash lines in main legend
    if(True):
        for tupi in range(len(lines_to_legend_tuples)): 
            lines_to_legend_tuples[tupi] = (lines_to_legend_tuples[tupi][0],); 

    # show
    if(True and (case not in ["cartoon_ks"])):
        color_legend = colorax.legend(lines_to_legend_tuples, lines_to_legend_labels,
           #bbox_to_anchor =(0.00,1.02,1.00,0.102),loc="lower left",mode="expand",borderaxespad=0.0, 
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple( ndivide=None)},fontsize=myfontsize);
    legend_ax.legend(fontsize = myfontsize);
    plt.tight_layout();
    folder = "/home/cpbunker/Desktop/FIGS_Cicc_WFM/";
    if(case in ["cartoon_ks"]):
        fname = folder+"cartoon_ks.pdf";
        dispax.remove();
    else:
        if(vskda): fname = folder+"vskda.pdf";
        else: fname = folder+"vsk.pdf";
    print("Saving plot to "+fname);
    plt.savefig(fname);
    
elif(case in ["CB_ws", "VB_ws"]): # entanglement *preservation* vs N, different colors for rho value
    my_unit_cell = 2; # since diatomic

    # Rice-Mele tight binding
    vval = -1.0; # sets energy scale
    wvals = np.array(sys.argv[3:]).astype(float);
    uval = 0.0; # always 
    Jval = -0.05;
    myspinS = 0.5;
    if(Jval == -0.005): myxvals = 30;
                                   
    # channels
    n_spin_dof = 3; # spin channels
    pair = (0,1); # pair[0] = |+> channel, pair[1] = |-> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement
                                          # in this case, elec up, MSQs in singlet or triplet
                                          # source must impinge on A orbital 
    # rhoJa = fixed throughout, fixed by specifying energy or wavenumber
    target_knumber = float(sys.argv[2])*np.pi;
    assert(target_knumber > 0 and target_knumber < 1.0);
                                    
    # return values
    # shaped by fixed rhoJval (color), MSQ-MSQ distance (x axis), spin dofs 
    # Transmission coefficients. Note:
        # we compute only source channel -> source channel scattering, leave the rest as NaNs
        # we evaluate T at SR boundary, namely the B site
    Tvals = np.full((len(wvals),myxvals,n_spin_dof), np.nan, dtype=float); 
    
    # Tsummed measures source channel -> all channels transmission
    Tsummed = np.full((len(wvals),myxvals,n_spin_dof), np.nan, dtype=float);
    TpRsummed = np.full((len(wvals),myxvals,n_spin_dof), np.nan, dtype=float); 
    
    # d = number of lattice constants between MSQ 1 and MSQ 2
    kdalims = 0.01*np.pi, 2.1*np.pi; 
    widelimsflag = False;
    try: 
        if(sys.argv[-1]=="widelims"): widelimsflag = True;
    except: print(">>> Not flagged to widen kda limits");
    if(widelimsflag): kdalims = 0.01*np.pi, 100*np.pi; kdaticks = [0.0, 40.0, 80.0];
    kdavals = np.full((len(wvals),myxvals), np.nan, dtype=float);  
    Distvals = np.full((len(wvals),myxvals), np.nan, dtype=int);  
    fixed_knumbers = np.full((len(wvals),), np.nan, dtype = float);  
    fixed_rhoJs = np.full((len(wvals),), np.nan, dtype = float);   
    fixed_Energies = np.full((len(wvals),), np.nan, dtype = complex);
    disp_ofw = []; # for plotting band structure inset on final plot   
    
    # iter over  w to manipulate rhoJavals
    for colori, wval in enumerate(wvals):
    
        # Rice-Mele matrices
        diag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
        diag_base_RM_spin[:n_spin_dof,:n_spin_dof] = uval*np.eye(n_spin_dof);
        diag_base_RM_spin[n_spin_dof:,n_spin_dof:] = -uval*np.eye(n_spin_dof);
        diag_base_RM_spin[:n_spin_dof,n_spin_dof:] = vval*np.eye(n_spin_dof);
        diag_base_RM_spin[n_spin_dof:,:n_spin_dof] = vval*np.eye(n_spin_dof);
        offdiag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
        offdiag_base_RM_spin[n_spin_dof:,:n_spin_dof] = wval*np.eye(n_spin_dof);
        diag_base_nospin = diag_base_RM_spin[::n_spin_dof,::n_spin_dof];
        offdiag_base_nospin = offdiag_base_RM_spin[::n_spin_dof,::n_spin_dof];
        assert(abs(np.sum(np.diagonal(diag_base_nospin))/len(diag_base_nospin)) < 1e-10); # u0 = 0
        band_edges = wfm.bandedges_RiceMele(diag_base_nospin, offdiag_base_nospin)[-2:];
    
        # output Rice-Mele
        title_RiceMele = wfm.string_RiceMele(diag_base_nospin, offdiag_base_nospin, energies=False, tex=True)
        print("\n\nRice-Mele "+title_RiceMele);
        print("h00 =\n",diag_base_nospin);
        print("h01 =\n",offdiag_base_nospin);

        # graphical dispersion for fixed energy
        fig, (dispax, dosax) = plt.subplots(ncols=2, sharey = True); myfontsize = 14;
        Ks_for_solution = np.logspace(-6,1,499); # creates a discrete set of energy points, 
                                             # logarithmic w/r/t desired band edge
        if(case in ["CB_ws"]): 
            discrete_band = np.min(band_edges)+Ks_for_solution;
            discrete_band = discrete_band[discrete_band < np.max(band_edges)]; # stay w/in conduction band
        elif(case in ["VB_ws"]): 
            discrete_band = np.min(-band_edges)+Ks_for_solution;
            discrete_band = discrete_band[discrete_band < np.max(-band_edges)]; # stay w/in valence band
        else: raise NotImplementedError("case = "+case);
        dispks = np.linspace(-np.pi, np.pi, len(Ks_for_solution));
        disp = wfm.dispersion_RiceMele(diag_base_nospin, offdiag_base_nospin, dispks);
        disp_ofw.append(1*disp);
        # plot and format the dispersion
        for dispvals in disp: dispax.plot(dispks/np.pi, dispvals,color="cornflowerblue");
    
        # highlight the parts of the band we are considering
        discrete_ks = np.arccos(1/(2*vval*wval)*(discrete_band**2 - uval**2 - vval**2 - wval**2))
        dispax.scatter(discrete_ks/np.pi, discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]); 
    
        # target wavenumber
        fixed_knumbers[colori] = discrete_ks[np.argmin(abs(discrete_ks-target_knumber))];
        fixed_Energies[colori] = complex(discrete_band[np.argmin(abs(discrete_ks-target_knumber))],0);

        # graphical density of states for fixed energy
        discrete_dos = 2/np.pi*abs(1/np.gradient(discrete_band, discrete_ks));
        fixed_rhoJs[colori] = discrete_dos[np.argmin(abs(discrete_ks-target_knumber))]*abs(Jval);
        dosax.scatter(discrete_dos,discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]);
        dosline_from_rhoJ = fixed_rhoJs[colori]/abs(Jval);
        print("\nfixed_rhoJ = {:.6f}".format(fixed_rhoJs[colori]));
        print("fixed_Energy = {:.6f}".format(np.real(fixed_Energies[colori])));
        print("fixed_knumber = {:.6f}".format(fixed_knumbers[colori]));
        dosax.axvline(dosline_from_rhoJ, color=UniversalAccents[1], linestyle = "dashed");
        dosax.axhline(np.real(fixed_Energies[colori]), color=UniversalAccents[1], linestyle="dashed");
        dispax.axvline(fixed_knumbers[colori]/np.pi, color=UniversalAccents[1], linestyle = "dashed");
        dispax.axhline(np.real(fixed_Energies[colori]), color=UniversalAccents[1], linestyle="dashed");
     
        # plotting
        if(case in ["VB_ws"]): 
            RiceMele_band = "-";
            RiceMele_shift_str = "$, E_{min}^{(VB)}="+"{:.2f}$".format(np.min(-band_edges))
        elif(case in ["CB_ws"]): 
            RiceMele_band = "+";
            RiceMele_shift_str="$,  E_{min}^{(CB)}="+"{:.2f}$".format(np.min(band_edges))
        RiceMele_shift_str += ",  $k_i a/\pi \in [{:.2f},{:.2f}]$".format(discrete_ks[0]/np.pi, discrete_ks[-1]/np.pi);
        dispax.set_ylabel("$E_\pm( k_i)$"+RiceMele_shift_str, fontsize = myfontsize);
        dosax.set_xlabel("$\\rho, \\rho J_{sd} a ="+"{:.2f}".format(fixed_rhoJs[colori])+", J_{sd} ="+"{:.2f}$".format( Jval), fontsize = myfontsize);
        dosax.set_xlim(0,10);
        dispax.set_xlabel("$k_i a/\pi$", fontsize = myfontsize);
        dispax.set_title(title_RiceMele, fontsize = myfontsize);
    
        # show
        plt.tight_layout();
        stopflag = False;
        try: 
            if(sys.argv[5]=="stop"):
                # if code gets here, we show the E vs k, rho plot locating the fixed numbers and stop
                plt.show();
                stopflag = True;
        except:
            # usually we go here and don't plot
            plt.close();
            print(">>> Not flagged to stop");
        assert(not stopflag); 
        
        ####
        #### finally done determining energy, wavenumber for this color set (rhoJa fixed value)
    
        # determine the number of lattice constants across this range
        kdavals[colori,:] = np.linspace(*kdalims, myxvals);
        Distvals[colori,:] = np.rint(kdavals[colori]/fixed_knumbers[colori]).astype(int);
        
        # truncate to remove 0's, then extend back to length myxvals
        kdavals_trunc = kdavals[colori, Distvals[colori] > 0];
        kdavals[colori,:] = np.append(np.full((myxvals-len(kdavals_trunc),),kdavals_trunc[0]),kdavals_trunc); # extend
        Distvals_trunc = Distvals[colori, Distvals[colori] > 0];
        Distvals[colori,:] = np.append(np.full((myxvals-len(Distvals_trunc),),Distvals_trunc[0]),Distvals_trunc); # extend
        print("Nd values covered ({:.0f} total) =\n".format(len(Distvals[colori])),Distvals[colori]);
    
        # iter over Distvals to compute T
        for Distvali in range(len(Distvals[colori])):
        
            # construct hams
            i1, i2 = [1], [Distvals[colori, Distvali]+1];
            hblocks_noRM = h_cicc_reduced(Jval, i1, i2, i2[-1]+2, my_unit_cell, myspinS); 
            # ^ the +2 is for each lead site
            hblocks = 1*hblocks_noRM;
            tnn = np.zeros_like(hblocks);
            # add Rice Mele terms
            for blocki in range(len(hblocks)):
                hblocks[blocki] += diag_base_RM_spin;
                tnn[blocki] += offdiag_base_RM_spin;
            tnn = tnn[:-1];
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
            if(Distvali==0): 
                print("hblocks =\n");
                blockstoprint = 3;
                for blocki in range(blockstoprint):
                    print("\n\n");
                    for chunki in range(my_unit_cell):
                        print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,chunki, chunki))
                        print(np.real(hblocks[blocki][chunki*n_spin_dof:(chunki+1)*n_spin_dof,chunki*n_spin_dof:(chunki+1)*n_spin_dof]));
                    print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,0, 1))
                    print(np.real(hblocks[blocki][0*n_spin_dof:(0+1)*n_spin_dof,1*n_spin_dof:(1+1)*n_spin_dof]));
                    print("t(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,1, 0))
                    print(np.real(tnn[blocki][n_spin_dof:,:n_spin_dof]));
                print("J = {:.4f}".format(Jval));
                print("rhoJ = {:.4f}".format(fixed_rhoJs[colori]));
                print("max N = {:.0f}\n".format(np.max(Distvals[colori])+2));

            for sigmai in range(len(sigmas)):   
                # sourcei is one of the pairs always 
                source = np.zeros(my_unit_cell*n_spin_dof);
                source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet, impinging on A site

                # get  T coefs
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, abs(vval), fixed_Energies[colori], "g_RiceMele", 
                          source, False, False, all_debug = True);
                Tdum = Tdum[n_spin_dof:]; # extract only at boundary (B site for T)
                Rdum = Rdum[:n_spin_dof]; # extract only at boundary (A site for R)
                Tvals[colori, Distvali,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
                Tsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum);
                TpRsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);
                if(not(abs(1- TpRsummed[colori, Distvali,sigmas[sigmai]] )<1e-10)):
                    print( abs(1- TpRsummed[colori, Distvali,sigmas[sigmai]] )); assert False
                 
        ####
        #### end loop over MSQ-MSQ distance
    
    ####
    #### end loop over wvals (colors)
    
    # figure setup
    width_ratios = [0.7,0.3];
    numrows = 2;
    colorfig, axes_arr = plt.subplots(numrows, len(width_ratios), sharex = "col",
                       gridspec_kw = dict(width_ratios=width_ratios));
    colorfig.set_size_inches(6*np.sum(width_ratios),2*numrows); #aspect ratio of run_wfm_single plots= 6*sum,4
    color_gridspec = axes_arr[0,0].get_gridspec();
    axes_arr[0,0].remove(); # remove both the 1st col axes where we will then create one combined
    axes_arr[1,0].remove();
    # combine into colorax
    colorax = colorfig.add_subplot(color_gridspec[:,0]) # gridspec grabs all 1st column rows

    # create and format axis for dispersion
    #axes_arr[1,1].remove();
    legend_ax = axes_arr[1,1];
    legend_ax.set_xticks(np.arange(2));
    legend_ax.set_yticks(np.arange(2));
    legend_ax.axis("off");
    dispax = axes_arr[0,1];
    dispax.axis("off")
    dispax.axvline(0,color="black");
    dispax.axhline(0,color="black");
    dispax.text(0.5,1.1,"$E_\pm(k_\sigma)$",fontsize=myfontsize,transform=dispax.transAxes);
    dispax.text(1.1,0.5,"$k_\sigma$",fontsize=myfontsize,transform=dispax.transAxes);
    for colori in range(len(wvals)):
        dispax.plot(dispks, disp_ofw[colori][0], color=UniversalColors[colori]);
        dispax.plot(dispks, disp_ofw[colori][1], color=UniversalColors[colori]);
    
    # figure formatting
    title_with_Jsd = "$J_{sd} = "+"{:.2f}".format(Jval)+", k_i a/\pi = "+"{:.2f}$".format(fixed_knumbers[0]/np.pi)+"$, E_{"+RiceMele_band+"}$ band";
    if(Jval == -0.005): 
        title_with_Jsd = "$J_{sd} = "+"{:.3f}".format(Jval)+", k_i a/\pi = "+"{:.3f}$".format(fixed_knumbers[0]/np.pi)+"$, E_{"+RiceMele_band+"}$ band"
    colorax.set_title(title_with_Jsd, fontsize=myfontsize);
    print(fixed_knumbers/np.pi);
    #colorax.set_ylabel("$T$", fontsize=myfontsize);
    colorax.set_ylim(0.0, 1.02);
    colorax.set_xlabel("$N_d k_i a / \pi$",fontsize=myfontsize);
    colorax.set_xticks( np.arange(int(np.rint(max(kdalims)+1))));
    colorax.set_xlim(0.0, max(kdalims)/np.pi);
    
    # efficiency
    efficiency_colors_N = (Tsummed[:,:,sigmas[1]]-Tsummed[:,:,sigmas[0]])/(Tsummed[:,:,sigmas[1]] + Tsummed[:,:,sigmas[0]]);
    
    # plot transmission coefficients vs N (1+MSQ-MSQ distance)
    # # |S>, |T0>, eta
    yvals_identifiers = ["$T(|S\\rangle )$", "$T(|T_0 \\rangle)$", "$\eta$"];
    yvals_to_plot = [Tsummed[:,:,sigmas[1]], Tsummed[:,:,sigmas[0]], efficiency_colors_N]; 
    yvals_styles = ["solid","dotted", "dashdot"];
    lines_to_legend_tuples = []; # append solid, dashed tuple for each color
    lines_to_legend_labels = [];
    for colori in range(len(wvals)):
    
        # x axis
        indep_vals = Distvals[colori]*fixed_knumbers[colori]/np.pi;
        
        # plot
        this_line_tuple = [];
        for stylei, yvals in enumerate(yvals_to_plot):
            # only label once per colori  
            if(stylei==0):
                style_label = "$w/|v| = {:.2f}$".format(wvals[colori]); 
                style_label += "$, \\rho(k_i) J_{sd} a ="+"{:.2f}$".format(fixed_rhoJs[colori]);
                lines_to_legend_labels.append(style_label);
                
            # handle identifiers: one for each style
            if(colori==0):
                legend_ax.plot([np.nan], [np.nan], color="black",linestyle=yvals_styles[stylei], label = yvals_identifiers[stylei]);

            # plot line
            line_fromstyle, = colorax.plot(indep_vals, yvals[colori], label=style_label, color=UniversalColors[colori], linestyle=yvals_styles[stylei]);
            
            # handle line object for passing to legend
            this_line_tuple.append(line_fromstyle);
        lines_to_legend_tuples.append(tuple(this_line_tuple)); 

    # truncate tuples if not desiring to combine solid-dotted-dash lines in main legend
    if(True):
        for tupi in range(len(lines_to_legend_tuples)): 
            lines_to_legend_tuples[tupi] = (lines_to_legend_tuples[tupi][0],); 

    # show
    if(False):
        color_legend = colorax.legend(lines_to_legend_tuples, lines_to_legend_labels,
           #bbox_to_anchor =(0.00,1.02,1.00,0.102),loc="lower left",mode="expand",borderaxespad=0.0, 
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple( ndivide=None)},fontsize=myfontsize);
    legend_ax.legend(fontsize = myfontsize);
    plt.tight_layout();
    folder = "/home/cpbunker/Desktop/FIGS_Cicc_WFM/"
    fname = folder+'disparity_'+case[:2]+'_nearGamma.pdf'
    if(Jval == -0.005): fname = folder+'disparity_'+case[:2]+'_Jsmall.pdf'
    print("Saving plot to "+fname);
    plt.show();
    plt.savefig(fname);

elif(case in ["VB_rho_fixed", "CB_rho_fixed"]):
    # x axis sweeps over k, but we also change w to fix rho
    # different colors for rho value
    my_unit_cell = 2; # since diatomic

    # Rice-Mele tight binding
    vval = -1.0; # sets energy scale
    uval = 0.0; # always 
    Jval = -0.05;
    myspinS = 0.5;
                                   
    # channels
    n_spin_dof = 3; # spin channels
    pair = (0,1); # pair[0] = |+> channel, pair[1] = |-> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement
                                          # in this case, elec up, MSQs in singlet or triplet
                                          # source must impinge on A orbital
                                          
    # rhoJa = fixed for all color curves
    target_rhoJ = float(sys.argv[2])
    target_knumbers = np.array(sys.argv[3:]).astype(float)*np.pi;
    # other color curve specific values
    ws_colors = np.zeros_like(target_knumbers);
    fixed_knumbers = np.full((len(ws_colors),), np.nan, dtype = float);  
    fixed_rhoJs = np.full((len(ws_colors),), np.nan, dtype = float);   
    fixed_Energies = np.full((len(ws_colors),), np.nan, dtype = complex);
    if(case in ["VB_rho_fixed"]):
        band_to_choose = 0;
    elif(case in ["CB_rho_fixed"]):
        band_to_choose = 1;
    else:
        raise NotImplementedError;
    disp_ofw = []; # for plotting band structure inset on final plot   
                                        
    # return values
    # shaped by fixed rhoJval (color), MSQ-MSQ distance (x axis), spin dofs 
    # Transmission coefficients. Note:
        # we compute only source channel -> source channel scattering, leave the rest as NaNs
        # we evaluate T at SR boundary, namely the B site
    Tvals = np.full((len(target_knumbers),myxvals,n_spin_dof), np.nan, dtype=float); 
    
    # Tsummed measures source channel -> all channels transmission
    Tsummed = np.full((len(target_knumbers),myxvals,n_spin_dof), np.nan, dtype=float);
    TpRsummed = np.full((len(target_knumbers),myxvals,n_spin_dof), np.nan, dtype=float);   

    # determine w value appropriate for each target wavenumber
    for ki in range(len(target_knumbers)):

        # graphical dispersion for near target wavenumber
        fig, (dosax, dispax) = plt.subplots(ncols=2); myfontsize = 14;
        ws_for_solution = np.linspace(-1,-0.01,499); # creates a discrete set of bands
           
        # find the local dos around the target wavenumber
        discrete_ks = target_knumbers[ki]*np.array([0.98,0.99,1.00,1.01,1.02])
        # make sure target_wavenumber is in middle of above array!!
        target_knumber_arg = len(discrete_ks)//2;
        dos_of_w = np.zeros_like(ws_for_solution);
        for wvali in range(len(ws_for_solution)):
            discrete_band = np.array([ np.sqrt(uval**2 + vval**2 + ws_for_solution[wvali]**2 + 2*vval*ws_for_solution[wvali]*np.cos(discrete_ks)),
                                      -np.sqrt(uval**2 + vval**2 + ws_for_solution[wvali]**2 + 2*vval*ws_for_solution[wvali]*np.cos(discrete_ks))])
            dos_of_w[wvali] = 2/np.pi*abs(1/np.gradient(discrete_band[band_to_choose], discrete_ks))[target_knumber_arg];
            # for k/pi=0.04, w=-1 should give rho=10

        # from the local dos determination, we find w
        w_target_arg = np.argmin(abs(target_rhoJ-dos_of_w*abs(Jval)));
        ws_colors[ki] = ws_for_solution[w_target_arg]


        # plot dos as a function of w value
        dosax.plot(dos_of_w, ws_for_solution);
        dosax.axvline(target_rhoJ/abs(Jval), color=UniversalAccents[1], linestyle = "dashed");
        dosax.axhline(ws_colors[ki], color=UniversalAccents[1], linestyle="dashed");
             
        # plot and format the chosen dispersion (fixed wval)
        dispks = np.linspace(-np.pi, np.pi, len(ws_for_solution));
        disp = np.array([ np.sqrt(uval**2 + vval**2 + ws_colors[ki]**2 + 2*vval*ws_colors[ki]*np.cos(dispks)),
                                   -np.sqrt(uval**2 + vval**2 + ws_colors[ki]**2 + 2*vval*ws_colors[ki]*np.cos(dispks))])
        dispax.scatter(dispks/np.pi, disp[band_to_choose], color=UniversalAccents[0], marker=AccentsMarkers[0]); 
        # save for final plot E vs k
        disp_ofw.append(1*disp);
        
        # from targets, back out fixed quantities
        fixed_knumbers[ki] = dispks[np.argmin(abs(target_knumbers[ki]-dispks))]
        fixed_Energies[ki] = complex(disp[band_to_choose][np.argmin(abs(target_knumbers[ki]-dispks))])
        fixed_rhoJs[ki] = abs(Jval)*dos_of_w[w_target_arg];
        dispax.axvline(fixed_knumbers[ki]/np.pi, color=UniversalAccents[1], linestyle="dashed");
        dispax.axhline(np.real(fixed_Energies[ki]), color=UniversalAccents[1], linestyle="dashed");
        print("\nfixed_rhoJ = {:.6f}".format(fixed_rhoJs[ki]));
        print("fixed_Energy = {:.6f}".format(np.real(fixed_Energies[ki])));
        print("fixed_knumber/pi = {:.6f}".format(fixed_knumbers[ki]/np.pi));
 
        # plotting
        dosax.set_xlabel("$\\rho, \\rho J_{sd} a ="+"{:.2f}".format(fixed_rhoJs[ki])+", J_{sd} ="+"{:.2f}$".format( Jval), fontsize = myfontsize);
        dosax.set_xlim(0,100);
        dosax.set_ylabel("$w_{soln}$", fontsize = myfontsize)
        dispax.set_xlabel("$k_i a/\pi$", fontsize = myfontsize);
        dispax.set_ylabel("$E_\pm( k_i)$", fontsize = myfontsize);
        dosax.set_title("$u = {:.2f}, v = {:.2f}, w = {:.2f}$".format(uval, vval, ws_colors[ki]), fontsize = myfontsize);
    
        # show
        plt.tight_layout();
        plt.show();

        ####
        #### finally done determining w for this color set (rhoJa fixed value)
    

    # now we have graphically determined all targets
    # and stored them in fixed_
    del target_rhoJ, target_knumbers

    # d = number of lattice constants between MSQ 1 and MSQ 2
    kdalims = 0.01*np.pi, 2.1*np.pi; 
    kdavals = np.full((len(ws_colors),myxvals), np.nan, dtype=float);  
    Distvals = np.full((len(ws_colors),myxvals), np.nan, dtype=int);
    
    # iter over colors
    for colori, wval in enumerate(ws_colors):

        # Rice-Mele matrices
        diag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
        diag_base_RM_spin[:n_spin_dof,:n_spin_dof] = uval*np.eye(n_spin_dof);
        diag_base_RM_spin[n_spin_dof:,n_spin_dof:] = -uval*np.eye(n_spin_dof);
        diag_base_RM_spin[:n_spin_dof,n_spin_dof:] = vval*np.eye(n_spin_dof);
        diag_base_RM_spin[n_spin_dof:,:n_spin_dof] = vval*np.eye(n_spin_dof);
        offdiag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
        offdiag_base_RM_spin[n_spin_dof:,:n_spin_dof] = wval*np.eye(n_spin_dof);
        diag_base_nospin = diag_base_RM_spin[::n_spin_dof,::n_spin_dof];
        offdiag_base_nospin = offdiag_base_RM_spin[::n_spin_dof,::n_spin_dof];
        assert(abs(np.sum(np.diagonal(diag_base_nospin))/len(diag_base_nospin)) < 1e-10); # u0 = 0
        band_edges = wfm.bandedges_RiceMele(diag_base_nospin, offdiag_base_nospin)[-2:];
    
        # output Rice-Mele
        title_RiceMele = wfm.string_RiceMele(diag_base_nospin, offdiag_base_nospin, energies=False, tex=True)
        print("\n\nRice-Mele "+title_RiceMele);
        print("h00 =\n",diag_base_nospin);
        print("h01 =\n",offdiag_base_nospin);
        
        # determine the number of lattice constants across this range
        kdavals[colori,:] = np.linspace(*kdalims, myxvals);
        Distvals[colori,:] = np.rint(kdavals[colori]/fixed_knumbers[colori]).astype(int);
        
        # truncate to remove 0's, then extend back to length myxvals
        kdavals_trunc = kdavals[colori, Distvals[colori] > 0];
        kdavals[colori,:] = np.append(np.full((myxvals-len(kdavals_trunc),),kdavals_trunc[0]),kdavals_trunc); # extend
        Distvals_trunc = Distvals[colori, Distvals[colori] > 0];
        Distvals[colori,:] = np.append(np.full((myxvals-len(Distvals_trunc),),Distvals_trunc[0]),Distvals_trunc); # extend
        print("Nd values covered ({:.0f} total) =\n".format(len(Distvals[colori])),Distvals[colori]);
    
        # iter over Distvals to compute T
        for Distvali in range(len(Distvals[colori])):
        
            # construct hams
            i1, i2 = [1], [Distvals[colori, Distvali]+1];
            hblocks_noRM = h_cicc_reduced(Jval, i1, i2, i2[-1]+2, my_unit_cell, myspinS); 
            # ^ the +2 is for each lead site
            hblocks = 1*hblocks_noRM;
            tnn = np.zeros_like(hblocks);
            # add Rice Mele terms
            for blocki in range(len(hblocks)):
                hblocks[blocki] += diag_base_RM_spin;
                tnn[blocki] += offdiag_base_RM_spin;
            tnn = tnn[:-1];
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
            if(Distvali==0 and False): 
                print("hblocks =\n");
                blockstoprint = 3;
                for blocki in range(blockstoprint):
                    print("\n\n");
                    for chunki in range(my_unit_cell):
                        print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,chunki, chunki))
                        print(np.real(hblocks[blocki][chunki*n_spin_dof:(chunki+1)*n_spin_dof,chunki*n_spin_dof:(chunki+1)*n_spin_dof]));
                    print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,0, 1))
                    print(np.real(hblocks[blocki][0*n_spin_dof:(0+1)*n_spin_dof,1*n_spin_dof:(1+1)*n_spin_dof]));
                    print("t(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,1, 0))
                    print(np.real(tnn[blocki][n_spin_dof:,:n_spin_dof]));
            if(Distvali==0):
                print("J = {:.4f}".format(Jval));
                print("rhoJ = {:.4f}".format(fixed_rhoJs[colori]));
                print("max N = {:.0f}\n".format(np.max(Distvals[colori])+2));

            for sigmai in range(len(sigmas)):   
                # sourcei is one of the pairs always 
                source = np.zeros(my_unit_cell*n_spin_dof);
                source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet, impinging on A site

                # get  T coefs
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, abs(vval), fixed_Energies[colori], "g_RiceMele", 
                          source, False, False, all_debug = True);
                Tdum = Tdum[n_spin_dof:]; # extract only at boundary (B site for T)
                Rdum = Rdum[:n_spin_dof]; # extract only at boundary (A site for R)
                Tvals[colori, Distvali,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
                Tsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum);
                TpRsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);
                if(not(abs(1- TpRsummed[colori, Distvali,sigmas[sigmai]] )<1e-10)):
                    print( abs(1- TpRsummed[colori, Distvali,sigmas[sigmai]] )); assert False
                 
        ####
        #### end loop over MSQ-MSQ distance
    
    ####
    #### end loop over wvals (colors)
    
    # figure setup
    width_ratios = [0.7,0.3];
    numrows = 2;
    colorfig, axes_arr = plt.subplots(numrows, len(width_ratios), sharex = "col",
                       gridspec_kw = dict(width_ratios=width_ratios));
    colorfig.set_size_inches(6*np.sum(width_ratios),2*numrows); #aspect ratio of run_wfm_single plots= 6*sum,4
    color_gridspec = axes_arr[0,0].get_gridspec();
    axes_arr[0,0].remove(); # remove both the 1st col axes where we will then create one combined
    axes_arr[1,0].remove();
    # combine into colorax
    colorax = colorfig.add_subplot(color_gridspec[:,0]) # gridspec grabs all 1st column rows

    # create and format axis for dispersion
    #axes_arr[1,1].remove();
    legend_ax = axes_arr[1,1];
    legend_ax.set_xticks(np.arange(2));
    legend_ax.set_yticks(np.arange(2));
    legend_ax.axis("off");
    dispax = axes_arr[0,1];
    dispax.axis("off")
    dispax.axvline(0,color="black");
    dispax.axhline(0,color="black");
    dispax.text(0.5,1.1,"$E_\pm(k_\sigma)$",fontsize=myfontsize,transform=dispax.transAxes);
    dispax.text(1.1,0.5,"$k_\sigma$",fontsize=myfontsize,transform=dispax.transAxes);
    for colori in range(len(ws_colors)):
        dispax.plot(dispks, disp_ofw[colori][0], color=UniversalColors[colori]);
        dispax.plot(dispks, disp_ofw[colori][1], color=UniversalColors[colori]);
    
    # figure formatting
    title_with_Jsd = "$J_{sd} = "+"{:.2f}".format(Jval)+", E_{"+["-","+"][band_to_choose]+"}$ band";
    colorax.set_title(title_with_Jsd, fontsize=myfontsize);
    #colorax.set_ylabel("$T$", fontsize=myfontsize);
    colorax.set_ylim(0.0, 1.02);
    colorax.set_xlabel("$N_d k_i a / \pi$",fontsize=myfontsize);
    colorax.set_xticks( np.arange(int(np.rint(max(kdalims)+1))));
    colorax.set_xlim(0.0, max(kdalims)/np.pi);
    
    # efficiency
    efficiency_colors_N = (Tsummed[:,:,sigmas[1]]-Tsummed[:,:,sigmas[0]])/(Tsummed[:,:,sigmas[1]] + Tsummed[:,:,sigmas[0]]);
    
    # plot transmission coefficients vs N (1+MSQ-MSQ distance)
    # # |S>, |T0>, eta
    yvals_identifiers = ["$T(|S\\rangle )$", "$T(|T_0 \\rangle)$", "$\eta$"];
    yvals_to_plot = [Tsummed[:,:,sigmas[1]], Tsummed[:,:,sigmas[0]], efficiency_colors_N]; 
    yvals_styles = ["solid","dotted", "dashdot"];
    lines_to_legend_tuples = []; # append solid, dashed tuple for each color
    lines_to_legend_labels = [];
    for colori in range(len(ws_colors)):
    
        # x axis
        indep_vals = Distvals[colori]*fixed_knumbers[colori]/np.pi;
       
        # plot
        this_line_tuple = [];
        for stylei, yvals in enumerate(yvals_to_plot):
            # only label once per colori  
            if(stylei==0):
                style_label = "$k_i a/\pi = {:.2f}$".format(fixed_knumbers[colori]/np.pi); 
                style_label += "$, \\rho(k_i) J_{sd} a ="+"{:.1f}$".format(fixed_rhoJs[colori]);
                lines_to_legend_labels.append(style_label);
                
            # handle identifiers: one for each style
            if(colori==0):
                legend_ax.plot([np.nan], [np.nan], color="black",linestyle=yvals_styles[stylei], label = yvals_identifiers[stylei]);

            # plot line
            line_fromstyle, = colorax.plot(indep_vals, yvals[colori], label=style_label,
                                           color=UniversalColors[colori], marker=ColorsMarkers[colori], markevery=0.2+0.2*colori,linestyle=yvals_styles[stylei]);
            
            # handle line object for passing to legend
            this_line_tuple.append(line_fromstyle);
        lines_to_legend_tuples.append(tuple(this_line_tuple)); 

    # truncate tuples if not desiring to combine solid-dotted-dash lines in main legend
    if(True):
        for tupi in range(len(lines_to_legend_tuples)): 
            lines_to_legend_tuples[tupi] = (lines_to_legend_tuples[tupi][0],); 

    # show
    if(True):
        color_legend = colorax.legend(lines_to_legend_tuples, lines_to_legend_labels,
           #bbox_to_anchor =(0.00,1.02,1.00,0.102),loc="lower left",mode="expand",borderaxespad=0.0, 
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple( ndivide=None)},fontsize=myfontsize);
    legend_ax.legend(fontsize = myfontsize);
    plt.tight_layout();
    folder = "/home/cpbunker/Desktop/FIGS_Cicc_WFM/"
    fname = folder+"rho_fixed.pdf";
    print("Saving plot to "+fname);
    plt.savefig(fname);

elif(case in ["VB_spins"]): # entanglement *preservation* vs N, different colors for rho value
    my_unit_cell = 2; # since diatomic

    # Rice-Mele tight binding
    vval = -1.0; # sets energy scale
    wval = -1.0 # always
    uval = 0.0; # always 
    Jval = -0.05;
    # rhoJa = fixed throughout, thus fixing energy and wavenumber
    target_rhoJ = float(sys.argv[2]);
    # we will iter over spinS
    spinSvals = np.array(sys.argv[3:]).astype(float)
    myxvals = 2*myxvals

    # Rice-Mele matrices
    n_spin_dof = 3; # spin dofs
    diag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
    diag_base_RM_spin[:n_spin_dof,:n_spin_dof] = uval*np.eye(n_spin_dof);
    diag_base_RM_spin[n_spin_dof:,n_spin_dof:] = -uval*np.eye(n_spin_dof);
    diag_base_RM_spin[:n_spin_dof,n_spin_dof:] = vval*np.eye(n_spin_dof);
    diag_base_RM_spin[n_spin_dof:,:n_spin_dof] = vval*np.eye(n_spin_dof);
    offdiag_base_RM_spin = np.zeros((my_unit_cell*n_spin_dof, my_unit_cell*n_spin_dof),dtype=float);
    offdiag_base_RM_spin[n_spin_dof:,:n_spin_dof] = wval*np.eye(n_spin_dof);
    diag_base_nospin = diag_base_RM_spin[::n_spin_dof,::n_spin_dof];
    offdiag_base_nospin = offdiag_base_RM_spin[::n_spin_dof,::n_spin_dof];
    assert(abs(np.sum(np.diagonal(diag_base_nospin))/len(diag_base_nospin)) < 1e-10); # u0 = 0
    band_edges = wfm.bandedges_RiceMele(diag_base_nospin, offdiag_base_nospin)[-2:];
    
    # output Rice-Mele
    title_RiceMele = wfm.string_RiceMele(diag_base_nospin, offdiag_base_nospin, energies=False, tex=True)
    print("\n\nRice-Mele "+title_RiceMele);
    print("h00 =\n",diag_base_nospin);
    print("h01 =\n",offdiag_base_nospin);
                                   
    # channels
    pair = (0,1); # pair[0] = |+> channel, pair[1] = |-> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement
                                          # in this case, elec up, MSQs in triplet, singlet
                                          # source must impinge on A orbital 
                                    
    # return values
    # shaped by fixed rhoJval (color), MSQ-MSQ distance (x axis), spin dofs 
    # Transmission coefficients. Note:
        # we compute only source channel -> source channel scattering, leave the rest as NaNs
        # we evaluate T at SR boundary, namely the B site
    Tvals = np.full((len(spinSvals),myxvals,n_spin_dof), np.nan, dtype=float);
   
    # Tsummed measures source channel -> all channels transmission
    Tsummed = np.full((len(spinSvals),myxvals,n_spin_dof), np.nan, dtype=float);
    TpRsummed = np.full((len(spinSvals),myxvals,n_spin_dof), np.nan, dtype=float); 
    
    # d = number of lattice constants between MSQ 1 and MSQ 2
    kdalims = 0.01*np.pi, 2.1*np.pi; 
    widelimsflag = False;
    try: 
        if(sys.argv[5]=="widelims"): widelimsflag = True;
    except: print(">>> Not flagged to widen kda limits");
    if(widelimsflag): kdalims = 0.01*np.pi, 100*np.pi; 
    kdavals = np.full((myxvals), np.nan, dtype=float);  
    Distvals = np.full((myxvals), np.nan, dtype=int);  

    if(True):      
        # graphical dispersion for fixed energy
        fig, (dispax, dosax) = plt.subplots(ncols=2, sharey = True); myfontsize = 14;
        Ks_for_solution = np.logspace(-6,1,499); # creates a discrete set of energy points, 
                                             # logarithmic w/r/t desired band edge
        if(case in ["CB_spins"]): 
            discrete_band = np.min(band_edges)+Ks_for_solution;
            discrete_band = discrete_band[discrete_band < np.max(band_edges)]; # stay w/in conduction band
        elif(case in ["VB_spins"]): 
            discrete_band = np.min(-band_edges)+Ks_for_solution;
            discrete_band = discrete_band[discrete_band < np.max(-band_edges)]; # stay w/in valence band
        else: raise NotImplementedError("case = "+case);
        dispks = np.linspace(-np.pi, np.pi,myxvals);
        disp = wfm.dispersion_RiceMele(diag_base_nospin, offdiag_base_nospin, dispks);
        # plot and format the dispersion
        for dispvals in disp: dispax.plot(dispks/np.pi, dispvals,color="cornflowerblue");
    
        # highlight the parts of the band we are considering
        discrete_ks = np.arccos(1/(2*vval*wval)*(discrete_band**2 - uval**2 - vval**2 - wval**2))
        dispax.scatter(discrete_ks/np.pi, discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]); 

        # graphical density of states for fixed energy
        discrete_dos = 2/np.pi*abs(1/np.gradient(discrete_band, discrete_ks));

        dosax.scatter(discrete_dos,discrete_band, color=UniversalAccents[0], marker=AccentsMarkers[0]);
        dosline_from_rhoJ = target_rhoJ/abs(Jval);
        # solve graphically for fixed E *specifically in VB/CB* that gives desired rhoJa
        fixed_Energy = complex(discrete_band[np.argmin(abs(discrete_dos-dosline_from_rhoJ))],0);
        # ^ grabs one of the discrete energy points in this_band, based on having closest to desired rho(E)
        # v grabs corresponding k and rho(E) values
        fixed_knumber = discrete_ks[np.argmin(abs(discrete_dos-dosline_from_rhoJ))];
        fixed_rhoJ = discrete_dos[np.argmin(abs(discrete_dos-dosline_from_rhoJ))]*abs(Jval);
        # NB we use this, the closest discrete rhoJ, rather than command-line rhoJ
        del target_rhoJ;
        print("\nfixed_rhoJ = {:.6f}".format(fixed_rhoJ));
        print("fixed_Energy = {:.6f}".format(np.real(fixed_Energy)));
        print("fixed_knumber = {:.6f}".format(fixed_knumber));
        dosax.axvline(dosline_from_rhoJ, color=UniversalAccents[1], linestyle = "dashed");
        dosax.axhline(np.real(fixed_Energy), color=UniversalAccents[1], linestyle="dashed");
        dispax.axvline(fixed_knumber/np.pi, color=UniversalAccents[1], linestyle = "dashed");
        dispax.axhline(np.real(fixed_Energy), color=UniversalAccents[1], linestyle="dashed");

        # plotting
        if(case in ["VB_spins"]): 
            RiceMele_band = "-";
            RiceMele_shift_str = "$, E_{min}^{(VB)}="+"{:.2f}$".format(np.min(-band_edges))
        elif(case in ["CB_spins"]): 
            RiceMele_band = "+";
            RiceMele_shift_str="$,  E_{min}^{(CB)}="+"{:.2f}$".format(np.min(band_edges))
        RiceMele_shift_str += ",  $k_i a/\pi \in [{:.2f},{:.2f}]$".format(discrete_ks[0]/np.pi, discrete_ks[-1]/np.pi);
        dispax.set_ylabel("$E_\pm( k_i)$"+RiceMele_shift_str, fontsize = myfontsize);
        dosax.set_xlabel("$\\rho, \\rho J_{sd} a ="+"{:.2f}".format(fixed_rhoJ)+", J_{sd} ="+"{:.2f}$".format( Jval), fontsize = myfontsize);
        dosax.set_xlim(0,10);
        dispax.set_xlabel("$k_i a/\pi$", fontsize = myfontsize);
        dispax.set_title(title_RiceMele, fontsize = myfontsize);
    
        # show
        plt.tight_layout();
        plt.show();
        stopflag = False;
        try: 
            if(sys.argv[-1]=="stop"): stopflag = True;
        except: print(">>> Not flagged to stop");
        assert(not stopflag); 
        
        ####
        #### finally done determining energy, wavenumber for this color set (rhoJa fixed value)
    
        # determine the number of lattice constants across this range
        kdavals[:] = np.linspace(*kdalims, myxvals);
        Distvals[:] = np.rint(kdavals/fixed_knumber).astype(int);
        
        # truncate to remove 0's, then extend back to length myxvals
        kdavals_trunc = kdavals[Distvals > 0];
        kdavals[:] = np.append(np.full((myxvals-len(kdavals_trunc),),kdavals_trunc[0]),kdavals_trunc); # extend
        Distvals_trunc = Distvals[Distvals > 0];
        Distvals[:] = np.append(np.full((myxvals-len(Distvals_trunc),),Distvals_trunc[0]),Distvals_trunc); # extend
        print("Nd values covered ({:.0f} total) =\n".format(len(Distvals)),Distvals);

    # iter over MSQ spin
    for colori, Sval in enumerate(spinSvals):
    
        # iter over Distvals to compute T
        for Distvali in range(len(Distvals)):
        
            # construct hams
            i1, i2 = [1], [Distvals[Distvali]+1];
            hblocks_noRM = h_cicc_reduced(Jval, i1, i2, i2[-1]+2, my_unit_cell, Sval); 
            # ^ the +2 is for each lead site
            hblocks = 1*hblocks_noRM;
            tnn = np.zeros_like(hblocks);
            # add Rice Mele terms
            for blocki in range(len(hblocks)):
                hblocks[blocki] += diag_base_RM_spin;
                tnn[blocki] += offdiag_base_RM_spin;
            tnn = tnn[:-1];
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
            if(Distvali==0): 
                print("hblocks =\n");
                blockstoprint = 3;
                for blocki in range(blockstoprint):
                    print("\n\n");
                    for chunki in range(my_unit_cell):
                        print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,chunki, chunki))
                        print(np.real(hblocks[blocki][chunki*n_spin_dof:(chunki+1)*n_spin_dof,chunki*n_spin_dof:(chunki+1)*n_spin_dof]));
                    print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,0, 1))
                    print(np.real(hblocks[blocki][0*n_spin_dof:(0+1)*n_spin_dof,1*n_spin_dof:(1+1)*n_spin_dof]));
                    print("t(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,1, 0))
                    print(np.real(tnn[blocki][n_spin_dof:,:n_spin_dof]));
                print("J = {:.4f}".format(Jval));
                print("rhoJ = {:.4f}".format(fixed_rhoJ));
                print("max N = {:.0f}\n".format(np.max(Distvals)+2));

            for sigmai in range(len(sigmas)):   
                # sourcei is one of the pairs always 
                source = np.zeros(my_unit_cell*n_spin_dof);
                source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet, impinging on A site

                # get  T coefs
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, abs(vval), fixed_Energy, "g_RiceMele", 
                          source, False, False, all_debug = True);
                Tdum = Tdum[n_spin_dof:]; # extract only at boundary (B site for T)
                Rdum = Rdum[:n_spin_dof]; # extract only at boundary (A site for R)
                Tvals[colori, Distvali,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
                Tsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum);
                TpRsummed[colori, Distvali,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);
                if(not(abs(1- TpRsummed[colori, Distvali,sigmas[sigmai]] )<1e-10)):
                    print( abs(1- TpRsummed[colori, Distvali,sigmas[sigmai]] )); assert False
                 
        ####
        #### end loop over MSQ-MSQ distance
    
    ####
    #### end loop over S (colors)
    
    width_ratios = [0.7,0.3];
    numrows = 2;
    colorfig, axes_arr = plt.subplots(numrows, len(width_ratios), sharex = "col",
                       gridspec_kw = dict(width_ratios=width_ratios));
    colorfig.set_size_inches(6*np.sum(width_ratios),2*numrows); #aspect ratio of run_wfm_single plots= 6*sum,4
    color_gridspec = axes_arr[0,0].get_gridspec();
    axes_arr[0,0].remove(); # remove both the 1st col axes where we will then create one combined
    axes_arr[1,0].remove();
    # combine into colorax
    colorax = colorfig.add_subplot(color_gridspec[:,0]) # gridspec grabs all 1st column rows

    # create and format axis for dispersion
    #axes_arr[1,1].remove();
    legend_ax = axes_arr[1,1];
    legend_ax.set_xticks(np.arange(2));
    legend_ax.set_yticks(np.arange(2));
    legend_ax.axis("off");
    dispax = axes_arr[0,1];
    dispax.axis("off")
    dispax.axvline(0,color="black");
    dispax.axhline(0,color="black");
    dispax.text(0.5,1.1,"$E_\pm(k_\sigma)$",fontsize=myfontsize,transform=dispax.transAxes);
    dispax.text(1.1,0.5,"$k_\sigma$",fontsize=myfontsize,transform=dispax.transAxes);
    for bandi in range(len(disp)):
        dispax.plot(dispks, disp[bandi], color=UniversalAccents[1]);

    
    # figure
    colorax.set_title("$J_{sd} = "+"{:.2f}".format(Jval)+", k_i a/\pi = "+"{:.2f}$".format(fixed_knumber/np.pi), fontsize=myfontsize);
    colorax.set_ylim(0.0, 1.0);
    colorax.set_xlabel("$N_d k_i a / \pi$",fontsize=myfontsize);
    colorax.set_xticks( np.arange(int(np.rint(max(kdalims)+1))));
    colorax.set_xlim(0.0, max(kdalims)/np.pi);
    
    # efficiency
    efficiency_colors_N = (Tsummed[:,:,sigmas[1]]-Tsummed[:,:,sigmas[0]])/(Tsummed[:,:,sigmas[1]] + Tsummed[:,:,sigmas[0]]);
        
    # plot transmission coefficients vs N (1+MSQ-MSQ distance)
    # to_plot = |T0>,|S>, eta
    yvals_identifiers = ["$T(|S\\rangle )$", "$T(|T_0 \\rangle)$"]#, "$\eta$"];
    yvals_to_plot = [Tsummed[:,:,sigmas[1]], Tsummed[:,:,sigmas[0]]]#, efficiency_colors_N]; 
    yvals_styles = ["solid","dotted"]#,"dashdot"];
    lines_to_legend_tuples = []; # append solid, dashed tuple for each color
    lines_to_legend_labels = [];    
    for colori in range(len(spinSvals)):
    
        # x axis
        indep_vals = Distvals*fixed_knumber/np.pi;
        
        # plot
        this_line_tuple = [];
        for stylei, yvals in enumerate(yvals_to_plot):
            # handle label--only label once per colori  
            if(stylei==0):
                lines_to_legend_labels.append("$s = {:.0f}/2$".format(2*spinSvals[colori]));
                
            # handle identifiers: one for each style
            if(colori==0):
                legend_ax.plot([np.nan], [np.nan], color="black",linestyle=yvals_styles[stylei], label = yvals_identifiers[stylei]);
                
            # plot line
            line_fromstyle, = colorax.plot(indep_vals, yvals[colori], color=UniversalColors[colori], linestyle=yvals_styles[stylei]);
            
            # handle line object for passing to legend
            this_line_tuple.append(line_fromstyle);
        lines_to_legend_tuples.append(tuple(this_line_tuple)); 

    # truncate tuples if not desiring to combine solid-dotted-dash lines in main legend
    if(True):
        for tupi in range(len(lines_to_legend_tuples)): 
            lines_to_legend_tuples[tupi] = (lines_to_legend_tuples[tupi][0],); 

    # show
    if(True):
        color_legend = colorax.legend(lines_to_legend_tuples, lines_to_legend_labels,
           #bbox_to_anchor =(0.00,1.02,1.00,0.102),loc="lower left",mode="expand",borderaxespad=0.0, 
           handler_map={tuple: matplotlib.legend_handler.HandlerTuple( ndivide=None)},fontsize=myfontsize);
    legend_ax.legend(fontsize = myfontsize);
    plt.tight_layout();
    folder = "/home/cpbunker/Desktop/FIGS_Cicc_WFM/"
    fname = folder+'vsS.pdf';
    print("Saving plot to "+fname);
    plt.savefig(fname);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##################################################################################
#### entanglement generation (cicc Fig 6)

elif(case in ["Nx"]): # compare T vs rhoJa for N not fixed

    # iter over E, getting T
    logElims = -4,-1
    Evals = np.logspace(*logElims,myxvals,dtype=complex);
    Rvals = np.empty((len(Evals),len(source)), dtype = float);
    Tvals = np.empty((len(Evals),len(source)), dtype = float);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions       
        # location of impurities, fixed by kx0 = pi
        k_rho = np.arccos(Energy/(-2*tl)); # = ka since \varepsilon_0ss = 0
        kx0 = 2.0*np.pi;
        N0 = max(1,int(kx0/(k_rho))); #N0 = (N-1)
        print(">>> N0 = ",N0);

        # construct hams
        i1, i2 = 1, 1+N0;
        hblocks, tnn = h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(Evali==0): 
            print("shape(hblocks) = ",np.shape(hblocks));
            print("sourcei = ",sourcei);
            
        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
        Rvals[Evali] = Rdum;
        Tvals[Evali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = np.real(Evals);
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/Nx/"+str(int(kx0*100)/100);
    print("Saving data to "+fname);
    np.save(fname, data);


elif(case in ["N2","N2_k"]): # compare T vs rhoJa for N=2 fixed

    # channels
    pair = (1,2); # following entanglement change of basis, pair[0] = |+> channel
    source = np.zeros(8); 
    sourcei = 4; # down up up - the initial state when *generating* entanglement
    sigmas = [pair[0],pair[1],sourcei]; # all the channels of interest to generating entanglement
                                        # NB the electron spin is well-defined
    source[sourcei] = 1; 

    # tight binding params
    tl = 1.0;
    Jval = -0.5*tl/100;
    Esplit = 0.0;
    Delta_zeeman = -Esplit; # Zeeman is only way to have energy difference btwn channels for spin-1/2
    
    # set number of lattice constants between MSQ 1 and MSQ
    # here it is fixed to be 1 bc N (number of sites in SR) is fixed at 2
    Distval = 1;
    
    # energy of the incident electron
    K_indep = True; # puts energy above the bottom of the band (logarithmically) on x axis
    if(case in ["N2_k"]): K_indep = False; # puts wavenumber on the x axis
    if(K_indep):               
        logKlims = -6, -4
        Kvals = np.logspace(*logKlims,myxvals, dtype = complex); # K > 0 always
        knumbers = np.arccos((Kvals-2*tl)/(-2*tl));
        indep_vals = np.real(Kvals);
    else:
        knumberlims = 0.1*(np.pi/Distval), 0.9*(np.pi/Distval);
        # ^ since we cannot exceed K = 4t, we cannot exceed k = \pi
        if(Distval == 1): knumberlims = 1e-3, 1e-2; # agrees with logKlims
        knumbers = np.linspace(knumberlims[0], knumberlims[1], myxvals, dtype=complex);
        Kvals = 2*tl - 2*tl*np.cos(knumbers);
        indep_vals = np.real(knumbers)/(np.pi/Distval);
    print("Kvals \in ",Kvals[0], Kvals[-1]);
    print("knumbers \in ",knumbers[0], knumbers[-1]);
    print("indep_vals \in ",indep_vals[0], indep_vals[-1]);

    # iter over E, getting T
    Rvals = np.empty((len(Kvals),len(source)), dtype = float);
    Tvals = np.empty((len(Kvals),len(source)), dtype = float);
    for Kvali in range(len(Kvals)):

        # energy
        Kval = Kvals[Kvali]; # Kval > 0 always, what I call Ki in paper
        Energy = Kval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [Distval+1];
        hblocks, tnn = h_cicc_eff(Jval, tl, i1, i2, i2[-1]+2, pair); # full 8 by 8, not reduced
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(Kvali==0): 
            print("shape(hblocks) = ",np.shape(hblocks));
            print("sourcei = ",sourcei);
        
        # Zeeman splitting effects. NB s=1/2 so 2s-1=0
        hzeeman = np.zeros_like(hblocks[0]);
        hzeeman[sourcei, sourcei] = Delta_zeeman;
        for hbi in range(len(hblocks)): hblocks[hbi] += np.copy(hzeeman);
        # shift so hblocks[0,i,i] = 0
        Eshift = hblocks[0,sourcei, sourcei];
        for hbi in range(len(hblocks)): hblocks[hbi] += -Eshift*np.eye(len(hblocks[0]));
        if(verbose > 3 and Kvali == 0): print("hblocks =\n",np.real(hblocks));

        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, False, False, all_debug = False);
        Rvals[Kvali] = Rdum;
        Tvals[Kvali] = Tdum;

 
