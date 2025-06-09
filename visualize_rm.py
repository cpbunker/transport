'''
'''

from transport import wfm, tdfci, tddmrg

import numpy as np
import matplotlib.pyplot as plt

import sys
import json

def tb_hamiltonian(the_params, initialize, verbose=0):
    '''
    Tight binding Hamiltonian of spinless system
    '''

    # construct Time=0 single-body Hamiltonian as matrix
    if(the_params["sys_type"] in ["SIETS_RM"]):
        h1e_t0, g2e_dummy = tddmrg.H_RM_builder(the_params, block=False);
    elif(the_params["sys_type"] in ["STT_RM"]):
        h1e_t0, g2e_dummy = tddmrg.H_STTRM_builder(the_params, block=False);

    # if Time = 0, add polarizer
    if(initialize):
        if(the_params["sys_type"] in ["SIETS_RM"]):
            h1e_t0, g2e_dummy = tddmrg.H_RM_polarizer(the_params, (h1e_t0, g2e_dummy), block=False);
        elif(the_params["sys_type"] in ["STT_RM"]):
            h1e_t0, g2e_dummy = tddmrg.H_STTRM_polarizer(the_params, (h1e_t0, g2e_dummy), block=False);

    # eigenstates
    del g2e_dummy
    h = h1e_t0[::2,::2]; # <- make spinless !!
    Evals, Evecs = tdfci.solver(h);
    Evals_offset = 0.0;
    if(initialize): Evals_offset = the_params["Vconf"]+abs(the_params["Be"])/2;
    Evals += Evals_offset

    h00 = np.array([[the_params["u"], the_params["v"]],[the_params["v"],-the_params["u"]]]);
    h01 = np.array([[0,0],[the_params["w"],0]]);
    kvals = wfm.inverted_RiceMele(h00, h01, Evals);
    return kvals, Evals, Evecs, h;

def get_occs_Ne_TwoSz(the_Ne, the_TwoSz, num_skip = 0):
    '''
    '''
    assert(the_Ne % 2 == the_TwoSz % 2);
    # lower energy levels that we skip and leave empty
    skip_levels = [];
    for leveli in range(num_skip):
        skip_levels.append(0);
    # spin_polarized levels
    spinpol_levels = [];
    for _ in range(the_TwoSz):
        spinpol_levels.append(1);
    # non spin polarized (double occupied) levels
    nonpol_levels = [];
    for _ in range((the_Ne - the_TwoSz)//2):
        nonpol_levels.append(2);
    return np.append(skip_levels, np.append(nonpol_levels, spinpol_levels)).astype(int);

def get_overlaps(the_params, the_occs, plotwfs=False, plothams=True):
    '''
    Main purpose of this function is to return
    knvals, array of wavenumbers describing single-e eigstates of time>0 system
    the_pdfs, an array of overlaps each kn eigenstate has with the initial state
      NB the initial state is just filling the Ne lowest energy out of all
      of the km states, which are the single-e eigstates of time=0 system
      
    '''
    if(not isinstance(the_occs, np.ndarray)): raise TypeError;
    if(the_occs.dtype != int): raise TypeError;
    n_subplots_here = (2,2);
    the_fig, the_axes = plt.subplots(*n_subplots_here);
    if(n_subplots_here == 1): the_axes = [the_axes];
    the_fig.set_size_inches(5*n_subplots_here[1],3*n_subplots_here[0]);
    

    # t<0 eigenstates (|k_m> states)
    m_HOMO = len(the_occs) - 1; # m value of the highest occupied k state
    kmvals, Emvals, Emvecs, h1e_t0 = tb_hamiltonian(the_params, initialize=True, verbose=1); 
    the_occs_final = np.zeros_like(Emvals, dtype = the_occs.dtype);
    the_occs_final[:len(the_occs)] = the_occs[:];
    print(the_occs,"-->\n", the_occs_final);
    the_occs = 1*the_occs_final; del the_occs_final;
    
    ms_to_probe = [max(0,m_HOMO-2), min(m_HOMO,len(Emvecs)-1)+1];
    ms_to_probe = range(ms_to_probe[0], ms_to_probe[1]);
    for m in ms_to_probe:
        the_axes[0,0].plot(Emvecs[m], label="$m ={:.0f}, k_m ={:.4f}$".format(m, kmvals[m]));
    diag_norm_t0 = np.max(abs(np.diag(h1e_t0)))
    if(diag_norm_t0 < 1e-10): diag_norm_t0 = 1;
    if(plothams):
        the_axes[0,0].plot(np.diag(h1e_t0)/diag_norm_t0,color="black")#,label="$\langle j | V |j \\rangle$"); # onsite
        the_axes[0,0].plot(np.diag(h1e_t0,k=1),color="teal",marker="o")#,label="$\langle j | V |j+1 \\rangle$"); # hopping
    else: the_axes[0,0].set_ylim(-0.4,0.4);
    the_axes[0,0].legend();
    the_axes[0,0].set_xlabel("Site");
    the_axes[0,0].set_ylabel("$\langle j | k_m \\rangle$");
    the_axes[0,0].set_title("$u ={:.2f}, v ={:.2f}, w ={:.2f}$".format(the_params["u"], the_params["v"], the_params["w"]));

    # t>0 eigenstates (|k_n> states)
    knvals, Envals, Envecs, h1e = tb_hamiltonian(the_params, initialize=False, verbose=0);
    band_divider = len(knvals)//2;
    for n in range(4):
        the_axes[1,0].plot(Envecs[n], label="$n ={:.0f}, k_n={:.4f}$".format(n,knvals[n]));
    diag_norm = np.max(abs(np.diag(h1e)));
    if(diag_norm < 1e-10): diag_norm = 1;
    if(plothams):
        the_axes[1,0].plot(np.diag(h1e)/diag_norm,color="black")#,label="$\langle j | V |j \\rangle$"); # onsite
        the_axes[1,0].plot(np.diag(h1e,k=1),color="teal",marker="o")#,label="$\langle j | V |j+1 \\rangle$"); # hopping
    else: the_axes[1,0].set_ylim(-0.4,0.4);
    the_axes[1,0].legend(loc="lower right");
    the_axes[1,0].set_xlabel("Site");
    the_axes[1,0].set_ylabel("$\langle j | k_n \\rangle$");
    print("km values ({:.0f} conf, {:.0f} total) =\n".format(the_params["Nconf"], len(kmvals)),kmvals[:band_divider]);
    print("Em values ({:.0f} conf, {:.0f} total) =\n".format(the_params["Nconf"], len(Emvals)),Emvals[:band_divider]);
    print("kn values ({:.0f} band, {:.0f} total) =\n".format(band_divider, len(knvals)),knvals[:band_divider]);
    print("En values ({:.0f} band, {:.0f} total) =\n".format(band_divider, len(Envals)),Envals[:band_divider]);

    #### 3) occupation of the t<0 states
    the_axes[0,1].scatter(Emvals[:band_divider], the_occs[:band_divider],marker="s", color="red");
    the_axes[0,1].scatter(Emvals[band_divider:], the_occs[band_divider:],marker="s", color="red");
    the_axes[0,1].set_xlabel("$E_m$");
    the_axes[0,1].set_ylabel("$n_m$");
    the_axes[0,1].set_ylim(0,2.0);

    ####  4) overlap of the *occupied* t<0 states with t>0 k states
    the_pdfs = np.zeros((len(knvals),),dtype=float);
    for knvali in range(len(knvals)):
        # iter over *occupied* |k_m> states
        for kmvali in range(len(kmvals)):
            overlap = np.dot( np.conj(Emvecs[kmvali]), Envecs[knvali]);
            the_pdfs[knvali] += the_occs[kmvali]*np.real(np.conj(overlap)*overlap);
    # overlap plotted vs E_n 
    the_axes[1,1].plot(Envals[:band_divider], the_pdfs[:band_divider], color="black");
    the_axes[1,1].plot(Envals[band_divider:], the_pdfs[band_divider:], color="black");
    the_axes[1,1].set_xlabel("$E_n$");
    the_axes[1,1].set_ylabel("$ \sum_{m}^{occ} |\langle k_m | k_n \\rangle |^2$");
    the_axes[1,1].set_ylim(0,2.0);

    the_axes[1,1].scatter(Emvals, the_occs,marker="s", color="red");
    the_axes[1,1].set_xlim(np.min(Envals), np.max(Envals));
    
    # visualize the k_m and k_n wavefunctions in real space
    the_fig.suptitle("$N_e ={:.0f}".format(np.sum(the_occs))+", m_{HOMO} ="+"{:.0f}".format(m_HOMO)
        +", N_{conf}="+"{:.0f}".format(the_params["Nconf"])+", V_{conf} ="+"{:.2f}$".format(the_params["Vconf"]));
    if(plotwfs): plt.tight_layout(), plt.show();
    else: plt.close();
   
    # charge density in real space - has some domain as eigenvector
    charge_pdf = np.zeros_like(Emvecs[0], dtype=float);
    # iter over *occupied* |k_m> states
    for kmvali in range(len(kmvals)):
        charge_pdf += the_occs[kmvali]*np.conj(Emvecs[kmvali])*Emvecs[kmvali];
        
    # real-space spin PDF
    spin_pdf = np.zeros_like(Emvecs[0], dtype=float);
    # iter over *occupied* |k_m> states
    for kmvali in range(len(knvals)):
        spin_pdf += the_occs[kmvali]*(2-the_occs[kmvali])*np.conj(Emvecs[kmvali])*Emvecs[kmvali];
        
    # return
    return Emvals, kmvals, Envals, knvals, the_pdfs, charge_pdf, spin_pdf;

def wvals_to_rhoEF(wvals, the_params, plot=False):
    '''
    '''

    rhovals = np.empty_like(wvals);
    levels_skip = 0; # fixed
    Ne, TwoSz = the_params["Ne"], the_params["TwoSz"];
    
    # set up figure
    fig, axes = plt.subplots();
    fig.set_size_inches(*UniversalFigRatios);
    occax = axes; 

    for pairi in range(len(wvals)): # iter over w

        # repack overridden params
        the_params["w"] = wvals[pairi];

        # occupation
        my_occs = get_occs_Ne_TwoSz(Ne, TwoSz, num_skip = levels_skip);
        m_HOMO = len(my_occs)-1;
        
        # get time<0 and time>0 wavefunctions etc
        myEm, mykm, myEn, mykn, _, _, _ = get_overlaps(the_params, my_occs, plotwfs=False);
        t0_occs = np.zeros((len(myEm),), dtype=int);
        t0_occs[:len(my_occs)] = my_occs[:];
        
        # plot time<0 wf occupation
        mylabel = "$v = {:.2f}, w = {:.2f}$".format( myparams["v"], myparams["w"]);
        occax.plot(t0_occs[:len(myEm)//2],myEm[:len(myEm)//2],label=mylabel, color=UniversalColors[pairi]);
        occax.plot(t0_occs[len(myEm)//2:],myEm[len(myEm)//2:], color=UniversalColors[pairi]);
        if(wvals[pairi] == myparams["v"]): occax.set_ylim(1.05*np.min(myEn), 1.05*np.max(myEn));

        # dos of the Em
        myEm_gradient = np.array([np.gradient(myEm[:len(myEm)//2],mykm[:len(myEm)//2]),
                                  np.gradient(myEm[len(myEm)//2:],mykm[len(myEm)//2:])]);
        occax.plot((2/np.pi)*1/abs(myEm_gradient[0]), myEm[:len(myEm)//2], color=UniversalColors[pairi],linestyle="dashed");
        occax.plot((2/np.pi)*1/abs(myEm_gradient[1]), myEm[len(myEm)//2:], color=UniversalColors[pairi],linestyle="dashed");
        rhovals[pairi] = (2/np.pi)*1/abs(myEm_gradient[0,m_HOMO])
        occax.text(rhovals[pairi], myEm[m_HOMO], "$\\rho(E_F) = {:.2f}$".format(rhovals[pairi]),color=UniversalColors[pairi]);
        

    # format
    occax.set_xlim(0.0, 3.0);
    occax.set_xlabel("$n_m, \,\, \\rho(E_m)$");
    occax.set_ylabel("$E_m$");
    occax.set_title("$N_e = {:.0f}$".format(myparams["Ne"])+", $N_{conf} =$"+"{:.0f}".format(myparams["Nconf"]));
        
    # show
    occax.legend(fontsize=myfontsize);
    plt.tight_layout();
    if(plot): plt.show();
    else: plt.close();

    # return
    return rhovals;

if(__name__ == "__main__"):
    
    # top level
    verbose = 2; assert verbose in [1,2,3];
    np.set_printoptions(precision = 6, suppress = True);
    myxvals = 999;
    case = sys.argv[1];
    json_name = sys.argv[2];
    try:
        try:
            myparams = json.load(open(json_name+".txt"));
        except:
            myparams = json.load(open(json_name));
            json_name = json_name[:-4];
        print(">>> Params = ",myparams);
    except:
        raise Exception(json_name+" cannot be found");

    # fig standardizing
    from transport.wfm import UniversalColors, UniversalAccents, ColorsMarkers, AccentsMarkers, UniversalMarkevery, UniversalPanels;
    mylinewidth = 1.0;
    myfontsize = 14;
    plt.rcParams.update({"font.family": "serif"});
    #plt.rcParams.update({"text.usetex": True});
    UniversalFigRatios = [4.5,5.5/1.25];
    
else: # name NOT main 
      # code here just keeps this script from failing when it is imported
    case = "None";

#### Rice Mele Time > 0 occupation distribution
if(case in ["None"]): pass;

elif(case in ["rhoEF"]):

    # parameters
    ws = np.array([-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,-0.1]);
    
    # plot rho(EF) vs w values
    rhofig, rhoax = plt.subplots();
    rhofig.set_size_inches(*UniversalFigRatios[::-1]);

    # for different fill factors
    Nes_override = [10,16]
    for Neindex in range(len(Nes_override)):
        # override
        myparams["Ne"] = Nes_override[Neindex];
        fill_factor = myparams["Ne"]/(2*myparams["Nconf"])

        # plot
        rhos = wvals_to_rhoEF(ws, myparams);
        rhoax.plot(ws, rhos, label="$N_e/N_{band} ="+"{:.2f}$".format(fill_factor),color=UniversalColors[Neindex], marker=ColorsMarkers[Neindex]);
    
    # format
    rhoax.set_xlabel("$w/|v|$", fontsize = myfontsize);
    rhoax.set_ylabel("$\\rho(E_F)$", fontsize = myfontsize);
    
    # show
    rhoax.legend(fontsize=myfontsize);
    plt.tight_layout();
    savename = "/home/cpbunker/Desktop/FIGS_Cicc_with_DMRG/"+case+".pdf";
    #print("Saving to "+savename);
    #plt.savefig(savename);
    plt.show();
    
elif(case in ["distroNconf"]):

    # parameters
    # wval is fixed
    my_levels_skip = 0; # fixed
    myNe, myTwoSz = myparams["Ne"], myparams["TwoSz"];
    Nconfvals = [2,5,10,20];

    # whether to plt vs En or kn
    plot_vs_kn = True;
    
    # set up figure
    fig, axes = plt.subplots();
    fig.set_size_inches(*UniversalFigRatios)
    distax = axes;
    if(plot_vs_kn): distax.set_ylabel("$k_n a/\pi$",fontsize=myfontsize);
    else: distax.set_ylabel("$E_n$",fontsize=myfontsize);
    distax.set_xlabel("$\sum_m |\langle k_m|k_n \\rangle |^2$",fontsize=myfontsize);

    for pairi in range(len(Nconfvals)): # iter over Nconf

        # repack overridden params
        myparams["Nconf"] = Nconfvals[pairi];

        # occupation
        my_occs = get_occs_Ne_TwoSz(myNe, myTwoSz, num_skip = my_levels_skip);

        # get time=0 and time>0 wavefunctions etc
        _, _, my_E, my_k, my_dist, _, _ = get_overlaps(myparams, my_occs, plotwfs=False);
        if(plot_vs_kn): indep_vals = 1*my_k/np.pi;
        else: indep_vals = 1*my_E

        # plot
        mylabel = "$N_{conf} = $"+"{:.0f}".format(myparams["Nconf"]);
        distax.plot(my_dist[:len(my_E)//2],indep_vals[:len(my_E)//2],label=mylabel, color=UniversalColors[pairi]);
        distax.plot(my_dist[len(my_E)//2:],indep_vals[len(my_E)//2:], color=UniversalColors[pairi]);

    # format
    distax.set_xticks([]);
    distax.set_title("$N_e = {:.0f}, v = {:.2f}, w = {:.2f}$".format(myparams["Ne"], myparams["v"], myparams["w"]));
    if(myparams["Ne"]==1 and not plot_vs_kn): distax.set_ylim(np.min(indep_vals), 0.0);
    ktarget = np.pi/20;
    h00 = np.array([[myparams["u"], myparams["v"]],[myparams["v"],-myparams["u"]]]);
    h01 = np.array([[0,0],[myparams["w"],0]]);
    Etarget = wfm.dispersion_RiceMele(h00, h01, [ktarget])[0];
    if(plot_vs_kn): distax.axhline(ktarget/np.pi, color="black", linestyle="dashed");
    else: distax.axhline(Etarget, color="black", linestyle="dashed");
    
    # show
    distax.legend();
    plt.tight_layout();
    plt.show();
 
elif(case in ["distroNe"]):

    # iter over Ne, excited_state values
    pairvals = np.array([(1,1),(2,0),(5,5), (10,0)]);
    my_levels_skip = 0; # fixed
    # ^ Nconf=20 sites (10 blocks) fixed,

    for pairi in range(len(pairvals)):

        # unpack overriden values
        myNe, myTwoSz = pairvals[pairi];
        # repack overridden params
        myparams["Ne"] = myNe;
        myparams["TwoSz"] = myTwoSz;

        # occupation
        my_occs = get_occs_Ne_TwoSz(myNe, myTwoSz, num_skip = my_levels_skip);

        # get time=0 and time>0 wavefunctions etc
        _, _, _, _, _, my_charge, my_spin = get_overlaps(myparams, my_occs, plotwfs=True);

#### Rice-Mele dos (continuous DOS, not surface DOS)
elif(case in ["dos"]):

    # construct Time=0 single-body Hamiltonian as matrix
    if(myparams["sys_type"] in ["SIETS_RM"]):
        h1e_twhen, g2e_dummy = tddmrg.H_RM_builder(myparams, block=False);
        h1e_twhen, g2e_dummy = tddmrg.H_RM_polarizer(myparams, (h1e_twhen, g2e_dummy), block=False);
        # for siets_rm, perturber Vb removed by polarizer
    elif(myparams["sys_type"] in ["STT_RM"]):
        h1e_twhen, g2e_dummy = tddmrg.H_STTRM_builder(myparams, block=False);
        # for stt_rm, perturber Vconf added by polarizer, so we skip polarizer       

    # eigenstates
    del g2e_dummy
    h1e_twhen = h1e_twhen[::2,::2]; # <- make spinless !!
    vals_twhen, vecs_twhen = tdfci.solver(h1e_twhen);

    # printout
    centrals = np.arange(myparams["NL"],myparams["NL"]+myparams["NFM"]);
    RMdofs = 2;
    print("h1e_t0 = ");
    print(h1e_twhen[:8,:8]); 
    print(h1e_twhen[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1), RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]); 
    print(h1e_twhen[-8:,-8:]);

    # load data from json
    uval, vval, wval = myparams["u"], myparams["v"], myparams["w"];
        
    # from Rice-Mele parameters, construct matrices that act on mu dofs
    h00 = np.array([[uval, vval], [vval,-uval]]);
    h01 = np.array([[0.0, 0.0],[wval, 0.0]]);
    band_edges = wfm.bandedges_RiceMele(h00, h01);
    print("\n\nRice-Mele "+wfm.string_RiceMele(h00, h01, tex=False));
    ks = np.linspace(-np.pi, np.pi, myxvals);
    Es = wfm.dispersion_RiceMele(h00, h01, ks); # includes band index
    fixed_rho_points = np.array([0,10]);

    if(True): 
	####
	####
	# DOS from analytical dispersion
        dos_Es = (2/np.pi)/abs(np.array([np.gradient(Es[0],ks),
                                         np.gradient(Es[1],ks)])); # handles both bands at once
        dos_mins = np.min(dos_Es,axis=1);

	# plotting
        ncols_here = 2
        fig, (dispax,dosax) = plt.subplots(ncols=ncols_here, sharey=True);
        fig.set_size_inches((4.5/1.3)*ncols_here,5.5/1.3)
        for bandi in [0,1]: # iter over band index
            dispax.plot(ks/np.pi, Es[bandi], color=UniversalColors[bandi]);
            dosax.plot(dos_Es[bandi], Es[bandi], color=UniversalColors[bandi]);

	####
	####
	# DOS from manual hamiltonian diagonalization
        Es_direct = 1*vals_twhen;
        E_gap_tilde = np.min(Es_direct[len(Es_direct)//2:]) - np.max(Es_direct[:len(Es_direct)//2]);
        print("E_gap_tilde = {:.6f}".format(E_gap_tilde));

	# analyze
        direct_spacings = (Es_direct[1:] - Es_direct[:-1]);

	# plotting
        for bandi in [0,1]:
            band_direct = Es_direct[bandi*len(Es_direct)//2:(bandi+1)*len(Es_direct)//2];
            ks_direct = np.arccos(1/(2*myparams["v"]*myparams["w"])*((band_direct)**2 - myparams["u"]**2 - myparams["v"]**2 - myparams["w"]**2))
            dispax.scatter(ks_direct/np.pi, band_direct, marker=AccentsMarkers[1],color=UniversalAccents[1]);
            dos_direct = (2/np.pi)/abs(np.gradient(band_direct, ks_direct))
            dosax.scatter(dos_direct, band_direct, marker=AccentsMarkers[1],color=UniversalAccents[1]);
           
	# title
        if(myparams["sys_type"] in ["SIETS_RM"]): title_or_label = "$t_h =${:.2f}, ".format(params["th"]);
        elif(myparams["sys_type"] in ["STT_RM"]): title_or_label = "";
        title_or_label += wfm.string_RiceMele(h00, h01);
        fig.suptitle(title_or_label, fontsize = myfontsize);

        # format
        dispax.set_xlim(-1.0,1.0);
        dispax.set_xlabel("$ka/\pi$",fontsize=myfontsize);
        dosax.set_xlim(*fixed_rho_points);
        dosax.set_xlabel("$\\rho, \\rho_{min} = $"+str(dos_mins),fontsize=myfontsize);
        if(myparams["u"] > 0 or myparams["w"] != -1.0):
            dosax.annotate("  $N_{sites} =$"+"{:.0f}".format(len(h1e_twhen))+", $\\tilde{E}_{gap} = $"+"{:.4f}".format(E_gap_tilde),(0,0),color=UniversalAccents[-1]);
        for bedge in band_edges:
            dispax.axhline(bedge, color="grey", linestyle="dashed");
            dosax.axhline( bedge, color="grey", linestyle="dashed");
        dispax.set_ylabel("$E(k)$",fontsize=myfontsize);

	# show
        plt.tight_layout();
        plt.show();
    #### end if(True) block






elif(case in ["transport"]):

    def snapshot(state,time):
        '''
        '''
        n_yjs = np.full((len(state),),np.nan,dtype=float);
        for ni in range(len(n_yjs)):
            nop = 0*np.eye(len(state));
            nop[ni,ni] = 1;
            n_yjs[ni] = np.real(np.dot( np.conj(state), np.matmul(nop, state)));
        plt.plot(range(len(n_yjs)), n_yjs);
        plt.suptitle("Time = {:.2f}".format(time));
        plt.show();

    # construct Time=0 single-body Hamiltonian as matrix
    h1e_t0, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
    h1e_t0, g2e_dummy = tddmrg.H_RM_polarizer(params, (h1e_t0, g2e_dummy), block=False);
    h1e_t0 = h1e_t0[::2,::2]; # <- make spinless !!
    #h1e_t0[0,0] += (-100); # gd state = 0th site only, for testing
    vals_t0, vecs_t0 = tdfci.solver(h1e_t0);
    centrals = np.arange(params["NL"],params["NL"]+params["NFM"]);
    RMdofs = 2;
    print("h1e_t0 = ");
    print(h1e_t0[:8,:8]); 
    print(h1e_t0[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1),RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]); 
    print(h1e_t0[-8:,-8:]);
    
    # selection of M^th Time=0 eigenstate as the initial state
    initindex = int(sys.argv[3]);
    initstate = vecs_t0[initindex];
    #print("Filling {:.0f}th state of {:.0f} total molecular orbs".format(initindex, len(vecs_t0)));
    #print("Init energy state = {:.4f}".format(vals_t0[initindex]));
    print("Init energy spectrum:");
    filled_str = ["[ ]","[X]"];
    for statei in range(len(vecs_t0)): print("{:.6f} ".format(vals_t0[statei])+filled_str[statei==initindex]);
    del h1e_t0, vals_t0, vecs_t0;
    
    # Time > 0 eigenstates and eigenvalues
    h1e, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
    # no polarizer !
    h1e = h1e[::2,::2]; # <- make spinless !!
    vals, vecs = tdfci.solver(h1e);
    print("h1e = ");
    print(h1e[:8,:8]);
    print(h1e[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1),RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]);
    print(h1e[-8:,-8:]);  
    
    # set up observables over time propagation
    evolvedstate = np.copy(initstate);
    time_N = params["Nupdates"]*int(params["tupdate"]/params["time_step"]);
    print("time_N = ", time_N);
    times = np.arange(time_N)*params["time_step"];
    n0_op = 0.0*np.eye(len(h1e)); n0_op[0,0] = 1;
    n0_yjs = np.full((time_N,),np.nan);
    nSR_op = 0.0*np.eye(len(h1e)); 
    for muj in [RMdofs*params["NL"],RMdofs*params["NL"]+1]: nSR_op[muj,muj] = 1;
    nSR_yjs = np.full((time_N,),np.nan);
    js_pass = np.append(centrals, [centrals[-1]+1]); # one extra
    GL_op = 0.0*np.eye(len(h1e),dtype=complex);
    GR_op = 0.0*np.eye(len(h1e),dtype=complex);
    GLR = [GL_op, GR_op];
    for Gi in range(len(GLR)):
        muA, muB_prev = RMdofs*js_pass[Gi], RMdofs*js_pass[Gi]-1;
        print("G_ at site {:.0f}".format(muA));  
        GLR[Gi][muA, muB_prev] += complex(0,-1);
        GLR[Gi][muB_prev, muA] += complex(0, 1);  
    GL_yjs = np.full((time_N,),np.nan);
    GR_yjs = np.full((time_N,),np.nan);
    
    # time propagate
    for time_stepi in range(time_N): # iter over time steps
 
        # measure observables at Time = time_step*time_stepi
        n0_yjs_ret = np.dot( np.conj(evolvedstate), np.matmul(n0_op, evolvedstate));
        assert(abs(np.imag(n0_yjs_ret)) < 1e-10);
        n0_yjs[time_stepi] = np.real(n0_yjs_ret);
        nSR_yjs_ret = np.dot( np.conj(evolvedstate), np.matmul(nSR_op, evolvedstate));
        assert(abs(np.imag(nSR_yjs_ret)) < 1e-10);
        nSR_yjs[time_stepi] = np.real(nSR_yjs_ret);
        
        # conductance
        GL_yjs_ret =  np.dot( np.conj(evolvedstate), np.matmul(GL_op, evolvedstate)); 
        assert(abs(np.imag(GL_yjs_ret)) < 1e-10);
        GL_yjs[time_stepi] = np.pi*params["th"]/params["Vb"]*np.real(GL_yjs_ret); 
        GR_yjs_ret =  np.dot( np.conj(evolvedstate), np.matmul(GR_op, evolvedstate)); 
        assert(abs(np.imag(GR_yjs_ret)) < 1e-10);
        GR_yjs[time_stepi] = np.pi*params["th"]/params["Vb"]*np.real(GR_yjs_ret);         
        
        if(time_stepi%100==0):snapshot(evolvedstate, time_stepi*params["time_step"]);
        evolvedstate = tdfci.propagator(evolvedstate, params["time_step"], vals, vecs);
        
    # plot
    plt.plot(times, n0_yjs, label = "$n_0$");
    plt.plot(times, nSR_yjs, label = "$n_{SR}$");
    plt.plot(times,GL_yjs, label = "$G_L$");
    plt.plot(times,GR_yjs, label = "$G_R$");
    
    # format
    plt.suptitle("$N_{eff} =$"+"{:.0f}, $t_h =${:.2f}, $V_g =${:.2f}, $u =${:.2f}, $v =${:.2f}, $w =${:.2f}".format(2*(initindex+1), params["th"], params["Vg"], params["u"], params["v"], params["w"]));
    
    # show
    plt.legend();
    plt.tight_layout();
    plt.show();
	
else: raise Exception("case = "+case+" not supported");


