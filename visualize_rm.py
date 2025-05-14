'''
'''

from transport import wfm, tdfci, tddmrg

import numpy as np
import matplotlib.pyplot as plt

import sys
import json

def lorentzian(xarr, x0, Gamma):
    return (1/np.pi)*(0.5*Gamma)/((xarr-x0)**2 + (0.5*Gamma)**2);

def square(xarr, x0, xlim,height):
    zs = 0*xarr;
    zs[abs(xarr-x0)<xlim] += height+zs[abs(xarr-x0)<xlim];
    #plt.plot(zs)
    #plt.show()
    return zs

def energies_to_dos(Evals, discretes, Gamma, height):
    dosvals = np.zeros_like(Evals);
    for E in discretes:
        dosvals += square(Evals, E, Gamma, height);

    return dosvals;
    
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

def tb_hamiltonian(the_params, initialize, verbose=0):
    '''
    '''

    # construct Time=0 single-body Hamiltonian as matrix
    if(params["sys_type"] in ["SIETS_RM"]):
        h1e_t0, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
    elif(params["sys_type"] in ["STT_RM"]):
        h1e_t0, g2e_dummy = tddmrg.H_STTRM_builder(params, block=False);

    # if Time = 0, add polarizer
    if(initialize):
        if(params["sys_type"] in ["SIETS_RM"]):
            h1e_t0, g2e_dummy = tddmrg.H_RM_polarizer(params, (h1e_t0, g2e_dummy), block=False);
        elif(params["sys_type"] in ["STT_RM"]):
            h1e_t0, g2e_dummy = tddmrg.H_STTRM_polarizer(params, (h1e_t0, g2e_dummy), block=False);

    # eigenstates
    del g2e_dummy
    h = h1e_t0[::2,::2]; # <- make spinless !!
    Evals, Evecs = tdfci.solver(h);
    kvals = np.arccos(1/(2*params["v"]*params["w"])*((Evals)**2 - params["u"]**2 - params["v"]**2 - params["w"]**2));
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
    kmvals, Emvals, Emvecs, h1e_t0 = tb_hamiltonian(params, initialize=True, verbose=1); 
    the_occs_final = np.zeros_like(Emvals, dtype = the_occs.dtype);
    the_occs_final[:len(the_occs)] = the_occs[:];
    print(the_occs,"-->\n", the_occs_final);
    the_occs = 1*the_occs_final; del the_occs_final;
    
    ms_to_probe = [max(0,m_HOMO-2), min(m_HOMO+1,len(Emvecs))];
    ms_to_probe = range(ms_to_probe[0], ms_to_probe[1]);
    for m in ms_to_probe:
        the_axes[0,0].plot(Emvecs[m], label="$m =${:.0f}, $k_m =${:.4f}".format(m, kmvals[m]));
    diag_norm_t0 = np.max(abs(np.diag(h1e_t0)))
    if(diag_norm_t0 < 1e-10): diag_norm_t0 = 1;
    if(plothams):
        the_axes[0,0].plot(np.diag(h1e_t0)/diag_norm_t0,color="black")#,label="$\langle j | V |j \\rangle$"); # onsite
        the_axes[0,0].plot(np.diag(h1e_t0,k=1),color="teal",marker="o")#,label="$\langle j | V |j+1 \\rangle$"); # hopping
    else: the_axes[0,0].set_ylim(-0.4,0.4);
    the_axes[0,0].legend(loc="lower right");
    the_axes[0,0].set_xlabel("$j$");
    the_axes[0,0].set_ylabel("$\langle j | k_m \\rangle$");
    the_axes[0,0].set_title("$u =${:.2f}, $v =${:.2f}, $w =${:.2f}".format(params["u"], params["v"], params["w"]));

    # t>0 eigenstates (|k_n> states)
    knvals, Envals, Envecs, h1e = tb_hamiltonian(params, initialize=False, verbose=0);
    for n in range(4):
        the_axes[1,0].plot(Envecs[n], label="$n =${:.0f}, $k_n=${:.4f}".format(n,knvals[n]));
    diag_norm = np.max(abs(np.diag(h1e)));
    if(diag_norm < 1e-10): diag_norm = 1;
    if(plothams):
        the_axes[1,0].plot(np.diag(h1e)/diag_norm,color="black")#,label="$\langle j | V |j \\rangle$"); # onsite
        the_axes[1,0].plot(np.diag(h1e,k=1),color="teal",marker="o")#,label="$\langle j | V |j+1 \\rangle$"); # hopping
    else: the_axes[1,0].set_ylim(-0.4,0.4);
    the_axes[1,0].legend(loc="lower right");
    the_axes[1,0].set_xlabel("$j$");
    the_axes[1,0].set_ylabel("$\langle j | k_n \\rangle$");

    ####  3) overlap of the *occupied* t<0 states with t>0 k states
    the_pdfs = np.zeros((len(knvals),),dtype=float);
    for knvali in range(len(knvals)):
        # iter over *occupied* |k_m> states
        for kmvali in range(len(kmvals)):
            overlap = np.dot( np.conj(Emvecs[kmvali]), Envecs[knvali]);
            the_pdfs[knvali] += the_occs[kmvali]*np.real(np.conj(overlap)*overlap);
    band_divider = len(knvals)//2;
    the_axes[0,1].plot(knvals[:band_divider], the_pdfs[:band_divider], color="black");
    the_axes[0,1].plot(knvals[band_divider:], the_pdfs[band_divider:], color="red");
    the_axes[0,1].set_xlabel("$k_n$");
    the_axes[0,1].set_ylabel("$ \sum_{k_m}^{occ} |\langle k_m | k_n \\rangle |^2$");

    #### overlap plotted again, but vs E_n rather than k_n
    the_axes[1,1].plot(Envals[:band_divider], the_pdfs[:band_divider], color="black");
    the_axes[1,1].plot(Envals[band_divider:], the_pdfs[band_divider:], color="red");
    
    # visualize the k-state wavefunctions
    the_fig.suptitle("$N_e =${:.0f}".format(np.sum(the_occs))+", $N_{conf} =$"+"{:.0f}".format(params["Nconf"])
            +", $m_{HOMO} =$"+"{:.0f}".format(m_HOMO));
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
    return knvals, the_pdfs, charge_pdf, spin_pdf;


if(__name__ == "__main__"):
    
    # top level
    verbose = 2; assert verbose in [1,2,3];
    np.set_printoptions(precision = 6, suppress = True);
    myxvals = 999;
    case = sys.argv[1];
    json_name = sys.argv[2];
    try:
        try:
            params = json.load(open(json_name+".txt"));
        except:
            params = json.load(open(json_name));
            json_name = json_name[:-4];
        print(">>> Params = ",params);
    except:
        raise Exception(json_name+" cannot be found");

    # fig standardizing
    from transport.wfm import UniversalColors, UniversalAccents, ColorsMarkers, AccentsMarkers, UniversalMarkevery, UniversalPanels;
    mylinewidth = 1.0;
    myfontsize = 14;

#### Rice Mele Time > 0 occupation distribution
if(case in ["distro"]):

    # plot either individual wfs or total charge density
    plot_wfs = True;

    # iter over Ne, excited_state values
    pairvals = np.array([(1,0),(2,0),(5,0), (10,0)]);
    # ^ Nconf=20 sites (10 blocks) fixed,
    is_spinpol = True;

    for pairi in range(len(pairvals)):

        # unpack overriden values
        myNe, my_levels_skip = pairvals[pairi];
        if(is_spinpol): myTwoSz = 1*myNe;
        else: myTwoSz=0; assert(myNe%2==0);
        # repack overridden params
        params["Ne"] = myNe;
        params["TwoSz"] = myTwoSz;

        # occupation
        my_occs = get_occs_Ne_TwoSz(myNe, myTwoSz, num_skip = my_levels_skip);

        # get time=0 and time>0 wavefunctions etc
        _, _, my_charge, my_spin = get_overlaps(params, my_occs, plotwfs=plot_wfs);
        if(not plot_wfs): # <-- this makes a time=0 charge density plot
                          # which can validate DMRG initialization
            pairfig, pairax = plt.subplots();
            pairax.fill_between(np.arange(len(my_charge)),my_charge, color="cornflowerblue");
            pairax.plot(np.arange(len(my_spin)),my_spin, color="darkblue",marker="o");
            pairax.set_title("$N_e =${:.0f}, $2S_z=${:.0f}".format(myNe,myTwoSz)+", $N_{conf} =$"+"{:.0f}".format(myNconf));
            for tick in [-1.0,-0.5,0.0,0.5,1.0]: pairax.axhline(tick,linestyle=(0,(5,5)),color="gray");
            plt.show();


#### Rice-Mele dos (continuous DOS, not surface DOS)
elif(case in ["dos"]):

    # construct Time=0 single-body Hamiltonian as matrix
    if(params["sys_type"] in ["SIETS_RM"]):
        h1e_twhen, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
        h1e_twhen, g2e_dummy = tddmrg.H_RM_polarizer(params, (h1e_twhen, g2e_dummy), block=False);
        # for siets_rm, perturber Vb removed by polarizer
    elif(params["sys_type"] in ["STT_RM"]):
        h1e_twhen, g2e_dummy = tddmrg.H_STTRM_builder(params, block=False);
        # for stt_rm, perturber Vconf added by polarizer, so we skip polarizer       

    # eigenstates
    del g2e_dummy
    h1e_twhen = h1e_twhen[::2,::2]; # <- make spinless !!
    vals_twhen, vecs_twhen = tdfci.solver(h1e_twhen);

    # printout
    centrals = np.arange(params["NL"],params["NL"]+params["NFM"]);
    RMdofs = 2;
    print("h1e_t0 = ");
    print(h1e_twhen[:8,:8]); 
    print(h1e_twhen[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1),RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]); 
    print(h1e_twhen[-8:,-8:]);

    if(True):
        # load data from json
        uval, vval, wval = params["u"], params["v"], params["w"];
        
        # from Rice-Mele parameters, construct matrices that act on mu dofs
        h00 = np.array([[uval, vval], [vval,-uval]]);
        h01 = np.array([[0.0, 0.0],[wval, 0.0]]);
        band_edges = wfm.bandedges_RiceMele(h00, h01);
        print("\n\nRice-Mele "+wfm.string_RiceMele(h00, h01, tex=False));
        ks = np.linspace(-np.pi, np.pi, myxvals);
        Es = wfm.dispersion_RiceMele(h00, h01, ks); # includes band index
        fixed_rho_points = np.array([0,10]);
        
	####
	####
	# DOS from analytical dispersion
        dos_Es = (2/np.pi)/abs(np.array([np.gradient(Es[0],ks),np.gradient(Es[0],ks)])); # handles both bands at once
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
        myGamma = np.mean(direct_spacings);
        myHeight = 1;
        print("Gamma = mean(E_spacing) = {:.6f}, height = {:.6f}".format(myGamma, myHeight));
        use_dos_direct = False;

	# plotting
        for bandi in [0,1]:
            band_direct = Es_direct[bandi*len(Es_direct)//2:(bandi+1)*len(Es_direct)//2];
            dos_direct = energies_to_dos(Es[bandi], band_direct, myGamma, myHeight);
            if(use_dos_direct): dosax.plot(dos_direct, Es[bandi]);
            else:
                for E in band_direct:
                    dosax.plot([0.5*np.mean(fixed_rho_points),1.5*np.mean(fixed_rho_points)],[E,E],color=UniversalAccents[-1]);

	# title
        if(params["sys_type"] in ["SIETS_RM"]): title_or_label = "$t_h =${:.2f}, ".format(params["th"]);
        elif(params["sys_type"] in ["STT_RM"]): title_or_label = "";
        title_or_label += wfm.string_RiceMele(h00, h01);
        fig.suptitle(title_or_label, fontsize = myfontsize);

        # format
        dispax.set_xlim(-1.0,1.0);
        dispax.set_xlabel("$ka/\pi$",fontsize=myfontsize);
        if(not use_dos_direct): dosax.set_xlim(*fixed_rho_points);
        else: dosax.set_xlim(*fixed_rho_points);
        dosax.set_xlabel("$\\rho, \\rho_{min} = $"+str(dos_mins),fontsize=myfontsize);
        if(params["u"] > 0 or params["w"] != -1.0):
            dosax.annotate("  $N_{sites} =$"+"{:.0f}".format(len(h1e_t0))+", $\\tilde{E}_{gap} = $"+"{:.4f}".format(E_gap_tilde),(0,0),color=UniversalAccents[-1]);
        for bedge in band_edges:
            dispax.axhline(bedge, color="grey", linestyle="dashed");
            dosax.axhline( bedge, color="grey", linestyle="dashed");
        dispax.set_ylabel("$E(k)$",fontsize=myfontsize);

	# show
        plt.tight_layout();
        plt.show();
    #### end if(True) block






elif(case in ["transport"]):

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


