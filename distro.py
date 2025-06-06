import numpy as np
import matplotlib.pyplot as plt

import sys
import json

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

def get_overlaps(the_params, the_occs, plot=False):
    '''
    '''
    if(not isinstance(the_occs, np.ndarray)): raise TypeError;
    if(the_occs.dtype != int): raise TypeError;
    n_subplots_here = 3;
    the_fig, the_axes = plt.subplots(n_subplots_here);
    the_fig.set_size_inches(6,3.5*n_subplots_here);
    
    # get params
    the_Nconf, the_NFM, the_NR=the_params["Nconf"], the_params["NFM"], the_params["NR"];
    the_NL = the_Nconf + 5; 
    the_Nsites = the_NL+the_NFM+the_NR;
    the_tl, the_Vconf = the_params["tl"], the_params["Vconf"];
    if("Vdelta" in the_params.keys()): the_Vdelta = the_params["Vdelta"];
    else: the_Vdelta = 0;
    if("tp" in the_params.keys()): the_tp = the_params["tp"];
    else: the_tp = 1*the_tl;
    assert(the_Vdelta==0);
    
    #### 1) eigenstates of the t<0 system
    nloc = 1; # consider molecular orbitals with occupancy = 0,1,2
    the_Nspinorbs = nloc*the_Nsites;
    h1e_t0= np.zeros((the_Nspinorbs, the_Nspinorbs),dtype=float);

    # classify site indices (spin not included)
    conf_sites = np.arange(the_Nconf);
    tp_symmetry = 1;
    if("tp_symmetry" in the_params.keys()): tp_symmetry = the_params["tp_symmetry"];
    if tp_symmetry: # modifies left and right lead symmetrically
        assert(the_Nsites - the_Nconf > the_Nconf);
        anticonf_sites = np.arange(the_Nsites-the_Nconf, the_Nsites);
        nonconf_sites = np.arange(the_Nconf, the_Nsites - the_Nconf);
    else: # modifies left lead only
        anticonf_sites = np.array([]);
        nonconf_sites = np.arange(the_Nconf, the_Nsites);
    all_sites = np.arange(the_Nsites);

    # j <-> j+1 hopping for fermions
    for j in conf_sites:
        h1e_t0[nloc*j+0,nloc*(j+1)+0] += -the_tp; # <-- different hopping !!
        h1e_t0[nloc*(j+1)+0,nloc*j+0] += -the_tp;
    for j in nonconf_sites[:-1]:
        h1e_t0[nloc*j+0,nloc*(j+1)+0] += -the_tl; # spinless
        h1e_t0[nloc*(j+1)+0,nloc*j+0] += -the_tl;
    for j in (anticonf_sites-1): # this iteration is empty if tp_symmetry == False
        h1e_t0[nloc*j+0,nloc*(j+1)+0] += -the_tp; # <-- different hopping !!
        h1e_t0[nloc*(j+1)+0,nloc*j+0] += -the_tp;

    # confinement
    for j in conf_sites:
        h1e_t0[nloc*j+0,nloc*j+0] += -the_Vconf; # spinless!

    # t<0 eigenstates (|k_m> states)
    m_HOMO = len(the_occs) - 1; # m value of the highest occupied k state
    vals_t0, vecs_t0 = np.linalg.eigh(h1e_t0);
    vecs_t0 = vecs_t0.T;
    the_occs_final = np.zeros_like(vals_t0, dtype = the_occs.dtype);
    the_occs_final[:len(the_occs)] = the_occs[:];
    print(the_occs,"-->\n", the_occs_final);
    the_occs = 1*the_occs_final; del the_occs_final;
    
    ms_to_probe = [max(0,m_HOMO-2), min(m_HOMO+1,len(vecs_t0))];
    ms_to_probe = range(ms_to_probe[0], ms_to_probe[1]);
    for m in ms_to_probe:
        if(m < the_Nconf): k_plotted_level = np.pi*(m+1)/the_Nconf;
        else: k_plotted_level = np.nan;
        the_axes[0].plot(vecs_t0[m], label="$m =${:.0f}, $k_m =${:.4f}".format(m+1, k_plotted_level));
    diag_norm_t0 = np.max(abs(np.diag(h1e_t0)))
    if(diag_norm_t0 < 1e-10): diag_norm_t0 = 1;
    the_axes[0].plot(np.diag(h1e_t0)/diag_norm_t0,color="black")#,label="$\langle j | V |j \\rangle$"); # onsite
    the_axes[0].plot(np.diag(h1e_t0,k=1),color="teal",marker="o")#,label="$\langle j | V |j+1 \\rangle$"); # hopping
    the_axes[0].legend(loc="lower right");
    the_axes[0].set_xlabel("$j$");
    the_axes[0].set_ylabel("$\langle j | k_m \\rangle$");

    #### 2) eigenstates of the t>0 system
    h1e = np.zeros((the_Nspinorbs, the_Nspinorbs),dtype=float);

    # j <-> j+1 hopping for fermions
    for j in conf_sites:
        h1e[nloc*j+0,nloc*(j+1)+0] += -the_tp; # <-- different hopping !!
        h1e[nloc*(j+1)+0,nloc*j+0] += -the_tp;
    for j in nonconf_sites[:-1]:
        h1e[nloc*j+0,nloc*(j+1)+0] += -the_tl; # spinless
        h1e[nloc*(j+1)+0,nloc*j+0] += -the_tl;
    for j in (anticonf_sites-1):
        h1e[nloc*j+0,nloc*(j+1)+0] += -the_tp; # <-- different hopping !!
        h1e[nloc*(j+1)+0,nloc*j+0] += -the_tp;
        
    # electric field
    for j in all_sites:
        h1e[nloc*j+0,nloc*j+0] += -(the_Vdelta)*j;

    # t>0 eigenstates (|k_n> states)
    vals, vecs = np.linalg.eigh(h1e);
    vecs = vecs.T;
    for n in range(4):
        the_axes[1].plot(vecs[n], label="$n =${:.0f}, $k_n=${:.4f}".format(n+1,np.pi*(n+1)/the_Nsites));
    diag_norm = np.max(abs(np.diag(h1e)))
    if(diag_norm < 1e-10): diag_norm = 1;
    the_axes[1].plot(np.diag(h1e)/diag_norm,color="black")#,label="$\langle j | V |j \\rangle$"); # onsite
    the_axes[1].plot(np.diag(h1e,k=1),color="teal",marker="o")#,label="$\langle j | V |j+1 \\rangle$"); # hopping
    the_axes[1].legend(loc="lower right");
    the_axes[1].set_xlabel("$j$");
    the_axes[1].set_ylabel("$\langle j | k_n \\rangle$");

    ####  3) overlap of the *occupied* t<0 states with t>0 k states
    knvals = np.pi/the_Nsites *np.arange(1,1+len(vals)); # possible |k_n>
    the_pdfs = np.zeros((len(knvals),),dtype=float);
    for knvali in range(len(knvals)):
        # iter over *occupied* |k_m> states
        for kmvali in range(len(vals_t0)):
            overlap = np.dot( np.conj(vecs_t0[kmvali]), vecs[knvali]);
            the_pdfs[knvali] += the_occs[kmvali]*np.real(np.conj(overlap)*overlap);
    the_axes[2].plot(knvals, the_pdfs, color="black");
    if(np.sum(the_occs)==1): the_axes[2].axvline(np.pi/the_Nconf, linestyle="dashed", color="black");
    the_axes[2].set_xlabel("$k_n$");
    the_axes[2].set_ylabel("$ \sum_{k_m}^{occ} |\langle k_m | k_n \\rangle |^2$");

    # visualize the k-state wavefunctions
    the_axes[0].set_title("$N_e =${:.0f}".format(np.sum(the_occs))+", $N_{conf} =$"+"{:.0f}".format(the_Nconf)
            +", $\Delta V =${:.3f}, $t_p =${:.1f}".format(the_Vdelta, the_tp)
            +", $m_{HOMO} =$"+"{:.0f}".format(m_HOMO));
    if(plot): plt.tight_layout(), plt.show();
    else: plt.close();
    
    # real-space PDF
    real_pdf = np.zeros_like(vecs_t0[0], dtype=float);
    # iter over *occupied* |k_m> states
    for kmvali in range(len(vals_t0)):
        real_pdf += the_occs[kmvali]*np.conj(vecs_t0[kmvali])*vecs_t0[kmvali];
        
    # real-space spin PDF
    spin_pdf = np.zeros_like(vecs_t0[0], dtype=float);
    # iter over *occupied* |k_m> states
    for kmvali in range(len(vals_t0)):
        spin_pdf += the_occs[kmvali]*(2-the_occs[kmvali])*np.conj(vecs_t0[kmvali])*vecs_t0[kmvali];
        
    # return
    return knvals, the_pdfs, real_pdf, spin_pdf;

# top level
norm_Ne = False;
myfontsize = 18;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
plt.rcParams.update({"font.family": "serif"});
#plt.rcParams.update({"text.usetex": True});

# physical params
datafile = sys.argv[1];
params = json.load(open(datafile+".txt"));
# override ones that will be changed later
to_override = ["Ne", "TwoSz", "Nconf"];
for key in to_override: params[key] = np.nan;

# iter over Ne, TwoSz, Nconf, excited_state values
pairvals = np.array([(1,1,2,0),(1,1,5,0),(1,1,10,0),(1,1,20,0)]); # Ne=1 fixed, ground vs excited states


#pairvals = np.array([(1,1,20,0),(5,5,20,0), (5,5,40,0),(5,5,60,0)]); # mix of interesting cases
#pairvals = np.array([(1,1,20,0),(5,5,20,0)]); # mix of interesting cases

#pairvals = np.array([(1,1,20,0),(2,2,20,0),(5,5,20,0), (10,10,20,0)]); # Nconf=20 fixed, spin polarized
#pairvals = np.array([(1,1,40,0),(2,2,40,0),(5,5,40,0), (10,10,40,0)]); # Nconf=40 fixed, spin polarized

#pairvals = np.array([(5,5,5,0),(10,10,10,0),(20,20,20,0)]); # half filling, spin pol

# plot either individual wfs or real-space PDFs
plot_wfs = True;
for pairi in range(len(pairvals)):
    myNe, myTwoSz, myNconf, my_levels_skip = pairvals[pairi];
    my_occs = get_occs_Ne_TwoSz(myNe, myTwoSz, num_skip = my_levels_skip);

    # unpack geometric params
    myNsites = myNconf+5+params["NFM"]+params["NR"];
    # repack overridden params
    params["Ne"] = myNe;
    params["TwoSz"] = myTwoSz;
    params["Nconf"] = myNconf;

    # get pdf values
    _, _, my_real, my_spin = get_overlaps(params, my_occs, plot=plot_wfs);
    if(not plot_wfs):
        pairfig, pairax = plt.subplots();
        pairax.fill_between(np.arange(len(my_real)),my_real, color="cornflowerblue");
        pairax.plot(np.arange(len(my_spin)),my_spin, color="darkblue",marker="o");
        pairax.set_title("$N_e =${:.0f}, $2S_z=${:.0f}".format(myNe,myTwoSz)+", $N_{conf} =$"+"{:.0f}".format(myNconf));
        for tick in [-1.0,-0.5,0.0,0.5,1.0]: pairax.axhline(tick,linestyle=(0,(5,5)),color="gray");
        plt.show();

# plot the PDFs
myfig, myax = plt.subplots();
for pairi in range(len(pairvals)):
    myNe, myTwoSz, myNconf, my_levels_skip = pairvals[pairi];
    my_occs = get_occs_Ne_TwoSz(myNe, myTwoSz, num_skip = my_levels_skip);
    
    # unpack geometric params
    myNsites = myNconf+5+params["NFM"]+params["NR"];
    params["Ne"] = myNe;
    params["TwoSz"] = myTwoSz;
    params["Nconf"] = myNconf;
    
    # get pdf values
    mykvals, mypdfs, _, _ = get_overlaps(params, my_occs, plot=False); 
    nvals_mykvals =  (mykvals*myNsites/np.pi).astype(int)
    if(norm_Ne): mypdfs = mypdfs/myNe;
    myax.plot(nvals_mykvals, mypdfs, color=mycolors[pairi], label = "$N_e =${:.0f}".format(myNe)+", $N_{conf} =$"+"{:.0f}".format(myNconf));

# format
myax.axvline(np.pi/20, color="black", label = "$k_n^{target}$");
myax.set_xlabel("$n$ ($N_{sites} =$"+"{:.0f})".format(myNsites),fontsize=myfontsize);
#myax.set_xlim(mykvals[0], mykvals[-1]);
pdf_ylabel = "$ \sum_{k_m}^{occ} |\langle k_m | k_n \\rangle |^2$";
if(norm_Ne): pdf_ylabel = "$\\frac{1}{N_e} $"+pdf_ylabel;
myax.set_ylabel(pdf_ylabel,fontsize=myfontsize);
#myax.set_ylim(0.0,0.5);
myax.legend(fontsize = myfontsize);

# show
plt.tight_layout();
plt.show();
