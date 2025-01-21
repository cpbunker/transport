import numpy as np
import matplotlib.pyplot as plt

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

def get_overlaps(the_Nconf, the_Nsites, the_tl, the_Vconf, the_Vdelta, the_occs, plot=False):
    '''
    '''
    if(not isinstance(the_occs, np.ndarray)): raise TypeError;
    if(the_occs.dtype != int): raise TypeError;
    the_fig, the_axes = plt.subplots(3);
    
    #### 1) eigenstates of the t<0 system
    nloc = 1; # consider molecular orbitals with occupancy = 0,1,2
    the_Nspinorbs = nloc*the_Nsites;
    h1e_t0= np.zeros((the_Nspinorbs, the_Nspinorbs),dtype=float);

    # classify site indices (spin not included)
    conf_sites = np.arange(the_Nconf);
    all_sites = np.arange(the_Nsites);

    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        h1e_t0[nloc*j+0,nloc*(j+1)+0] += -the_tl; # spinless
        h1e_t0[nloc*(j+1)+0,nloc*j+0] += -the_tl;

    # confinement
    for j in conf_sites:
        h1e_t0[nloc*j+0,nloc*j+0] += -the_Vconf; # spinless!

    # t<0 eigenstates (|k_m> states)
    vals_t0, vecs_t0 = np.linalg.eigh(h1e_t0);
    vecs_t0 = vecs_t0.T;
    the_occs_final = np.zeros_like(vals_t0, dtype = the_occs.dtype);
    the_occs_final[:len(the_occs)] = the_occs[:];
    print(the_occs,"-->\n", the_occs_final);
    the_occs = 1*the_occs_final; del the_occs_final;
    
    for vali in range(4):
        the_axes[0].plot(vecs_t0[vali], label="$k_m =${:.4f}".format(np.pi*(vali+1)/the_Nconf));
    the_axes[0].plot(np.diag(h1e_t0)/np.max(abs(np.diag(h1e_t0))),color="black",label="$V_j$");
    the_axes[0].legend(loc="lower right");
    the_axes[0].set_xlabel("$j$");
    the_axes[0].set_ylabel("$\langle j | k_m \\rangle$");

    #### 2) eigenstates of the t>0 system
    h1e = np.zeros((the_Nspinorbs, the_Nspinorbs),dtype=float);

    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        h1e[nloc*j+0,nloc*(j+1)+0] += -the_tl; # spinless
        h1e[nloc*(j+1)+0,nloc*j+0] += -the_tl;
        
    # electric field
    for j in all_sites:
        h1e[nloc*j+0,nloc*j+0] += -(the_Vdelta)*j;

    # t>0 eigenstates (|k_n> states)
    vals, vecs = np.linalg.eigh(h1e);
    vecs = vecs.T;
    for vali in range(4):
        the_axes[1].plot(vecs[vali], label="$k_n=${:.4f}".format(np.pi*(vali+1)/the_Nsites));
    the_axes[1].plot(np.diag(h1e)/np.max(abs(np.diag(h1e))),color="black",label="$V_j$");
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
    the_axes[0].set_title("$N_e =${:.0f}".format(np.sum(the_occs))+", $N_{conf} =$"+"{:.0f}".format(the_Nconf));
    if(plot): plt.show();
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

# physical params
mytl = 1.0;
myVconf = 30.0;
myVdelta = -0.01;

# iter over Ne, TwoSz, Nconf values
#pairvals = np.array([(1,1,5,0),(1,1,10,0), (1,1,20,0),(1,1,40,0)]); # Ne=1 fixed
#pairvals = np.array([(1,1,20,0),(1,1,20,2),(1,1,20,4)]); # Ne=1 fixed, ground vs excited states
#pairvals = np.array([(1,1,20,0),(5,5,20,0), (5,5,40,0),(5,5,60,0)]); # mix of interesting cases
#pairvals = np.array([(1,1,20,0),(5,5,20,0)]); # mix of interesting cases
pairvals = np.array([(1,1,20,0),(2,2,20,0),(5,5,20,0), (10,10,20,0)]); # Nconf=20 fixed, spin polarized
#pairvals = np.array([(2,0,20,0),(10,0,20,0)]); # Nconf=20 fixed not spin polarized
#pairvals = np.array([(2,0,20,0),(4,0,20,0),(10,0,20,0), (20,0,20,0)]); # Nconf=20 fixed not spin polarized
#pairvals = np.array([(2,2,20,0),(4,2,20,0),(10,2,20,0), (20,2,20,0)]); # Nconf=20 fixed slightly spin polarized

# plot either individual wfs or real-space PDFs
plot_wfs = True;
for pairi in range(len(pairvals)):
    myNe, myTwoSz, myNconf, my_levels_skip = pairvals[pairi];
    my_occs = get_occs_Ne_TwoSz(myNe, myTwoSz, num_skip = my_levels_skip);

    # geometric params
    myNL = myNconf+5;
    myNFM = 2;
    myNR = 30;
    myNsites = myNL+myNFM+myNR;

    # get pdf values
    _, _, my_real, my_spin = get_overlaps(myNconf, myNsites, mytl, myVconf, myVdelta, my_occs, plot=plot_wfs);
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

    # geometric params
    myNL = myNconf+5;
    myNFM = 2;
    myNR = 30;
    myNsites = myNL+myNFM+myNR;

    # get pdf values
    mykvals, mypdfs, _, _ = get_overlaps(myNconf, myNsites, mytl, myVconf, myVdelta, my_occs, plot=False);  
    if(norm_Ne): mypdfs = mypdfs/myNe;
    myax.plot(mykvals, mypdfs, color=mycolors[pairi], label = "$N_e =${:.0f}, $\Delta V=${:.2f}".format(myNe,myVdelta)+", $N_{conf} =$"+"{:.0f}".format(myNconf));
    #myax.axvline(np.pi/myNconf, color=mycolors[pairi], linestyle="dashed", label="$k^0$");

# format
myax.axvline(np.pi/20, color="black", label = "$k_n^{target}$");
myax.set_xlabel("$k_n$",fontsize=myfontsize);
myax.set_xlim(mykvals[0], mykvals[-1]);
pdf_ylabel = "$ \sum_{k_m}^{occ} |\langle k_m | k_n \\rangle |^2$";
if(norm_Ne): pdf_ylabel = "$\\frac{1}{N_e} $"+pdf_ylabel;
myax.set_ylabel(pdf_ylabel,fontsize=myfontsize);
#myax.set_ylim(0.0,0.5);
myax.legend(fontsize = myfontsize);

# show
plt.tight_layout();
plt.show();
