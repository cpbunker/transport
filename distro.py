import numpy as np
import matplotlib.pyplot as plt

def get_overlaps(the_Nconf, the_Nsites, the_Ne, the_tl, the_Vconf, plot=False):
    '''
    '''
    the_fig, the_axes = plt.subplots(3);
    
    #### 1) eigenstates of the t<0 system
    nloc = 1; # spinless !
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

    # low energy eigenstates
    vals_t0, vecs_t0 = np.linalg.eigh(h1e_t0);
    vecs_t0 = vecs_t0.T;
    
    for vali in range(4):
        the_axes[0].plot(vecs_t0[vali], label="$k_m =${:.4f}".format(np.pi*(vali+1)/the_Nconf));
    the_axes[0].legend(loc="lower right");
    the_axes[0].set_xlabel("$j$");
    the_axes[0].set_ylabel("$\langle j | k_m \\rangle$");

    #### 2) eigenstates of the t>0 system
    h1e = np.zeros((the_Nspinorbs, the_Nspinorbs),dtype=float);

    # j <-> j+1 hopping for fermions
    for j in all_sites[:-1]:
        h1e[nloc*j+0,nloc*(j+1)+0] += -the_tl; # spinless
        h1e[nloc*(j+1)+0,nloc*j+0] += -the_tl;

    # low energy eigenstates
    vals, vecs = np.linalg.eigh(h1e);
    vecs = vecs.T;
    for vali in range(4):
        the_axes[1].plot(vecs[vali], label="$k_n=${:.4f}".format(np.pi*(vali+1)/the_Nsites));
    the_axes[1].legend(loc="lower right");
    the_axes[1].set_xlabel("$j$");
    the_axes[1].set_ylabel("$\langle j | k_n \\rangle$");

    ####  3) overlap of the filled t<0 states with t>0 k states
    the_kvals = np.pi/the_Nsites *np.arange(1,1+len(vals));
    the_pdfs = np.zeros((len(the_kvals),),dtype=float);
    for kvali in range(len(the_kvals)):
        # iter over **filled** initial states
        for initi in range(the_Ne):
            overlap = np.dot( np.conj(vecs_t0[initi]), vecs[kvali]);
            the_pdfs[kvali] += np.real(np.conj(overlap)*overlap);
    the_axes[2].plot(the_kvals, the_pdfs, color="black");
    if(the_Ne==1): the_axes[2].axvline(np.pi/the_Nconf, linestyle="dashed", color="black");
    the_axes[2].set_xlabel("$k_n$");
    the_axes[2].set_ylabel("$ \sum_{k_m} |\langle k_m | k_n \\rangle |^2$");

    # visualize the k-state wavefunctions
    the_axes[0].set_title("$N_e =${:.0f}".format(the_Ne)+", $N_{conf} =$"+"{:.0f}".format(the_Nconf));
    if(plot): plt.show();
    else: plt.close();

    # return
    return the_kvals, the_pdfs;

# top level
norm_Ne = False;
myfontsize = 18;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];

# physical params
mytl = 1.0;
myVconf = 30.0;

# iter over Ne, Nconf values
pairvals = np.array([(2,2),(5,5), (10,10)]); # always half filling
pairvals = np.array([(1,10),(2,10),(5,10), (10,10)]); # Nconf=10 fixed
pairvals = np.array([(1,20),(2,20),(5,20), (10,20)]); # Nconf=20 fixed (low energy)
#pairvals = np.array([(1,5),(1,10), (1,20),(1,40)]); # Ne=1 fixed
for pairi in range(len(pairvals)):
    myNe, myNconf = pairvals[pairi];

    # geometric params
    myNL = myNconf+5;
    myNFM = 2;
    myNR = 30;
    myNsites = myNL+myNFM+myNR;

    # get pdf values
    get_overlaps(myNconf, myNsites, myNe, mytl, myVconf, plot=True);

# plot the PDFs
myfig, myax = plt.subplots();
for pairi in range(len(pairvals)):
    myNe, myNconf = pairvals[pairi];

    # geometric params
    myNL = myNconf+5;
    myNFM = 2;
    myNR = 30;
    myNsites = myNL+myNFM+myNR;

    # get pdf values
    mykvals, mypdfs = get_overlaps(myNconf, myNsites, myNe, mytl, myVconf, plot=False);  
    if(norm_Ne): mypdfs = mypdfs/myNe;
    myax.plot(mykvals, mypdfs, color=mycolors[pairi], label = "$N_e =${:.0f}".format(myNe)+", $N_{conf} =$"+"{:.0f}".format(myNconf));
    #myax.axvline(np.pi/myNconf, color=mycolors[pairi], linestyle="dashed", label="$k^0$");

# format
myax.axvline(np.pi/20, color="black", label = "$k_n^{target}$");
myax.set_xlabel("$k_n$",fontsize=myfontsize);
myax.set_xlim(mykvals[0], mykvals[-1]);
pdf_ylabel = "$ \sum_{k_m} |\langle k_m | k_n \\rangle |^2$";
if(norm_Ne): pdf_ylabel = "$\\frac{1}{N_e} $"+pdf_ylabel;
myax.set_ylabel(pdf_ylabel,fontsize=myfontsize);
myax.set_ylim(0.0,0.5);
myax.legend(fontsize = myfontsize);

# show
plt.tight_layout();
plt.show();
