'''
Christian Bunker
M^2QM at UF
February 2023

Play around with absorbing and emitting boundary conditions
'''

from transport import bardeen, fci_mod
from transport.bardeen import Hsysmat

import numpy as np

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 3;

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

def print_H_j(H) -> None:
    if(len(np.shape(H)) != 4): raise ValueError;
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def print_H_alpha(H) -> None:
    if(len(np.shape(H)) != 4): raise ValueError;
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

def plot_H_alpha(H) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    if(len(np.shape(H)) != 4): raise ValueError;
    n_loc_dof = np.shape(H)[-1];
    spatial_orbs = np.shape(H)[0];
    assert(spatial_orbs % 2 == 1);
    mid = spatial_orbs // 2;
    jvals = np.array(range(-mid,mid+1));
    fig, axes = plt.subplots(n_loc_dof);
    if(n_loc_dof == 1): axes = [axes];
    axes[0].set_title("Hamiltonian");
    for alpha in range(n_loc_dof):
        axes[alpha].plot(jvals,np.real(np.diag(HL[:,:,alpha,alpha])),color='black',linestyle='solid');
        axes[alpha].plot(jvals,np.imag(np.diag(HL[:,:,alpha,alpha])),color='black',linestyle='dashed');
    plt.show();

def h_kondo(J, s2) -> np.ndarray:
    '''
    Kondo interaction between spin 1/2 and spin s2
    '''
    n_loc_dof = int(2*(2*s2+1));
    h = np.zeros((n_loc_dof,n_loc_dof),dtype=complex);
    if(s2 == 0.5):
        h[0,0] = 1;
        h[1,1] = -1;
        h[2,2] = -1;
        h[3,3] = 1;
        h[1,2] = 2;
        h[2,1] = 2;
        h *= J/4;
    else: raise NotImplementedError;
    return h;

def h_tb(t, V, N) -> np.ndarray:
    if(not isinstance(V, np.ndarray)): raise TypeError;
    spatial_orbs = N;
    n_loc_dof = len(V);
    h_4d = np.zeros((spatial_orbs,spatial_orbs,n_loc_dof,n_loc_dof),dtype=complex);
    for spacei in range(spatial_orbs):
        h_4d[spacei,spacei] += V;
        if(spacei < spatial_orbs-1):
            h_4d[spacei+1,spacei] += -t;
            h_4d[spacei,spacei+1] += -t;
    return h_4d;

def get_self_energy(t, V, E) -> np.ndarray:
    if(not isinstance(t, float) or t < 0): raise TypeError;
    if(not isinstance(V, float)): raise TypeError;
    dummy = (E-V)/(-2*t);
    return -(dummy+np.lib.scimath.sqrt(dummy*dummy-1));

def couple_to_cont(H, E, alpha0):
    '''
    Couple a 4d Hamiltonian H to a continuum state with energy E and spin alpha0
    by using absorbing/emitting bcs
    '''
    if(len(np.shape(H)) != 4): raise ValueError;

    # get the self energy
    selfenergy = get_self_energy(-np.real(H[0,1,alpha0,alpha0]),np.real(H[0,0,alpha0,alpha0]),E);

    # emit on the left
    H[0,0,alpha0,alpha0] += np.conj(selfenergy);

    # absorb on the right
    H[-1,-1,alpha0,alpha0] += selfenergy;

    return H;

def get_eigs(h_4d, E_cutoff) -> tuple:
    '''
    Get eigenvalues and eigenvectors of a 4d (non hermitian) hamiltonian
    '''

    # make 2d
    h_2d = fci_mod.mat_4d_to_2d(h_4d);
    eigvals, eigvecs = np.linalg.eig(h_2d);
    
    # sort
    inds = np.argsort(eigvals);
    eigvals = eigvals[inds];
    eigvecs = eigvecs[:,inds].T;
    
    # truncate
    eigvecs = eigvecs[eigvals < E_cutoff];
    eigvals = eigvals[eigvals < E_cutoff];
    
    return eigvals, eigvecs;

def get_fourier_coefs(wf_full,n_loc_dof,ninf=10) -> np.ndarray:
    '''
    Get the comple fourier coefficients for a wf
    '''
    if(not isinstance(wf_full, np.ndarray)): raise TypeError;
    if(not isinstance(ninf,int)): raise TypeError;

    # decompose left and right
    import matplotlib
    import matplotlib.pyplot as plt
    mycolors=matplotlib.colormaps['tab10'].colors; # differentiates spin comps
    mystyles=['solid','dashed']; # differentiates real vs imaginary
    figtest, (axright, axleft, axcomp) = plt.subplots(3, sharex=True);

    # color by spin
    for sigma in range(n_loc_dof):
        wf = wf_full[sigma::n_loc_dof];
        assert(len(wf) % 2 == 1);
        L = len(wf) // 2;
        jvals = np.array(range(-L,L+1));

        # get fourier coef c_n for n \in {-ninf,...+ninf}
        cns = np.empty((2*ninf+1,),dtype=complex);
        for n in range(-ninf,ninf+1):
            cns[n] = np.trapz(wf*np.exp(-complex(0,1)*n*np.pi*jvals/L)/(2*L), jvals);

        # break up into right moving and left moving parts
        wf_right, wf_left = np.zeros_like(wf), np.zeros_like(wf);
        for n in range(-ninf,ninf+1):
            if(n>0): # right moving
                wf_right += cns[n]*np.exp(complex(0,1)*n*np.pi*jvals/L);
            elif(n<0): # left moving
                wf_left += cns[n]*np.exp(complex(0,1)*n*np.pi*jvals/L);

        axright.plot(jvals, np.real(wf_right),linestyle=mystyles[0],color=mycolors[sigma]);
        axright.plot(jvals, np.imag(wf_right),linestyle=mystyles[1],color=mycolors[sigma]);
        axleft.plot(jvals, np.real(wf_left),linestyle=mystyles[0],color=mycolors[sigma]);
        axleft.plot(jvals, np.imag(wf_left),linestyle=mystyles[1],color=mycolors[sigma]);
        axcomp.plot(jvals, np.real((np.conj(wf_left)*wf_left)/(np.conj(wf)*wf)),linestyle=mystyles[0],color=mycolors[sigma]);

    # show
    axright.set_ylabel("Right going");
    axleft.set_ylabel("Left going");
    axright.set_title("Bound state Fourier decomposition");
    axcomp.set_ylabel("|Left going|/|all|");
    plt.show();
    

def plot_eigs(H, E0, alpha0, E_cutoff) -> None:
    '''
    '''

    # unpack
    if(len(np.shape(H)) != 4): raise ValueError;
    n_loc_dof = np.shape(H)[-1];
    spatial_orbs = np.shape(H)[0];
    assert(spatial_orbs % 2 == 1);
    mid = spatial_orbs // 2;
    jvals = np.array(range(-mid,mid+1));

    # eigenstates
    Es, psis = get_eigs(H,E_cutoff);

    # operators
    Sz_op = np.diagflat([complex(1,0) if i%2==0 else -1.0 for i in range(len(psis[0]))]);
    Sx_op = np.zeros_like(Sz_op);
    for i in range(len(Sx_op)-1): Sx_op[i,i+1] = 1.0; Sx_op[i+1,i] = 1.0;
    if(n_loc_dof != 2): # these operators are not valid
        Sz_op, Sx_op = np.zeros_like(Sz_op), np.zeros_like(Sx_op);
        Sz_op[0,0], Sx_op[0,0] = np.nan, np.nan

    #### plot bound states
    import matplotlib
    import matplotlib.pyplot as plt
    mycolors=matplotlib.colormaps['tab10'].colors; # differentiates spin comps
    mystyles=['solid','dashed']; # differentiates real vs imaginary
    print("Plotting bound states with absorbing/emitting bcs.");

    # look for the bound state coupled to the continuum
    V_cont, t_cont = np.real(H[0,0,alpha0,alpha0]), -np.real(H[0,1,alpha0,alpha0]);
    ks = np.arccos((Es - V_cont)/(-2*t_cont));
    k_cont = np.arccos((E0 - V_cont)/(-2*t_cont));
    print("--->",k_cont)
    coupled_continuum = False;
    for m in range(len(Es)):
        psim = psis[m];
        klocal = psim[alpha0+n_loc_dof]/psim[alpha0+0];
        print(m,Es[m].round(4),ks[m].round(4),klocal.round(4));

        # only plot the coupled state
        if(abs(np.real(Es[m]-E0)) < 1e-9 or True):
            coupled_continuum = True;
            Szm = np.dot(np.conj(psim),np.dot(Sz_op,psim));
            Sxm = np.dot(np.conj(psim),np.dot(Sx_op,psim));


            # plot spin components in different colors
            myfig, (wfax, derivax) = plt.subplots(2);
            for sigma in range(n_loc_dof):
                    psimup = psim[sigma::n_loc_dof];

                    # real is solid, dashed is imaginary
                    wfax.plot(np.real(jvals), 1e-6*sigma+np.real(psimup),color=mycolors[sigma],linestyle=mystyles[0]);
                    wfax.plot(np.real(jvals), 1e-6*sigma+np.imag(psimup),color=mycolors[sigma],linestyle=mystyles[1]);
                    derivax.plot(np.real(jvals), 1e-6*sigma+np.real(complex(0,-1)*np.gradient(psimup)),color=mycolors[sigma],linestyle=mystyles[0]);
                    derivax.plot(np.real(jvals), 1e-6*sigma+np.imag(complex(0,-1)*np.gradient(psimup)),color=mycolors[sigma],linestyle=mystyles[1]); 
            # show
            wfax.set_ylabel('$\psi$');
            derivax.set_ylabel('$-i\hbar d \psi/dj$');
            wfax.set_title("Bound state: <S_z> = "+str(Szm)+", <S_x> = "+str(Sxm));
            plt.show();

            # fourier decomp
            get_fourier_coefs(psim,n_loc_dof);

    # check
    if(not coupled_continuum): raise Exception("bound state energy not coupled to continuum");

#################################################################
#### visualize eigenspectrum of different HL's

# redo 1d spinless bardeen theory with absorbing/emitting bcs
# for no reflection, absorbing bcs should be smooth
# look at xgz paper for better spin-flip bcs
# spin rotation region + infinite wall

if True: # spinless case

    # setup
    NL = 17;
    tL = 1.0*np.eye(1);
    VL = 0.0*np.eye(len(tL));

    # construct well and add spin parts
    E_cut = -1.7;
    E_cont = -1.733; # energy of the continuum state
    alpha_cont = 0; # spin of the continuum state
    HL = h_tb(tL,VL,NL);
    HL = couple_to_cont(HL,E_cont,alpha_cont);
    print_H_alpha(HL);
    plot_H_alpha(HL);

    # plot HL eigenfunctions
    plot_eigs(HL,E_cont,alpha_cont,E_cut);
    
if False: # spinful case

    # setup
    NL = 17;
    tL = np.eye(2);
    VL = 0.0*np.eye(len(tL));
    Jval = -0.5;

    # construct well and add spin parts
    E_cut = -1.7;
    E_cont = -1.733;
    alpha_cont = 0;
    HL = h_tb(tL,VL,NL);

    # mix in spin
    spinpart = Jval*np.array([[-1,2],[2,-1]])/4;
    #HL[NL//2,NL//2] += spinpart;
    HL = couple_to_cont(HL,E_cont,alpha_cont);
    print_H_alpha(HL);
    plot_H_alpha(HL);

    # plot HL eigenfunctions
    plot_eigs(HL,E_cont,alpha_cont,E_cut);

        

