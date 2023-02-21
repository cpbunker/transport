'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) 
benchmarked to exact solution
solved in time-dependent QM using bardeen theory method in transport/bardeen
'''

from transport import bardeen, fci_mod
from transport.bardeen import Hsysmat

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def print_H_alpha(H):
    assert(len(np.shape(H)) == 4);
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

def h_kondo(J,s2):
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

def h_simple(Vinfty,V,Ninfty,N):
    if(not isinstance(Vinfty, np.ndarray)): raise TypeError;
    spatial_orbs = 2*Ninfty+N;
    n_loc_dof = len(Vinfty);
    h_4d = np.zeros((spatial_orbs,spatial_orbs,n_loc_dof,n_loc_dof),dtype=complex);
    t = np.eye(n_loc_dof);
    for spacei in range(spatial_orbs):
        if(spacei < Ninfty):
            h_4d[spacei,spacei] += Vinfty;
        elif(spacei < Ninfty+N):
            h_4d[spacei,spacei] += V;
        else:
            h_4d[spacei,spacei] += Vinfty;
        if(spacei < spatial_orbs-1):
            h_4d[spacei+1,spacei] += -t;
            h_4d[spacei,spacei+1] += -t;
    return h_4d;

def get_eigs(h_4d):
    h_2d = fci_mod.mat_4d_to_2d(h_4d);
    eigvals, eigvecs = np.linalg.eigh(h_2d);
    return eigvals, eigvecs.T;

def plot_eigs(psis,jvals,n_loc_dof,num=8) -> None:
    mycolors=matplotlib.colormaps['tab10'].colors; # differentiates spin comps
    mystyles=['solid','dashed']; # differentiates real vs imaginary

    # operators
    assert(n_loc_dof==2);
    Sz_op = np.diagflat([1.0 if i%2==0 else -1.0 for i in range(len(psis[0]))]);
    Sx_op = np.zeros_like(Sz_op);
    for i in range(len(Sx_op)-1): Sx_op[i,i+1] = 1.0; Sx_op[i+1,i] = 1.0;

    # plot eigenfunctions
    for m in range(num):
        psim = psis[m];
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
                wfax.set_title("<S_z> = "+str(Szm)+", <S_x> = "+str(Sxm));
        plt.show();

#################################################################
#### visualize eigenspectrum of different HL's

if True:

    # setup
    Ninfty = 2;
    NL = 13;
    Ntot = 2*Ninfty+NL;
    assert(Ntot % 2 == 1);
    mid = Ntot // 2;
    js = np.array(range(-mid,mid+1));
    loc_dof = 2;
    Vinfty = 0.5*np.eye(loc_dof);
    V = 0.0*np.eye(loc_dof);

    # construct well and add spin parts
    HL = h_simple(Vinfty,V,Ninfty,NL);
    spin_part = np.array([[0,complex(0,-1)],[complex(0,1),0]]);
    HL[0,0] += spin_part;
    HL[-1,-1] += spin_part;
    print_H_alpha(HL);
    fig, axes = plt.subplots(loc_dof);
    for sigma in range(loc_dof):
        axes[sigma].plot(js,np.diag(HL[:,:,sigma,sigma]));
    plt.show();

    # plot HL eigenfunctions
    Es, psis = get_eigs(HL);
    plot_eigs(psis,js,loc_dof);

    # e^ikx
    ks = np.arccos((Es-V[0,0])/(-2));
    for m in range(4):
        print(Es[m]);
        plusvals = np.exp(complex(0,1)*ks[m]*js);
        minusvals = np.exp(complex(0,-1)*ks[m]*js);
        psi = plusvals;
        #psi = (plusvals-minusvals)/complex(0,2);
        fig, (wfax,derivax) = plt.subplots(2);
        wfax.plot(np.real(js), np.real(psi),color='tab:blue');
        wfax.plot(np.real(js), np.imag(psi),color='tab:blue',linestyle='dashed');
        derivax.plot(np.real(js), np.real(complex(0,-1)*np.gradient(psi)),color='tab:blue');
        derivax.plot(np.real(js), np.imag(complex(0,-1)*np.gradient(psi)),color='tab:blue',linestyle='dashed');
        plt.show();
        

