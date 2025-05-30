'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a spin-1/2 impurity

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt

import sys

def h_kondo(J,s2,spin_trunc,unit_cell):
    '''
    Kondo interaction between spin 1/2 and spin s2
    '''

    # construct in spin space
    n_spin_dof = int(2*(2*s2+1));
    h_spinspace = np.zeros((n_spin_dof,n_spin_dof),dtype=complex);
    if(s2 == 0.5):
        h_spinspace[0,0] = 1;
        h_spinspace[1,1] = -1;
        h_spinspace[2,2] = -1;
        h_spinspace[3,3] = 1;
        h_spinspace[1,2] = 2;
        h_spinspace[2,1] = 2;
        h_spinspace *= J/4;
    else: raise NotImplementedError;

    # truncate in unit space
    if(spin_trunc is not None):
        if(not isinstance(spin_trunc,tuple)): raise TypeError;
        h_spinspace = h_spinspace[spin_trunc[0]:spin_trunc[1],spin_trunc[0]:spin_trunc[1]];
        n_spin_dof = len(h_spinspace);

    # expand into unit cell space
    h_unitspace = np.zeros((unit_cell*n_spin_dof,unit_cell*n_spin_dof),dtype=complex);
    h_unitspace[:n_spin_dof,:n_spin_dof] = h_spinspace[:,:];
    return h_unitspace;
    
if(__name__=="__main__"):

    # top level
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;
    case = sys.argv[1];
    myconverger = sys.argv[2]; # tells code how to evaluate the surface greens function

    # fig standardizing
    myxvals = 199;
    myfontsize = 14;
    mycolors = ["black","darkblue","darkgreen","darkred", "darkcyan", "darkmagenta","darkgray"];
    mymarkers = ["o","^","s","d","*","X","P"];
    mymarkevery = (40, 40);
    mylinewidth = 1.0;
    mypanels = ["(a)","(b)","(c)","(d)"];
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"text.usetex": True}) 

    # tight binding params
    Msites = 1; # non contact interaction
    tl = 1.0;

    # def energy range
    logKlims = -6,0
    Kvals = np.logspace(*logKlims,myxvals, dtype=complex);

#################################################################
#### **DIATOMIC UNIT CELL**
#### Rice-Mele model

if(myconverger=="g_RiceMele" and case in ["VB","CB"]):
    my_unit_cell = 2; # since diatomic

    # Rice-Mele tight binding
    vval = float(sys.argv[3]);
    uval = float(sys.argv[4]); 
    # w is always -tl; # <- change
    band_edges = np.array([np.sqrt(uval*uval+(-tl+vval)*(-tl+vval)),
                           np.sqrt(uval*uval+(-tl-vval)*(-tl-vval))]);
    RiceMele_shift = np.min(-band_edges) + 2*tl; # new band bottom - old band bottom
    if(case=="CB"): RiceMele_shift = np.min(band_edges) + 2*tl; #new band=conduction band!
    RiceMele_Energies = Kvals - 2*tl + RiceMele_shift; # value in the RM band
    RiceMele_numbers = np.arccos(1/(2*vval*(-tl))*(RiceMele_Energies**2 - uval**2 - vval**2 - tl**2));


    # Rice-Mele matrices
    diag_base_RM_spin=np.array([[+uval,0, +vval,0],  # elec up, imp dw, A orb
                                [0,+uval, 0,+vval],  # elec dw, imp up, A orb
                                [+vval,0, -uval,0],  # elec up, imp dw, B orb
                                [0,+vval, 0,-uval]]);# elec dw, imp up, B orb
    offdiag_base_RM_spin=np.array([[0,0,   0,0], 
                                   [0,0,   0,0], 
                                   [-tl,0, 0,0],  
                                   [0,-tl, 0,0]]);

    # inelastic ?
    if(case in ["inelastic"]): inelastic = True; Delta = 0.001; raise NotImplementedError
    else: inelastic = False; Delta = 0.0;

    # set up figure
    num_plots = 4;
    plot_differences = False;
    if(inelastic or not plot_differences): num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(6,3*num_plots/2);

    # iter over effective J
    Jvals = np.array([-0.005,-0.05,-0.5,-5.0]);
    for Jvali in range(len(Jvals)):
        Jval = Jvals[Jvali];
        
        # S dot s
        hSR = h_kondo(Jval,0.5,(1,3),my_unit_cell); # 4x4 matrix
                                                    # 2 for A/B orbs, 2 for up,dw/dw,up
        hSR += diag_base_RM_spin
        print("shape h_kondo = ",np.shape(hSR));

        # leads
        hLL = 1*diag_base_RM_spin;
        hRL = 1*diag_base_RM_spin;

        # source for diatomic system
        source = np.zeros(np.shape(hSR)[0]);
        in_noflip = 0;  # |elec up, imp dw, A orb> = incident channel
        out_noflip = 2; # |elec up, imp dw, B orb> = **no spin flip** transmission channel
        out_flip = 3;   # |elec dw, imp up, B orb> = **spin flip** transmission channel
        source[in_noflip] = 1;

        # package together hamiltonian blocks
        hblocks = [hLL];
        for _ in range(Msites): hblocks.append(np.copy(hSR)); # Msites is size of scattering region
        assert(Msites==1);
        hblocks.append(hRL);
        hblocks = np.array(hblocks); # len(hblocks = 1 + Msites + 1)

        # hopping
        tnn = [];
        for _ in range(len(hblocks)-1): tnn.append(offdiag_base_RM_spin);
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Jvali == 0): 
            print("\nhblocks:\n", np.real(hblocks));
            print("\ntnn:\n", np.real(tnn),"\ntnnn:\n",np.real(tnnn));
            if(inelastic): assert False

        # Menezes' exact results
        kavals = np.arccos((Kvals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tf, menez_Tnf = np.real(menez_Tf), np.real(menez_Tnf);
        del kavals;
        Rvals = np.empty((len(Kvals),len(source)), dtype = float);
        Tvals = np.empty((len(Kvals),len(source)), dtype = float); 
        for Kvali in range(len(Kvals)):

            # energy
            Kval = Kvals[Kvali]; # Eval > 0 always, what I call K in paper
            Energy = Kval-2*tl+RiceMele_shift; #energy that is `Kval` above either VB or CB

            if(Kvali < 3): # verbose
                Rdum,Tdum=wfm.kernel(hblocks,tnn,tnnn,tl,Energy,myconverger,source, 
                                False, False, all_debug = True, verbose = verbose);
            else: # not verbose
                 Rdum,Tdum=wfm.kernel(hblocks,tnn,tnnn,tl,Energy,myconverger,source, 
                                False, False, all_debug = False, verbose = 0);
            Rvals[Kvali] = Rdum;
            Tvals[Kvali] = Tdum;

        # plot tight binding results
        ax0, ax1, ax2, ax3 = 0,1,2,3;
        if(inelastic or not plot_differences): ax0, ax2 = 0,1
        axes[ax0].plot(np.real(Kvals),Tvals[:,out_flip], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[ax2].plot(np.real(Kvals),Tvals[:,out_noflip], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[-1].plot(np.real(Kvals), totals, color="red", label = "total ");
        axes[ax0].set_ylabel('$T_{spin flip}$', fontsize = myfontsize );
        axes[ax2].set_ylabel('$T_{no flip}$', fontsize = myfontsize );
            
       
        # continuum results
        lower_y = 0.08;
        if inelastic:
            #axes[ax0].axvline(0.025, color = "gray");
            axes[ax0].plot(Kvals, menez_Tf, color = mycolors[Jvali],linestyle = "dashed", marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth); 
            axes[ax2].plot(Kvals, menez_Tnf, color = mycolors[Jvali],linestyle = "dashed", marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax0].set_ylim(-0.4*lower_y,0.4)
            axes[ax0].set_ylabel('$T_{f}$', fontsize = myfontsize );
            axes[ax2].set_ylim(-1*lower_y,1*(1+lower_y));
            axes[ax2].set_ylabel('$T_{nf}$', fontsize = myfontsize );

        # differences
        if(not inelastic and plot_differences):
            axes[ax1].plot(np.real(Kvals),abs(Tvals[:,out_flip]-menez_Tf)/menez_Tf,
              color = mycolors[Jvali], label="$J=${:.4f}".format(Jval),
              marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax3].plot(np.real(Kvals),abs(Tvals[:,out_noflip]-menez_Tnf)/menez_Tnf,
              color = mycolors[Jvali], label="$J=${:.4f}".format(Jval),
              marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax0].set_ylim(-0.4*lower_y,0.4)          
            axes[ax1].set_ylim(-0.1*lower_y,0.1);
            axes[ax1].set_ylabel('$|T_{f}-T_{f,c}|/T_{f,c}$', fontsize = myfontsize );
            axes[ax2].set_ylim(-1*lower_y,1*(1+lower_y));          
            axes[ax3].set_ylim(-0.1*lower_y,0.1);
            axes[ax3].set_ylabel('$|T_{nf}-T_{nf,c}|/T_{nf,c}$', fontsize = myfontsize );
    
    # format
    title_str = "$u=${:.2f}, $v=${:.2f}, $w=${:.2f}".format(uval, vval, -tl)
    axes[0].set_title(title_str);
    axes[-1].legend();
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logKlims[0]), 10**(logKlims[1]));
    axes[-1].set_xticks([10**(logKlims[0]), 10**(logKlims[1])]);
    RiceMele_shift_str = "$-E_{min}^{(VB)}, E_{min}^{(VB)}=$"+"{:.2f}".format(np.min(-band_edges))
    if(case=="CB"): RiceMele_shift_str="$-E_{min}^{(CB)},  E_{min}^{(CB)}=$"+"{:.2f}".format(np.min(band_edges))
    RiceMele_shift_str += ",  $ka/\pi \in $[{:.2f},{:.2f}]".format(np.real(RiceMele_numbers[0]/np.pi), np.real(RiceMele_numbers[-1]/np.pi))
    axes[-1].set_xlabel("$E$"+RiceMele_shift_str,fontsize = myfontsize);

    # show 
    plt.tight_layout();
    fname = 'figs/'+case+'.pdf'
    plt.show();

#################################################################
#### replication of continuum solution, monatomic unit cell

elif(myconverger=="g_closed" and case in ["continuum", "inelastic"]):
    my_unit_cell = 1;

    # inelastic ?
    if(case in ["inelastic"]): inelastic = True; Delta = 0.001;
    else: inelastic = False; Delta = 0.0;
    num_plots = 4;
    if inelastic: num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # iter over effective J
    Jvals = np.array([-0.005,-0.05,-0.5,-5.0]);
    for Jvali in range(len(Jvals)):
        Jval = Jvals[Jvali];
        
        # S dot s
        hSR = h_kondo(Jval,0.5,(1,3),my_unit_cell)

        # zeeman splitting
        hzeeman = np.array([[0, 0, 0, 0],
                        [0,Delta, 0, 0],
                        [0, 0, 0, 0], # spin flip gains PE delta
                        [0, 0, 0, Delta]]);
        hzeeman = hzeeman[1:3,1:3]; # truncate to coupled channels
        print("hSR =\n",np.real(hSR));
        print("hzeeman =\n",np.real(hzeeman));
        hSR += hzeeman;
        print("hSR+hzeeman =\n",np.real(hSR));

        # leads
        hLL = np.copy(hzeeman);
        hRL = np.copy(hzeeman)

        # source = up electron, down impurity
        sourcei, flipi = 1,0
        source = np.zeros(np.shape(hSR)[0]);
        source[sourcei] = 1;

        # package together hamiltonian blocks
        hblocks = [hLL];
        for _ in range(Msites): hblocks.append(np.copy(hSR));
        hblocks.append(hRL);
        hblocks = np.array(hblocks);

        # hopping
        tnn = [];
        for _ in range(len(hblocks)-1): tnn.append(-tl*np.eye(*np.shape(hSR)));
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Jvali == 0): 
            print("\nhblocks:\n", np.real(hblocks));
            print("\ntnn:\n", np.real(tnn),"\ntnnn:\n", np.real(tnnn));
            if(inelastic): assert False;

        # Menezes' exact results
        kavals = np.arccos((Kvals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tf, menez_Tnf = np.real(menez_Tf), np.real(menez_Tnf);
        Rvals = np.empty((len(Kvals),len(source)), dtype = float);
        Tvals = np.empty((len(Kvals),len(source)), dtype = float); 
        for Kvali in range(len(Kvals)):

            # energy
            Kval = Kvals[Kvali]; # Eval > 0 always, what I call K in paper
            Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            if(Kvali < 3): # verbose
                Rdum,Tdum=wfm.kernel(hblocks,tnn,tnnn,tl,Energy,myconverger,source, 
                                False, False, all_debug = True, verbose = verbose);
            else: # not verbose
                 Rdum,Tdum=wfm.kernel(hblocks,tnn,tnnn,tl,Energy,myconverger,source, 
                                False, False, all_debug = False, verbose = 0);
            Rvals[Kvali] = Rdum;
            Tvals[Kvali] = Tdum;

        # plot tight binding results
        ax0, ax1, ax2, ax3 = 0,1,2,3;
        if inelastic: ax0, ax2 = 0,1
        axes[ax0].plot(np.real(Kvals),Tvals[:,flipi], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[ax2].plot(np.real(Kvals),Tvals[:,sourcei], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[1].plot(np.real(Kvals), totals, color="red", label = "total ");
        #axes[2].plot(np.real(Kvals),Rvals[:,flipi], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        #axes[3].plot(np.real(Kvals),Rvals[:,sourcei], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        
        # continuum results
        lower_y = 0.08;
        if inelastic:
            #axes[ax0].axvline(0.025, color = "gray");
            axes[ax0].plot(Kvals, menez_Tf, color = mycolors[Jvali],linestyle = "dashed", marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth); 
            axes[ax2].plot(Kvals, menez_Tnf, color = mycolors[Jvali],linestyle = "dashed", marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax0].set_ylim(-0.4*lower_y,0.4)
            axes[ax0].set_ylabel('$T_{f}$', fontsize = myfontsize );
            axes[ax2].set_ylim(-1*lower_y,1*(1+lower_y));
            axes[ax2].set_ylabel('$T_{nf}$', fontsize = myfontsize );
            
        # differences
        if not inelastic:
            axes[ax1].plot(np.real(Kvals),abs(Tvals[:,flipi]-menez_Tf)/menez_Tf,color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax3].plot(np.real(Kvals),abs(Tvals[:,sourcei]-menez_Tnf)/menez_Tnf,color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax0].set_ylim(-0.4*lower_y,0.4)
            axes[ax0].set_ylabel('$T_{f}$', fontsize = myfontsize );
            axes[ax1].set_ylim(-0.1*lower_y,0.1);
            axes[ax1].set_ylabel('$|T_{f}-T_{f,c}|/T_{f,c}$', fontsize = myfontsize );
            axes[ax2].set_ylim(-1*lower_y,1*(1+lower_y));
            axes[ax2].set_ylabel('$T_{nf}$', fontsize = myfontsize );
            axes[ax3].set_ylim(-0.1*lower_y,0.1);
            axes[ax3].set_ylabel('$|T_{nf}-T_{nf,c}|/T_{nf,c}$', fontsize = myfontsize );
    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logKlims[0]), 10**(logKlims[1]));
    axes[-1].set_xticks([10**(logKlims[0]), 10**(logKlims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.065, y = 0.74, fontsize = myfontsize); 
    plt.tight_layout();
    fname = 'figs/'+case+'.pdf'
    plt.show();


#################################################################
#### physical origin

elif(case in ["origin"]):
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # tight binding parameters
    th = 1.0;
    U1 = 0.0;
    U2 = 100.0;

    # iter over effective J by changing epsilon
    Jvals = np.array([-0.005,-0.05,-0.5,-5.0]);
    epsvals = (U1-U2)/2 + np.sqrt(U1*U2 + np.power((U1-U2)/2,2) - 2*th*th*(U1+U2)/Jvals);
    for epsi in range(len(epsvals)):
        epsilon = epsvals[epsi];
        Jval = Jvals[epsi];
        print("Jval = ",Jval);
        print("U1 - epsilon = ",U1 - epsvals[epsi]);
        print("U2+epsilon = ",U2+epsvals[epsi]);

        # SR physics: site 1 is in chain, site 2 is imp with large U
        hSR = np.array([[U1,-th,th,0], # up down, -
                        [-th,epsilon, 0,-th], # up, down (source)
                        [th, 0, epsilon, th], # down, up (flip)
                        [0,-th,th,U2+2*epsilon]]); # -, up down
        hSR += (Jvals[epsi]/4)*np.eye(4);
        
        # source = up electron, down impurity
        source = np.zeros(np.shape(hSR)[0]);
        sourcei, flipi = 1,2;
        source[sourcei] = 1;

        # lead physics
        hLL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);
        hRL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);

        # package together hamiltonian blocks
        hblocks = np.array([hLL, hSR, hRL]);
        for hb in hblocks: hb += -epsilon*np.eye(len(source));  # constant shift so source is at zero
        tnn_mat = -tl*np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]);
        tnn = np.array([np.copy(tnn_mat), np.copy(tnn_mat)]);
        tnnn = np.zeros_like(tnn)[:-1];
        #if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn, "\ntnnn:", tnnn)

        if True: # do the downfolding explicitly
            matA = np.array([[0, 0],[0,0]]);
            matB = np.array([[-th,-th],[th,th]]);
            matC = np.array([[-th,th],[-th,th]]);
            matD = np.array([[U1-epsilon, 0],[0,U2+epsilon]]);
            mat_downfolded = matA - np.dot(matB, np.dot(np.linalg.inv(matD), matC))  
            print("Downfolded J = ",2*abs(mat_downfolded[0,0]) );
        
        # sweep over range of energies
        # def range
        logElims = -6,0
        Evals = np.logspace(*logElims,myxvals, dtype=complex);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            Rdum, Tdum =wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot Tvals vs E
        axes[0].plot(Evals,Tvals[:,flipi], color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        axes[1].plot(Evals,Tvals[:,sourcei], color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        totals = Tvals[:,sourcei] + Tvals[:,flipi] + Rvals[:,sourcei] + Rvals[:,flipi];
        #axes[1].plot(Evals, totals, color="red");

        # menezes prediction in the continuous case
        axes[0].plot(Evals, menez_Tf, color = mycolors[epsi],linestyle = "dashed", marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth); 
        axes[1].plot(Evals, menez_Tnf, color = mycolors[epsi],linestyle = "dashed", marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        lower_y = 0.08
        axes[0].set_ylim(-0.4*lower_y,0.4)
        axes[0].set_ylabel('$T_{f}$', fontsize = myfontsize );
        axes[1].set_ylim(-1*lower_y,1*(1+lower_y));
        axes[1].set_ylabel('$T_{nf}$', fontsize = myfontsize );

    # format
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.065, y = 0.74, fontsize = myfontsize);
    plt.tight_layout();
    fname = 'figs/'+case+'.pdf'
    plt.show();
    
else: raise NotImplementedError("case = "+case+", myconverger = "+str(myconverger));

