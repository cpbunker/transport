'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

#from transport import fci_mod

import numpy as np

##################################################################################
#### driver of transmission coefficient calculations

def kernel(h, tnn, tnnn, tl, E, Ajsigma, verbose = 0, all_debug = True):
    '''

    '''

    return;

def Hsysmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR):
    '''
    Make the TB Hamiltonian for the full system
    '''
    for N in [Ninfty, NL, NC, NR]:
        if(not isinstance(N, int)): raise TypeError;
    for N in [Ninfty, NL, NR]:
        if(N <= 0): raise ValueError;
    minusinfty = -NC - NL - Ninfty;
    plusinfty = NC + NR + Ninfty;
    Nsites = -minusinfty + plusinfty + 1;

    # Hamiltonian matrix
    Hmat = np.zeros((Nsites,Nsites));
    for j in range(minusinfty, plusinfty+1):

        # diag
        if(j < -NL - NC):           
            Hmat[j-minusinfty,j-minusinfty] += Vinfty
        elif(j >= -NL-NC and j < -NC):
            Hmat[j-minusinfty,j-minusinfty] += VL;
        elif(j >= -NC and j <= NC):
            Hmat[j-minusinfty,j-minusinfty] += VC;
        elif(j > NC and j <= NC+NR):
            Hmat[j-minusinfty,j-minusinfty] += VR;
        elif(j > NC+NR):
            Hmat[j-minusinfty,j-minusinfty] += Vinfty;

        # off diag
        if(j < -NL - NC):           
            Hmat[j-minusinfty,j+1-minusinfty] += -tinfty;
            Hmat[j+1-minusinfty,j-minusinfty] += -tinfty;
        elif(j >= -NL-NC and j < -NC):
            Hmat[j-minusinfty,j+1-minusinfty] += -tL;
            Hmat[j+1-minusinfty,j-minusinfty] += -tL;
        if(j >= -NC and j < NC):
            Hmat[j-minusinfty,j+1-minusinfty] += -tC;
            Hmat[j+1-minusinfty,j-minusinfty] += -tC;
        elif(j > NC and j <= NC+NR):
            Hmat[j-minusinfty,j-1-minusinfty] += -tR;
            Hmat[j-1-minusinfty,j-minusinfty] += -tR; 
        elif(j > NC+NR):
            Hmat[j-minusinfty,j-1-minusinfty] += -tinfty;
            Hmat[j-1-minusinfty,j-minusinfty] += -tinfty;         
            
    return Hmat, minusinfty;

def HLmat(tinfty, tL, tC, Vinfty, VL, VC, Ninfty, NL, NC, NR):
    '''
    Make the TB Hamiltonian for a left side finite quantum well of NL sites
    '''

    return Hsysmat(tinfty, tL, tC, tC, Vinfty, VL, VC, VC, Ninfty, NL, NC, NR);


def HRmat(tinfty, tC, tR, Vinfty, VC, VR, Ninfty, NL, NC, NR):
    '''
    Make the TB Hamiltonian for a right side finite quantum well of NR sites
    '''          
            
    return Hsysmat(tinfty, tC, tC, tR, Vinfty, VC, VC, VR, Ninfty, NL, NC, NR);

##################################################################################
#### util functions

def plot_wfs(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR):
    '''
    Visualize the problem by plotting some LL wfs against Hsys
    '''
    fig, axes = plt.subplots(4, sharex = True);

    # plot left well eigenstates
    HL, offset = HLmat(tinfty, tL, tC, Vinfty, VL, VC, Ninfty, NL, NC, NR);
    jvals = np.array(range(len(HL))) + offset;
    axes[0].plot(jvals, np.diag(HL), color = 'black', linestyle = 'dashed', linewidth = 2);
    Ems, psims = np.linalg.eigh(HL);
    Ems_bound = Ems[Ems + 2*tL < VC];
    ms_bound = np.linspace(0,len(Ems_bound)-1,3,dtype = int);
    for counter in range(len(ms_bound)):
        m = ms_bound[counter]
        axes[0].plot(jvals[jvals <= NC+NR], psims[:,m][jvals <= NC+NR], color = mycolors(counter));
        axes[0].plot([NC+NR,jvals[-1]],(2*tL+ Ems[m])*np.ones((2,)), color = mycolors(counter));
    axes[0].set_ylabel('$V_j/t_L$');
    axes[0].set_ylim(VL-2*VC,VL+2*VC);

    # plot system ham
    if True:
        Hsys, _ = Hsysmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR);
        axes[1].plot(jvals, np.diag(Hsys), color = 'black', linestyle = 'dashed', linewidth = 2);
        axes[1].plot(jvals, np.diag(Hsys-HL), color = 'darkblue', linestyle = 'dashed', linewidth = 2);
        axes[1].set_ylabel('$V_j/t_L$');

    # plot (Hsys-HL)*psi_m
    if True:
        for counter in range(len(ms_bound)):
            m = ms_bound[counter];
            axes[2].plot(jvals, np.dot(Hsys-HL,psims[:,m]), color = mycolors(counter));
        axes[2].set_ylabel('$(H_{sys}-H_L) \psi_m $');

    # plot right well eigenstates
    HR, offset = HRmat(tinfty, tC, tR, Vinfty, VC, VR, Ninfty, NL, NC, NR);
    axes[3].plot(jvals, np.diag(HR), color = 'black', linestyle = 'dashed', linewidth = 2);
    Emprimes, psimprimes = np.linalg.eigh(HR);
    for counter in range(len(ms_bound)):
        mprime = ms_bound[counter];
        axes[3].plot(jvals[jvals > -NL-NC], psimprimes[:,mprime][jvals > -NL-NC], color = mycolors(counter));
        axes[3].plot([jvals[0],-NL-NC],(2*tL+ Ems[mprime])*np.ones((2,)), color = mycolors(counter));
    axes[3].set_ylabel('$V_j/t_L$');
    axes[3].set_ylim(VR-2*VC,VR+2*VC);
        
    # format
    axes[-1].set_xlabel('$j$');
    plt.tight_layout();
    plt.show();
    
def TvsE(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR, VCprime = None):
    '''
    Calculate a transmission coefficient for each LL eigenstate and return
    these as a function of their energy
    '''
    if VCprime == None: VCprime = VC;

    # left well eigenstates
    HL, _ = Hsysmat(tinfty, tL, tC, tC, Vinfty, VL, VC, VCprime, Ninfty, NL, NC, NR);
    Ems, psims = np.linalg.eigh(HL); # eigenstates of the left well
    kms = np.arccos((Ems-VL)/(-2*tL)); # wavenumbers in the left well

    # right well eigenstates
    HR, _ = HRmat(tinfty, tC, tR, Vinfty, VC, VR, Ninfty, NL, NC, NR);
    Emprimes, psimprimes = np.linalg.eigh(HR); # eigenstates of the right well
    kmprimes = np.arccos((Emprimes-VR)/(-2*tR)); # wavenumbers in the right well

    # operator
    Hsys, offset = Hsysmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR);
    op = Hsys - HL;

    # debugging
    if False:
        jvals = np.array(range(len(Hsys))) + offset;
        for H in [HL,HR,Hsys]:
            myfig,myax = plt.subplots();
            myax.plot(jvals, np.diag(H), color = 'black', linestyle = 'dashed', linewidth = 2);
            myfig.show();

    # compute T
    Tms = np.zeros_like(Ems);
    for m in range(len(Ems)):
        mprime = m;
        M = np.dot(psimprimes[:,mprime],np.dot(op,psims[:,m]));
        Tms[m] = M*np.conj(M)*NL/(kms[m]*tL)*NR/(kmprimes[mprime]*tR);
        
    return Ems, Tms;

##################################################################################
#### test code

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # fig standardizing
    myxvals = 199;
    myfontsize = 14;
    mycolors = cm.get_cmap('Set1');
    mymarkers = ["o","^","s","d","*","X","P"];
    mymarkevery = (40, 40);
    mylinewidth = 1.0;
    mypanels = ["(a)","(b)","(c)","(d)"];
    plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

    # left lead quantum well test
    # tb params, in tL
    tinfty = 1.0;
    tL = 1.0;
    tC = 1*tL;
    tR = 1*tL;
    ts = (tinfty, tL, tC, tR);
    Vinfty = tL/2;
    VL = 0.0;
    VC = tL/10;
    VR = 1*VL;
    Vs = (Vinfty, VL, VC, VR);
    Ninfty = 100;
    NL = 100;
    NC = NL//20;
    NR = 100;
    Ns = (Ninfty, NL, NC, NR);

    # visualize the problem
    if False:
        fig, ax = plt.subplots();
        Hsys, offset = Hsysmat(*ts, Vinfty, VL, VC, VC, *Ns);
        jvals = np.array(range(len(Hsys))) + offset;
        ax.plot(jvals, np.diag(Hsys), color = 'black', linestyle = 'dashed', linewidth = 2);
        ax.set_ylabel('$V_j/t_L$');
        ax.set_xlabel('$j$');
        plt.tight_layout();
        plt.show();
        plot_wfs(*ts, *Vs, *Ns);

    # test matrix elements vs NC
    if False:
        del NC;
        NCvals = [0,2,7];
        numplots = len(NCvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different barrier width
        for NCi in range(len(NCvals)):
            Evals, Tvals = TvsE(*ts, *Vs, Ninfty, NL, NCvals[NCi], NR);
            Evals = (Evals+2*tL);
            Evals, Tvals = Evals[Evals <= VC], Tvals[Evals <= VC]; # bound states only
            axes[NCi].scatter(Evals, Tvals, marker=mymarkers[0], color = mycolors(0));
            axes[NCi].set_ylim(0,1.1*max(Tvals));

            # compare
            logElims = -3,0
            Evals = np.logspace(*logElims,myxvals, dtype=complex);
            kavals = np.arccos((Evals-2*tL-VL)/(-2*tL));
            kappavals = np.arccosh((Evals-2*tL-VC)/(-2*tL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*(2*NCvals[NCi]+1)*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[NCi].plot(Evals,np.real(ideal_Tvals), color = 'black', linewidth = mylinewidth);
            axes[NCi].set_ylabel('$T$');
            axes[NCi].set_title('$d = '+str(2*NCvals[NCi]+1)+'a$', x=0.17, y = 0.7, fontsize = myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$');
        plt.tight_layout();
        plt.savefig("figs/bardeen/NC");

    # test matrix elements vs VC
    if False:
        del VC;
        VCvals = [0.01,0.1,1.0];
        numplots = len(VCvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different barrier height
        for VCi in range(len(VCvals)):
            Evals, Tvals = TvsE(*ts, 5*VCvals[VCi], VL, VCvals[VCi], VR, *Ns);
            Evals = (Evals+2*tL);
            Evals, Tvals = Evals[Evals <= VCvals[VCi]], Tvals[Evals <= VCvals[VCi]]; # bound states only
            axes[VCi].scatter(Evals, Tvals, marker=mymarkers[0], color = mycolors(0));
            axes[VCi].set_ylim(0,1.1*max(Tvals));

            # compare
            logElims = -3,0
            Evals = np.logspace(*logElims,myxvals, dtype=complex);
            kavals = np.arccos((Evals-2*tL-VL)/(-2*tL));
            kappavals = np.arccosh((Evals-2*tL-VCvals[VCi])/(-2*tL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*(2*NC+1)*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[VCi].plot(Evals,np.real(ideal_Tvals), color = 'black', linewidth = mylinewidth);
            axes[VCi].set_ylabel('$T$');
            axes[VCi].set_title('$V_C = '+str(VCvals[VCi])+'$', x=0.17, y = 0.7, fontsize = myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$');
        plt.tight_layout();
        plt.savefig("figs/bardeen/VC");

    # test matrix elements vs NR
    if False:
        del NR;
        NRvals = [200];
        numplots = len(NRvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);
        symmetric = False;

        # bardeen results for different well thicknesses
        for NRi in range(len(NRvals)):
            if symmetric:
                fname = "figs/bardeen/NR_symmetric";
                Evals, Tvals = TvsE(*ts, *Vs, Ninfty, NRvals[NRi], NC, NRvals[NRi]);
            else:
                fname = "figs/bardeen/NR";
                Evals, Tvals = TvsE(*ts, *Vs, Ninfty, NL, NC, NRvals[NRi]);
            Evals = (Evals+2*tL);
            Evals, Tvals = Evals[Evals <= VC], Tvals[Evals <= VC]; # bound states only
            axes[NRi].scatter(Evals, Tvals, marker=mymarkers[0], color = mycolors(0));
            axes[NRi].set_ylim(0,1.1*max(Tvals));

            # compare
            logElims = -3,0
            Evals = np.logspace(*logElims,myxvals, dtype=complex);
            kavals = np.arccos((Evals-2*tL-VL)/(-2*tL));
            kappavals = np.arccosh((Evals-2*tL-VC)/(-2*tL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*(2*NC+1)*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[NRi].plot(Evals,np.real(ideal_Tvals), color = 'black', linewidth = mylinewidth);
            axes[NRi].set_ylabel('$T$');
            axes[NRi].set_title('$N_R = '+str(NRvals[NRi])+'$', x=0.17, y = 0.7, fontsize = myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$');
        plt.tight_layout();
        plt.show();

    # test matrix elements vs VCprime
    if True:
        VCPvals = [0.01,0.1,1.0,10.0];
        numplots = len(VCPvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different well thicknesses
        for VCPi in range(len(VCPvals)):
            Evals, Tvals = TvsE(*ts, *Vs, *Ns, VCprime = VCPvals[VCPi]);
            Evals = (Evals+2*tL);
            Evals, Tvals = Evals[Evals <= VC], Tvals[Evals <= VC]; # bound states only
            axes[VCPi].scatter(Evals, Tvals, marker=mymarkers[0], color = mycolors(0));
            axes[VCPi].set_ylim(0,1.1*max(Tvals));

            # compare
            logElims = -3,0
            Evals = np.logspace(*logElims,myxvals, dtype=complex);
            kavals = np.arccos((Evals-2*tL-VL)/(-2*tL));
            kappavals = np.arccosh((Evals-2*tL-VC)/(-2*tL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*(2*NC+1)*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[VCPi].plot(Evals,np.real(ideal_Tvals), color = 'black', linewidth = mylinewidth);
            axes[VCPi].set_ylabel('$T$');
            axes[VCPi].set_title("$V_C' = "+str(VCPvals[VCPi])+"$", x=0.17, y = 0.7, fontsize = myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$');
        plt.tight_layout();
        plt.show();

    





    
    


    








