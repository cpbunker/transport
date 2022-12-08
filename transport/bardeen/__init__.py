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
    if(NC % 2 != 1): raise ValueError; # NC must be odd
    littleNC = NC // 2;
    del NC
    minusinfty = -littleNC - NL - Ninfty;
    plusinfty = littleNC + NR + Ninfty;
    Nsites = -minusinfty + plusinfty + 1;

    # Hamiltonian matrix
    Hmat = np.zeros((Nsites,Nsites));
    for j in range(minusinfty, plusinfty+1):

        # diag
        if(j < -NL - littleNC):           
            Hmat[j-minusinfty,j-minusinfty] += Vinfty
        elif(j >= -NL-littleNC and j < -littleNC):
            Hmat[j-minusinfty,j-minusinfty] += VL;
        elif(j >= -littleNC and j <= littleNC):
            Hmat[j-minusinfty,j-minusinfty] += VC;
        elif(j > littleNC and j <= littleNC+NR):
            Hmat[j-minusinfty,j-minusinfty] += VR;
        elif(j > littleNC+NR):
            Hmat[j-minusinfty,j-minusinfty] += Vinfty;

        # off diag
        if(j < -NL - littleNC):           
            Hmat[j-minusinfty,j+1-minusinfty] += -tinfty;
            Hmat[j+1-minusinfty,j-minusinfty] += -tinfty;
        elif(j >= -NL-littleNC and j < -littleNC):
            Hmat[j-minusinfty,j+1-minusinfty] += -tL;
            Hmat[j+1-minusinfty,j-minusinfty] += -tL;
        if(j >= -littleNC and j < littleNC):
            Hmat[j-minusinfty,j+1-minusinfty] += -tC;
            Hmat[j+1-minusinfty,j-minusinfty] += -tC;
        elif(j > littleNC and j <= littleNC+NR):
            Hmat[j-minusinfty,j-1-minusinfty] += -tR;
            Hmat[j-1-minusinfty,j-minusinfty] += -tR; 
        elif(j > littleNC+NR):
            Hmat[j-minusinfty,j-1-minusinfty] += -tinfty;
            Hmat[j-1-minusinfty,j-minusinfty] += -tinfty;         
            
    return Hmat, minusinfty;

def HLmat(tinfty, tL, tC, Vinfty, VL, VC, Ninfty, NL, NC, NR):
    '''
    Make the TB Hamiltonian for a left side finite quantum well of NL sites
    '''
    raise Exception
    return Hsysmat(tinfty, tL, tC, tC, Vinfty, VL, VC, VC, Ninfty, NL, NC, NR);


def HRmat(tinfty, tC, tR, Vinfty, VC, VR, Ninfty, NL, NC, NR):
    '''
    Make the TB Hamiltonian for a right side finite quantum well of NR sites
    '''          
    raise Exception
    return Hsysmat(tinfty, tC, tC, tR, Vinfty, VC, VC, VR, Ninfty, NL, NC, NR);

##################################################################################
#### util functions

def plot_wfs(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR, tLprime = None, VLprime = None, tRprime = None, VRprime = None):
    '''
    Visualize the problem by plotting some LL wfs against Hsys
    '''
    if tLprime == None: tLprime = tC;
    if VLprime == None: VLprime = VC;
    if tRprime == None: tRprime = tC;
    if VRprime == None: VRprime = VC;

    # plot
    wffig, wfaxes = plt.subplots(4, sharex = True);

    # plot left well eigenstates
    HL, offset = Hsysmat(tinfty, tL, tC, tRprime, Vinfty, VL, VC, VRprime, Ninfty, NL, NC, NR);
    jvals = np.array(range(len(HL))) + offset;
    wfaxes[0].plot(jvals, np.diag(HL), color=accentcolors[0], linestyle='dashed', linewidth=2*mylinewidth);
    Ems, psims = np.linalg.eigh(HL);
    Ems_bound = Ems[Ems + 2*tL < VC];
    ms_bound = np.linspace(0,len(Ems_bound)-1,3,dtype = int);
    for counter in range(len(ms_bound)):
        m = ms_bound[counter]
        if False: # wfs and energies
            mask = jvals <= NC+NR; 
        else: # just wfs
            mask = jvals <= len(HL); 
        wfaxes[0].plot(jvals[mask], -psims[:,m][mask], color=mycolors[counter]);
        wfaxes[0].plot([NC+NR,jvals[-1]],(2*tL+ Ems[m])*np.ones((2,)), color=mycolors[counter]);
    wfaxes[0].set_ylabel('$\langle j | k_m \\rangle $');
    wfaxes[0].set_ylim(VL-2*VC,VL+2*VC);

    # plot system ham
    if True:
        Hsys, _ = Hsysmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR);
        wfaxes[1].plot(jvals, np.diag(Hsys-HL), color=accentcolors[0], linestyle='dashed', linewidth=2*mylinewidth);
        wfaxes[1].set_ylabel('$H_{sys}-H_L$');

    # plot (Hsys-HL)*psi_m
    if True:
        for counter in range(len(ms_bound)):
            m = ms_bound[counter];
            wfaxes[2].plot(jvals, np.dot(Hsys-HL,psims[:,m]), color = mycolors[counter]);
        wfaxes[2].set_ylabel('$\langle j |(H_{sys}-H_L)| k_m \\rangle $');

    # plot right well eigenstates
    HR, _ = Hsysmat(tinfty, tLprime, tC, tR, Vinfty, VLprime, VC, VR, Ninfty, NL, NC, NR);
    wfaxes[3].plot(jvals, np.diag(HR), color=accentcolors[0], linestyle='dashed', linewidth=2*mylinewidth);
    Emprimes, psimprimes = np.linalg.eigh(HR);
    for counter in range(len(ms_bound)):
        mprime = ms_bound[counter];
        if False: # wfs and energies
            mask = jvals > -NL-NC; 
        else: # just wfs
            mask = jvals <= len(HL); 
        wfaxes[3].plot(jvals[mask], -psimprimes[:,mprime][mask], color=mycolors[counter]);
        wfaxes[3].plot([jvals[0],-NL-NC],(2*tL+ Emprimes[mprime])*np.ones((2,)), color = mycolors[counter]);
    wfaxes[3].set_ylabel("$\langle j |k_{m'} \\rangle $");
    wfaxes[3].set_ylim(VR-2*VC,VR+2*VC);
    for H in [HL,HR]: 
        for jp1 in range(1,len(HL)):
            el = np.diagonal(H,1)[jp1-1]
            if el != -1.0:
                print(el, jp1-1+offset, len(HL));
        
    # format
    wfaxes[-1].set_xlabel('$j$');
    plt.tight_layout();
    plt.show();
    
def TvsE(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR, tLprime = None, VLprime = None, tRprime = None, VRprime = None):
    '''
    Calculate a transmission coefficient for each LL eigenstate and return
    these as a function of their energy
    '''
    if tLprime == None: tLprime = tC;
    if VLprime == None: VLprime = VC;
    if tRprime == None: tRprime = tC;
    if VRprime == None: VRprime = VC;

    # left well eigenstates
    HL, _ = Hsysmat(tinfty, tL, tC, tRprime, Vinfty, VL, VC, VRprime, Ninfty, NL, NC, NR);
    Ems, psims = np.linalg.eigh(HL); # eigenstates of the left well
    Ems = Ems.astype(complex);
    kms = np.arccos((Ems-VL)/(-2*tL)); # wavenumbers in the left well

    # right well eigenstates  
    HR, _ = Hsysmat(tinfty, tLprime, tC, tR, Vinfty, VLprime, VC, VR, Ninfty, NL, NC, NR);
    Emprimes, psimprimes = np.linalg.eigh(HR); # eigenstates of the right well
    Emprimes = Emprimes.astype(complex);
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
        Tms[m] = M*np.conj(M) *NL/(kms[m]*tL) *NR/(kmprimes[mprime]*tR);
        
    return Ems, Tms;

##################################################################################
#### test code

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # fig standardizing
    myxvals = 199;
    myfontsize = 14;
    mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
    accentcolors = ["black","red"];
    mymarkers = ["o","^","s","d","*","X","P"];
    mymarkevery = (40, 40);
    mylinewidth = 1.0;
    mypanels = ["(a)","(b)","(c)","(d)"];
    plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

    # left lead quantum well test
    # tb params, in tL
    mytL = 1.0;
    mytinfty = 1*mytL;
    mytC = 1*mytL;
    mytR = 1*mytL;
    myts = (mytinfty, mytL, mytC, mytR);
    myVinfty = mytL/2;
    myVL = 0.0;
    myVC = mytL/10;
    myVR = 1*myVL;
    myVs = (myVinfty, myVL, myVC, myVR);
    myNinfty = 100;
    myNL = 100;
    myNC = 11;
    myNR = 1*myNL;
    myNs = (myNinfty, myNL, myNC, myNR);

    # visualize the problem
    if False:
        fig, ax = plt.subplots();
        fig.set_size_inches(7/2,3/2);
        HL, offset = Hsysmat(*myts, myVC, myVL, myVC, myVC, *myNs);
        jvals = np.array(range(len(HL))) + offset;
        Hsys, _ = Hsysmat(*myts, myVC, myVL, myVC, myVR, *myNs);
        ax.plot(jvals, np.diag(Hsys-HL), color = accentcolors[0], linestyle="solid", linewidth=2*mylinewidth);
        myEs, mypsis = np.linalg.eigh(HL);
        psi0 = mypsis[:,1];
        ax.plot(jvals,-psi0/5, color = mycolors[0]);

        # format and show
        #ax.set_ylabel('$V_j/t_L$', fontsize=myfontsize);
        #ax.set_xlabel('$j$', fontsize=myfontsize);
        plt.tight_layout();
        plt.axis('off');
        plt.show();
        #plot_wfs(*myts, *myVs, *myNs);

    # matrix elements vs VC
    if False:
        del myVC;
        VCvals = [0.1,0.5];
        numplots = len(VCvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different barrier height
        for VCi in range(len(VCvals)):
            Evals, Tvals = TvsE(*myts, 5*VCvals[VCi], myVL, VCvals[VCi], myVR, *myNs);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            axes[VCi].plot(Evals, Tvals, color = mycolors[0]);
            axes[VCi].set_ylim(0,1.1*max(Tvals));
            axes[VCi].set_ylabel("$M_{m'm}$",fontsize=myfontsize);
            axes[VCi].set_title('$V_C = '+str(VCvals[VCi])+'$', x=0.2, y=0.7,fontsize=myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$', fontsize=myfontsize);
        axes[-1].set_xlim(10**(-3),1)
        plt.tight_layout();
        #plt.show();
        #plt.savefig("figs/bardeen/matrixelements.pdf");
        raise Exception("You have to do this manually");

    # T vs VC
    if False:
        del myVC;
        VCvals = np.array([0.01,0.05,0.1,0.2,0.5,1.0]);
        numplots = len(VCvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different barrier height
        for VCi in range(len(VCvals)):
            Evals, Tvals = TvsE(*myts, 5*VCvals[VCi], myVL, VCvals[VCi], myVR, *myNs);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            Evals, Tvals = Evals[Evals <= VCvals[VCi]], Tvals[Evals <= VCvals[VCi]]; # bound states only
            axes[VCi].scatter(Evals, Tvals, marker=mymarkers[0], color=mycolors[0]);
            axes[VCi].set_ylim(0,1.1*max(Tvals));

            # compare
            kavals = np.arccos((Evals-2*mytL-myVL)/(-2*mytL));
            kappavals = np.arccosh((Evals-2*mytL-VCvals[VCi])/(-2*mytL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*myNC*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[VCi].plot(Evals,np.real(ideal_Tvals), color=accentcolors[0], linewidth=mylinewidth);
            axes[VCi].set_ylabel('$T$',fontsize=myfontsize);
            if VCi==0:
                axes[VCi].set_title('$V_C = '+str(VCvals[VCi])+'$', x=0.7, y = 0.7, fontsize = myfontsize);
            else:
                axes[VCi].set_title('$V_C = '+str(VCvals[VCi])+'$', x=0.2, y = 0.7, fontsize = myfontsize);

            # % error
            axright = axes[VCi].twinx();
            axright.plot(Evals,100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals),color=accentcolors[1]);
            axright.set_ylabel("$\%$ error",fontsize=myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
        plt.tight_layout();
        plt.show();
        #plt.savefig("figs/bardeen/VC.pdf");

    # T vs NC
    if False:
        del myNC;
        NCvals = [11,21,31];
        numplots = len(NCvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different barrier width
        for NCi in range(len(NCvals)):
            Evals, Tvals = TvsE(*myts, *myVs, myNinfty, myNL, NCvals[NCi], myNR);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            Evals, Tvals = Evals[Evals <= myVC], Tvals[Evals <= myVC]; # bound states only
            axes[NCi].scatter(Evals, Tvals, marker=mymarkers[0], color=mycolors[0]);
            axes[NCi].set_ylim(0,1.1*max(Tvals));

            # compare
            kavals = np.arccos((Evals-2*mytL-myVL)/(-2*mytL));
            kappavals = np.arccosh((Evals-2*mytL-myVC)/(-2*mytL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*NCvals[NCi]*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[NCi].plot(Evals,np.real(ideal_Tvals), color=accentcolors[0], linewidth=mylinewidth);
            axes[NCi].set_ylabel('$T$',fontsize=myfontsize);
            axes[NCi].set_title('$N_C = '+str(NCvals[NCi])+'$', x=0.2, y=0.7,fontsize=myfontsize);

            # % error
            axright = axes[NCi].twinx();
            axright.plot(Evals,100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals),color=accentcolors[1]);
            axright.set_ylabel('$\%$ error',fontsize=myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
        plt.tight_layout();
        #plt.show();
        plt.savefig("figs/bardeen/NC.pdf");

    # T vs NR
    if True:
        del myNR;
        NRvals = [100,200];
        numplots = len(NRvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);
        symmetric = True;

        # bardeen results for different well thicknesses
        for NRi in range(len(NRvals)):
            if symmetric:
                fname = "figs/bardeen/NR.pdf";
                Evals, Tvals = TvsE(*myts, *myVs, myNinfty, NRvals[NRi], myNC, NRvals[NRi]);
            else:
                fname = "figs/bardeen/NR_asymmetric";
                Evals, Tvals = TvsE(*myts, *myVs, myNinfty, myNL, myNC, NRvals[NRi]);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            Evals, Tvals = Evals[Evals <= myVC], Tvals[Evals <= myVC]; # bound states only
            axes[NRi].scatter(Evals, Tvals, marker=mymarkers[0], color=mycolors[0]);
            axes[NRi].set_ylim(0,1.1*max(Tvals));

            # compare
            kavals = np.arccos((Evals-2*mytL-myVL)/(-2*mytL));
            kappavals = np.arccosh((Evals-2*mytL-myVC)/(-2*mytL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*myNC*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[NRi].plot(Evals,np.real(ideal_Tvals), color=accentcolors[0], linewidth=mylinewidth);
            axes[NRi].set_ylabel('$T$',fontsize=myfontsize);
            axes[NRi].set_title('$N_R = '+str(NRvals[NRi])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

            # % error
            axright = axes[NRi].twinx();
            axright.plot(Evals,100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals),color=accentcolors[1]);
            axright.set_ylabel('$\%$ error',fontsize=myfontsize);
            axright.set_ylim(0,30);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$K_i / t$',fontsize=myfontsize);
        plt.tight_layout();
        plt.savefig("figs/poster.pdf");

    # T vs VRprime
    if False:
        VRPvals = [0.01,0.1,1.0];
        numplots = len(VRPvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different well thicknesses
        for VRPi in range(len(VRPvals)):
            Evals, Tvals = TvsE(*myts, *myVs, *myNs, VLprime = VRPvals[VRPi], VRprime = VRPvals[VRPi]);
            plot_wfs(*myts, *myVs, *myNs, VLprime = VRPvals[VRPi], VRprime = VRPvals[VRPi]);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            Evals, Tvals = Evals[Evals <= min(myVC, VRPvals[VRPi])], Tvals[Evals <= min(myVC, VRPvals[VRPi])]; # bound states only
            axes[VRPi].scatter(Evals, Tvals, marker=mymarkers[0], color = mycolors[0]);
            axes[VRPi].set_ylim(0,1.1*max(Tvals));

            # compare
            kavals = np.arccos((Evals-2*mytL-myVL)/(-2*mytL));
            kappavals = np.arccosh((Evals-2*mytL-myVC)/(-2*mytL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*myNC*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[VRPi].plot(Evals,np.real(ideal_Tvals), color=accentcolors[0], linewidth=mylinewidth);
            axes[VRPi].set_ylabel('$T$', fontsize=myfontsize);
            axes[VRPi].set_title("$V_R' = "+str(VRPvals[VRPi])+"$", x=0.2, y = 0.7, fontsize=myfontsize);

            # % error
            axright = axes[VRPi].twinx();
            axright.plot(Evals,100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals),color=accentcolors[1]);
            axright.set_ylabel('$\%$ error', fontsize=myfontsize);
            
        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$', fontsize=myfontsize);
        plt.tight_layout();
        plt.show();
        #plt.savefig("figs/bardeen/VRprime.pdf");

    # T vs tRprime
    if False:
        tRPvals = [1.0,0.1,0.01];
        numplots = len(tRPvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results for different well thicknesses
        for tRPi in range(len(tRPvals)):
            #plot_wfs(*myts, *myVs, *myNs, tLprime=2*tRPvals[tRPi], tRprime=tRPvals[tRPi]);
            Evals, Tvals = TvsE(*myts, *myVs, *myNs, tLprime=tRPvals[tRPi], tRprime=tRPvals[tRPi]);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            Evals, Tvals = Evals[Evals <= myVC], Tvals[Evals <= myVC]; # bound states only
            axes[tRPi].scatter(Evals, Tvals, marker=mymarkers[0], color=mycolors[0]);
            axes[tRPi].set_ylim(0,1.1*max(Tvals));

            # compare
            kavals = np.arccos((Evals-2*mytL-myVL)/(-2*mytL));
            kappavals = np.arccosh((Evals-2*mytL-myVC)/(-2*mytL));
            ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
            ideal_exp = np.exp(-2*myNC*kappavals);
            ideal_Tvals = ideal_prefactor*ideal_exp;
            ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
            ideal_Tvals *= ideal_correction;
            axes[tRPi].plot(Evals,np.real(ideal_Tvals), color=accentcolors[0], linewidth=mylinewidth);
            axes[tRPi].set_ylabel('$T$', fontsize=myfontsize);
            axes[tRPi].set_title("$t_R' = "+str(tRPvals[tRPi])+"$", x=0.2, y = 0.7, fontsize=myfontsize);

            # % error
            axright = axes[tRPi].twinx();
            axright.plot(Evals,100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals),color=accentcolors[1]);
            axright.set_ylabel('$\%$ error');
            
        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$', fontsize=myfontsize);
        plt.tight_layout();
        #plt.show();
        plt.savefig("figs/bardeen/tRprime.pdf");

    





    
    


    








