'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

from transport import fci_mod

import numpy as np

##################################################################################
#### driver of transmission coefficient calculations

def kernel(tinfty, tL, tLprime, tR, tRprime, Vinfty, VL, VLprime, VR, VRprime, Ninfty, NL, NR, HC,HCprime,cutoff=1.0,verbose=0):
    '''
    Calculate a transmission coefficient for each left well state as
    a function of energy
    '''
    from transport import wfm
    if(np.shape(HC) != np.shape(HCprime)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];

    # convert from matrices to _{alpha alpha} elements
    to_convert = [tL, VL, tR, VR];
    converted = [];
    for convert in to_convert:
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ):
            raise ValueError; # VL must be diag
        converted.append(np.diagonal(convert));
    tLa, VLa, tRa, VRa = tuple(converted);

    # left well eigenstates
    HL, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HL),n_loc_dof));
    Emas, psimas = [], []; # will index as Emas[alpha,m]
    n_ms = 0;
    for alpha in range(n_loc_dof):
        Ems, psims = np.linalg.eigh(HL[:,:,alpha,alpha]);
        psims = psims.T[Ems+2*tLa[alpha] < cutoff];
        Ems = Ems[Ems+2*tLa[alpha] < cutoff];
        Emas.append(Ems);
        psimas.append(psims);
        n_ms = max(n_ms, len(Emas[alpha]));
    Emas_arr = np.empty((n_loc_dof,n_ms), dtype = complex); # make un-ragged
    psimas_arr = np.empty((n_loc_dof,n_ms,n_spatial_dof), dtype = complex);
    for alpha in range(n_loc_dof):# un-ragged the array by filling in highest Es
        Ems = Emas[alpha];
        Ems_arr = np.append(Ems, np.full((n_ms-len(Ems),), Ems[-1]));
        Emas_arr[alpha] = Ems_arr;
        psims = psimas[alpha];
        psims_arr = np.append(psims, np.full((n_ms-len(Ems),n_spatial_dof), psims[-1]),axis=0);
        psimas_arr[alpha] = psims_arr;
    Emas, psimas = Emas_arr, psimas_arr # shape is (n_loc_dof, n_ms)
    kmas = np.arccos((Emas-fci_mod.scal_to_vec(VLa,n_ms))
                    /(-2*fci_mod.scal_to_vec(tLa,n_ms))); # wavenumbers in the left well
    
    # right well eigenstates  
    HR, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HR),n_loc_dof));
    Enbs, psinbs = [], []; # will index as Enbs[beta,n]
    n_ns = 0;
    for beta in range(n_loc_dof):
        Ens, psins = np.linalg.eigh(HR[:,:,beta,beta]);
        psins = psins.T[Ens+2*tRa[alpha] < cutoff];
        Ens = Ens[Ens+2*tRa[alpha] < cutoff];
        Enbs.append(Ens.astype(complex));
        psinbs.append(psins);
        n_ns = max(n_ns, len(Ens));
    assert(n_ms == n_ns);
    Enbs_arr = np.empty((n_loc_dof,n_ns), dtype = complex); # make un-ragged
    psinbs_arr = np.empty((n_loc_dof,n_ns,n_spatial_dof), dtype = complex);
    for alpha in range(n_loc_dof):# un-ragged the array by filling in highest Es
        Ens = Enbs[alpha];
        Ens_arr = np.append(Ens, np.full((n_ns-len(Ens),), Ens[-1]));
        Enbs_arr[alpha] = Ens_arr;
        psins = psinbs[alpha];
        psins_arr = np.append(psins, np.full((n_ns-len(Ens),n_spatial_dof), psins[-1]),axis=0);
        psinbs_arr[alpha] = psins_arr
    Enbs, psinbs = Enbs_arr, psinbs_arr # shape is (n_loc_dof, n_ms)
    knbs = np.arccos((Enbs-fci_mod.scal_to_vec(VRa,n_ns))
                    /(-2*fci_mod.scal_to_vec(tRa,n_ns))); # wavenumbers in the left well

    # operator
    Hsys, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    Hdiff = fci_mod.mat_4d_to_2d(Hsys - HL);

    # visualize
    if(verbose > 4):
        import matplotlib.pyplot as plt
        jvals = np.array(range(len(Hsys))) + offset;
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL,HR,Hsys];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(jvals, Hi*0.01+np.diag(Hs[Hi][:,:,alpha,alpha]));
        plt.show();
        assert False;

    # compute matrix elements
    T_nb_mas = np.empty((n_loc_dof,n_ns,n_loc_dof,n_ms),dtype=float);
    for alpha in range(n_loc_dof):
        for m in range(n_ms):
            for beta in range(n_loc_dof):
                for n in range(n_ns):
                    melement = matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]);
                    T_nb_mas[beta,n,alpha,m] = np.real(melement*np.conj(melement)* NL/(kmas[alpha,m]*tLa[alpha]) *NR/(kmas[alpha,m]*tRa[alpha]));

            '''
            # average matrix elements over final states |k_n \beta>
            # with the same energy as the intial state |k_m \alpha>
            Mma = 0.0;
            for beta in range(n_loc_dof):
                for n in range(n_ns):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < 1e-9): # equal energy
                        Mma += np.real(M_nb_mas[beta,n,alpha,m]*np.conj(M_nb_mas[beta,n,alpha,m]));

            # update T based on average
            Tmas[alpha,m] = NL/(kmas[alpha,m]*tLa[alpha]) *NR/(kmas[alpha,m]*tRa[alpha]) *Mma;
            '''
    return Emas, T_nb_mas;

def Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC):
    '''
    Make the TB Hamiltonian for the full system, general 1D case
    '''
    for arg in [tinfty, tL, tR, Vinfty, VL, VR]:
        if(type(arg) != np.ndarray): raise TypeError;
    for N in [Ninfty, NL, NR]:
        if(not isinstance(N, int)): raise TypeError;
        if(N <= 0): raise ValueError;
    if(np.shape(HC[0,0]) != np.shape(tinfty)): raise ValueError;
    if(len(HC) % 2 != 1): raise ValueError; # NC must be odd
    littleNC = len(HC) // 2;
    minusinfty = -littleNC - NL - Ninfty;
    plusinfty = littleNC + NR + Ninfty;
    nsites = -minusinfty + plusinfty + 1;
    n_loc_dof = np.shape(tinfty)[0];

    # Hamiltonian matrix
    Hmat = np.zeros((nsites,nsites,n_loc_dof,n_loc_dof),dtype=complex);
    for j in range(minusinfty, plusinfty+1):

        # diag outside HC
        if(j < -NL - littleNC):           
            Hmat[j-minusinfty,j-minusinfty] += Vinfty
        elif(j >= -NL-littleNC and j < -littleNC):
            Hmat[j-minusinfty,j-minusinfty] += VL;
        elif(j > littleNC and j <= littleNC+NR):
            Hmat[j-minusinfty,j-minusinfty] += VR;
        elif(j > littleNC+NR):
            Hmat[j-minusinfty,j-minusinfty] += Vinfty;

        # off diag outside HC
        if(j < -NL - littleNC):           
            Hmat[j-minusinfty,j+1-minusinfty] += -tinfty;
            Hmat[j+1-minusinfty,j-minusinfty] += -tinfty;
        elif(j >= -NL-littleNC and j < -littleNC):
            Hmat[j-minusinfty,j+1-minusinfty] += -tL;
            Hmat[j+1-minusinfty,j-minusinfty] += -tL;
        elif(j > littleNC and j <= littleNC+NR):
            Hmat[j-minusinfty,j-1-minusinfty] += -tR;
            Hmat[j-1-minusinfty,j-minusinfty] += -tR; 
        elif(j > littleNC+NR):
            Hmat[j-minusinfty,j-1-minusinfty] += -tinfty;
            Hmat[j-1-minusinfty,j-minusinfty] += -tinfty;

    # HC
    Hmat[-littleNC-minusinfty:littleNC+1-minusinfty,-littleNC-minusinfty:littleNC+1-minusinfty] = HC;
            
    return Hmat, minusinfty;

def Hwellmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR):
    '''
    Make the TB Hamiltonian for the full system, 1D well case
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

##################################################################################
#### utils

def is_alpha_conserving(T,n_loc_dof):
    '''
    Determines if a tensor T conserves alpha in the sense that it has
    only nonzero elements for a certain value of alpha
    '''
    if( type(T) != np.ndarray): raise TypeError;

    shape = np.shape(T);
    indices = np.array(range(*shape));
    if len(shape) == 1: # is a vector
        alphas = np.full(n_loc_dof, 1, dtype = int);
        for ai in range(n_loc_dof):
            alphas[ai] = np.any(T[indices % n_loc_dof == ai]);
        return (sum(alphas) == 1 or sum(alphas) == 0);

    elif len(shape) == 2: #matrix
        for i in range(shape[0]):
            for j in range(shape[1]):
                if(abs(T[i,j]) > 1e-9):
                    if(i % n_loc_dof != j % n_loc_dof):
                        return False;
        return True;

    else: raise Exception; # not supported

def matrix_element(beta,psin,op,alpha,psim):
    '''
    Take the matrix element of a (not in general alpha conserving), spin separated
    (2d) operator between spin conserving states
    '''
    from transport import wfm
    if(len(np.shape(op))!=2): raise ValueError; # op should be flattened
    n_loc_dof = np.shape(psim)[0];
    n_spatial_dof = np.shape(psim)[1]
    n_ov_dof = len(op);
    if(n_ov_dof % n_spatial_dof != 0): raise ValueError;
    if(n_ov_dof // n_spatial_dof != n_loc_dof): raise ValueError;

    # flatten psis's
    psimalpha = np.zeros_like(psim);
    psimalpha[alpha] = psim[alpha]; # all zeros except for psi[alphas]
    psimalpha = fci_mod.vec_2d_to_1d(psimalpha.T); # flatten
    assert(is_alpha_conserving(psimalpha,n_loc_dof));
    psinbeta = np.zeros_like(psin);
    psinbeta[beta] = psin[beta] # all zeros except for psi[beta]
    psinbeta = fci_mod.vec_2d_to_1d(psinbeta.T); # flatten
    assert(is_alpha_conserving(psinbeta,n_loc_dof));
    return np.dot(psinbeta, np.dot(op,psimalpha));

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
    HL, _ = Hwellmat(tinfty, tL, tC, tRprime, Vinfty, VL, VC, VRprime, Ninfty, NL, NC, NR);
    Ems, psims = np.linalg.eigh(HL); # eigenstates of the left well
    Ems = Ems.astype(complex);
    kms = np.arccos((Ems-VL)/(-2*tL)); # wavenumbers in the left well

    # right well eigenstates  
    HR, _ = Hwellmat(tinfty, tLprime, tC, tR, Vinfty, VLprime, VC, VR, Ninfty, NL, NC, NR);
    Emprimes, psimprimes = np.linalg.eigh(HR); # eigenstates of the right well
    #Emprimes = Emprimes.astype(complex);
    #kmprimes = np.arccos((Emprimes-VR)/(-2*tR)); # wavenumbers in the right well

    # operator
    Hsys, offset = Hwellmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR);
    op = Hsys - HL;

    # visualize
    if False:
        import matplotlib.pyplot as plt
        jvals = np.array(range(len(Hsys))) + offset;
        myfig,myaxes = plt.subplots(1,sharex=True);
        myaxes = [myaxes];
        Hs = [HL,HR,Hsys];
        for Hi in range(len(Hs)):
            myaxes[0].plot(jvals, Hi*0.01+np.diag(Hs[Hi][:,:]));
        plt.show();
        assert False;

    # compute T
    Tms = np.zeros_like(Ems);
    for m in range(len(Ems)):
        mprime = m;
        M = np.dot(psimprimes[:,mprime],np.dot(op,psims[:,m]));
        Tms[m] = M*np.conj(M) *NL/(kms[m]*tL) *NR/(kms[m]*tR);
        
    return Ems, Tms;

##################################################################################
#### test code

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # fig standardizing
    myxvals = 199;
    myfontsize = 14;
    mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
    accentcolors = ["black","red"];
    mymarkers = ["o","^","s","d","*","X","P"];
    mymarkevery = (40, 40);
    mylinewidth = 1.0;
    mypanels = ["(a)","(b)","(c)","(d)"];
    #plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

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
    myNL = 500;
    myNC = 11;
    myNR = 1*myNL;
    myNs = (myNinfty, myNL, myNC, myNR);

    # visualize the problem
    if False:
        fig, ax = plt.subplots();
        Hsys, offset = Hsysmat(*ts, Vinfty, VL, VC, 0.5*VC, *Ns);
        jvals = np.array(range(len(Hsys))) + offset;
        ax.plot(jvals, np.diag(Hsys), color = accentcolors[0], linestyle='dashed', linewidth=2*mylinewidth);
        ax.set_ylabel('$V_j/t_L$', fontsize=myfontsize);
        ax.set_xlabel('$j$', fontsize=myfontsize);
        plt.tight_layout();
        plt.show();
        plot_wfs(*ts, *Vs, *Ns);
    
    # T vs VRprime
    if False:
        VRPvals = [0.1,1.0,10.0];
        numplots = len(VRPvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);
        maxerror = 0;
        axrights = [];

        # bardeen results for different well thicknesses
        for VRPi in range(len(VRPvals)):
            Evals, Tvals = TvsE(*myts, *myVs, *myNs, VLprime = VRPvals[VRPi], VRprime = VRPvals[VRPi]);
            #plot_wfs(*myts, *myVs, *myNs, VLprime = VRPvals[VRPi], VRprime = VRPvals[VRPi]);
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
            axrights.append(axright);
            errorvals = 100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals);
            maxerror = np.max((maxerror,np.max(errorvals)));
            axright.plot(Evals,errorvals,color=accentcolors[1]);
            axright.set_ylabel('$\%$ error', fontsize=myfontsize);
            
        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$', fontsize=myfontsize);
        maxerror = 10
        for axright in axrights: axright.set_ylim(0,maxerror);
        plt.tight_layout();
        plt.savefig("figs/bardeen_benchmark/VRprime.png");

    # T vs tRprime
    if False:
        tRPvals = [1.0,0.1,0.01];
        numplots = len(tRPvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);
        maxerror = 0;
        axrights = [];

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
            axrights.append(axright);
            errorvals = 100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals);
            maxerror = np.max((maxerror,np.max(errorvals)));
            axright.plot(Evals,errorvals,color=accentcolors[1]);
            axright.set_ylabel('$\%$ error', fontsize=myfontsize);
            
        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$', fontsize=myfontsize);
        for axright in axrights: axright.set_ylim(0,maxerror);
        plt.tight_layout();
        #plt.show();
        plt.savefig("figs/bardeen_benchmark/tRprime.png");

    # T vs NR
    if True:
        del myNR;
        NRvals = [50,100];
        numplots = len(NRvals);
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);
        maxerror = 0;
        axrights = [];

        # bardeen results for different well widths
        for NRi in range(len(NRvals)):
            Evals, Tvals = TvsE(*myts, *myVs, myNinfty, NRvals[NRi], myNC, NRvals[NRi], VLprime=myVinfty,VRprime=myVinfty);
            Evals = np.real(Evals+2*mytL);
            Tvals = np.real(Tvals);
            Evals, Tvals = Evals[Evals <= myVC], Tvals[Evals <= myVC]; # bound states only
            axes[NRi].scatter(Evals, Tvals, marker=mymarkers[0], color=mycolors[0]);
            axes[NRi].set_ylim(0,1.1*max(Tvals));

            # compare
            kavals = np.arccos((Evals-2*mytL-myVL)/(-2*mytL));
            kappavals = np.arccosh((Evals-2*mytC-myVC)/(-2*mytC));
            print("Evals:\n",Evals[:8]);
            print("kavals:\n",kavals[:8]);
            print("kappavals:\n",kappavals[:8]);
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
            axrights.append(axright);
            errorvals = 100*abs((Tvals-np.real(ideal_Tvals))/ideal_Tvals);
            maxerror = np.max((maxerror,np.max(errorvals)));
            axright.plot(Evals,errorvals,color=accentcolors[1]);
            axright.set_ylabel('$\%$ error', fontsize=myfontsize);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
        for axright in axrights: axright.set_ylim(0,20);
        plt.tight_layout();
        plt.show();
        #fname = "figs/bardeen_benchmark/NR.png";
        #plt.savefig(fname);

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




    
    


    








