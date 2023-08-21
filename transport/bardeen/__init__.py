'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

from transport import fci_mod, wfm

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### driver of transmission coefficient calculations

def kernel(Hsys_4d, tbulk, cutiL, cutiR, interval=1e-9, E_cutoff=1.0, verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nbma averaged over final energy
    states n in aninterval close to the initial energy state m

    Instead of setting up Hsys explicitly as below, takes any TB Hsys
    and generates HL (HR) by cutting the hopping at site cutiL (cutiR)
    '''
    if( not isinstance(Hsys_4d, np.ndarray)): raise TypeError;
    n_spatial_dof, _, n_loc_dof, _ = np.shape(Hsys_4d);
    mid = n_spatial_dof // 2;
    if(cutiL >= n_spatial_dof-1 or cutiR >= n_spatial_dof-1): raise ValueError;

    # generate HL and HR
    HL_4d = np.copy(Hsys_4d);
    HL_4d[cutiL-1,cutiL] = np.zeros_like(HL_4d[cutiL-1,cutiL]);
    HL_4d[cutiL,cutiL-1] = np.zeros_like(HL_4d[cutiL,cutiL-1]);
    HR_4d = np.copy(Hsys_4d);
    HR_4d[cutiR-1,cutiR] = np.zeros_like(HR_4d[cutiR-1,cutiR]);
    HR_4d[cutiR,cutiR-1] = np.zeros_like(HR_4d[cutiR,cutiR-1]);   
    if(verbose):
        print("HC = \n"+str(Hsys_4d[mid,mid]))
        print("Hsys = "+str(np.shape(Hsys_4d))+"\n",Hsys_4d[mid-2:mid+2,mid-2:mid+2,0,0]);
        print("HL = "+str(np.shape(HL_4d))+"\n",HL_4d[mid-2:mid+2,mid-2:mid+2,0,0]);
        print("HR = "+str(np.shape(HL_4d))+"\n",HR_4d[mid-2:mid+2,mid-2:mid+2,0,0]);

    # eigenstates of HL
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HL_4d),n_loc_dof));
    Emas, psimas = [], []; # will index as Emas[alpha,m]
    n_bound_left = 0;
    for alpha in range(n_loc_dof):
        Ems, psims = np.linalg.eigh(HL_4d[:,:,alpha,alpha]);
        psims = psims.T[Ems+2*tbulk < tbulk*E_cutoff];
        Ems = Ems[Ems+2*tbulk < tbulk*E_cutoff];
        Emas.append(Ems);
        psimas.append(psims);
        n_bound_left = max(n_bound_left, len(Emas[alpha]));
    Emas_arr = np.empty((n_loc_dof,n_bound_left), dtype = complex); # make un-ragged
    psimas_arr = np.empty((n_loc_dof,n_bound_left,n_spatial_dof), dtype = complex);
    for alpha in range(n_loc_dof):# un-ragged the array by filling in highest Es
        Ems = Emas[alpha];
        Ems_arr = np.append(Ems, np.full((n_bound_left-len(Ems),), Ems[-1]));
        Emas_arr[alpha] = Ems_arr;
        psims = psimas[alpha];
        psims_arr = np.append(psims, np.full((n_bound_left-len(Ems),n_spatial_dof), psims[-1]),axis=0);
        psimas_arr[alpha] = psims_arr;
    del Ems, psims, Emas, psimas;
    #Emas_arr, psimas_arr # shape is (n_loc_dof, n_bound_left);

    # flag initial vs final states
    mid = len(Hsys_4d)//2;
    flagsL = np.zeros_like(Emas_arr,dtype=int)
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):
            psim = psimas_arr[alpha,m];
            weight_left = np.dot( np.conj(psim[:mid]), psim[:mid]);
            weight_right = np.dot( np.conj(psim[mid:]), psim[mid:]);
            if(weight_left > weight_right): # this is an initial state
                flagsL[alpha,m] = 1;
    Emas = np.where(flagsL==1,Emas_arr,np.nan);

    # eigenstates of HR
    if False:
        raise NotImplementedError
        del alpha
        assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HR_4d),n_loc_dof));
        Enbs, psinbs = [], []; # will index as Emas[alpha,m]
        n_bound_right = 0;
        for beta in range(n_loc_dof):
            Ens, psins = np.linalg.eigh(HR_4d[:,:,beta,beta]);
            psins = psins.T[Ens+2*tbulk < tbulk*E_cutoff];
            Ens = Ens[Ens+2*tbulk < tbulk*E_cutoff];
            Enbs.append(Ens);
            psinbs.append(psins);
            n_bound_right = max(n_bound_right, len(Emas[beta]));
        Enbs_arr = np.empty((n_loc_dof,n_bound_right), dtype = complex); # make un-ragged
        psinbs_arr = np.empty((n_loc_dof,n_bound_right,n_spatial_dof), dtype = complex);
        for beta in range(n_loc_dof):# un-ragged the array by filling in highest Es
            Ens = Enbs[beta];
            Ens_arr = np.append(Ens, np.full((n_bound_right-len(Ens),), Ens[-1]));
            Enbs_arr[beta] = Ens_arr;
            psins = psinbs[beta];
            psins_arr = np.append(psins, np.full((n_bound_right-len(Ens),n_spatial_dof), psins[-1]),axis=0);
            psinbs_arr[beta] = psins_arr;
        del Ens, psins, Enbs, psinbs;
        #Enbs_arr, psinbs_arr # shape is (n_loc_dof, n_bound_left);

        # flag initial vs final states
        mid = len(Hsys_4d)//2;
        flagsR = np.zeros_like(Enbs_arr,dtype=int)
        for beta in range(n_loc_dof):
            for n in range(n_bound_left):
                psin = psinbs_arr[beta,n];
                weight_left = np.dot( np.conj(psin[:mid]), psin[:mid]);
                weight_right = np.dot( np.conj(psin[mid:]), psin[mid:]);
                if(weight_left > weight_right): # this is an initial state
                    flagsR[beta,n] = 1;
        Enbs = np.where(flagsR==1,Enbs_arr,np.nan);

    else:
        Enbs = np.where(flagsL==0,Emas_arr,np.nan);
                    
    # visualize
    jvals = np.linspace(-mid, -mid +len(Hsys_4d)-1,len(Hsys_4d), dtype=int);
    if(verbose > 9):

        # energies
        print("Emas_arr "+str(np.shape(Emas_arr))+"\n",Emas_arr/tbulk);
        print("Emas "+str(np.shape(Emas))+"\n",Emas/tbulk);
        print("Enbs "+str(np.shape(Enbs))+"\n",Enbs/tbulk);

        # plot hams
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [Hsys_4d];
            Hstrs = ["$H_{sys}$","$H_{sys}-H_L$","$H_{sys}-H_{R}$"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(jvals, Hi*0.001+np.diag(Hs[Hi][:,:,alpha,alpha]),label = Hstrs[Hi]);
            myaxes[alpha].set_xlabel("$j$"); myaxes[alpha].set_ylabel("$V_j$");
        plt.legend();plt.show();

        # plot left wfs
        for m in range(6): #n_bound_left):
            wffig, wfax = plt.subplots();
            alpha_colors=["tab:blue","tab:orange"];
            for alpha in range(n_loc_dof):
                if(not np.isnan(Emas[alpha,m])):
                    wfax.set_title("$"+str(np.real(Emas[alpha,m].round(4)))+" \\rightarrow "+str(np.real(Enbs[alpha,m-1].round(4)))+"$");
                    wfax.plot(jvals,np.diag(HL_4d[:,:,alpha,alpha]),color="black");
                    wfax.plot(jvals[:-1], np.diagonal(HL_4d[:,:,alpha,alpha], offset=1), color="black", linestyle="dotted")
                    wfax.plot(jvals, np.real(psimas_arr[alpha,m]),color=alpha_colors[alpha],linestyle="solid");
                    wfax.plot(jvals, np.real(psimas_arr[alpha,m-1]),color=alpha_colors[alpha],linestyle="dotted");
            plt.show();
        assert False;

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):               
            for beta in range(n_loc_dof):
                # inelastic means averaging over an interval
                Mns = [];
                for n in range(n_bound_left):
                    if((not np.isnan(Emas[alpha,m])) and abs(Emas[alpha,m] - Enbs[beta,n]) < interval/2):
                        melement = matrix_element(beta,psimas_arr[:,n],Hdiff,alpha,psimas_arr[:,m]);
                        Mns.append(np.real(melement*np.conj(melement)));

                # update T based on average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;

    return Emas, Mbmas;

def kernel_well(tinfty, tL, tR,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC,HCprime,E_cutoff,
           interval=1e-9,verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nbma averaged over final energy
    states n in aninterval close to the initial energy state m

    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR

    This kernel requires the initial and final states to have definite spin,
    and so CAN RESOLVE the spin -> spin transitions

    Optional args:
    -interval, float, rectangle func energy window, corresponding to 2\pi\hbar/t
    -E_cutoff, float, don't calculate for m, n with energy higher 
        than this. That way we limit to bound states
    '''
    if(np.shape(HC) != np.shape(HCprime)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];

    # convert from matrices to spin-diagonal, spin-independent elements
    to_convert = [tL, tR];
    converted = [];
    for convert in to_convert:
        # check spin-diagonal
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ): raise ValueError("not spin diagonal"); 
        # check spin-independent
        diag = np.diagonal(convert);
        if(np.any(diag-diag[0])): raise ValueError("not spin independent");
        converted.append(convert[0,0]);
    tLa, tRa = tuple(converted);

    # left well eigenstates
    HL_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HL_4d),n_loc_dof));
    Emas, psimas = [], []; # will index as Emas[alpha,m]
    n_bound_left = 0;
    for alpha in range(n_loc_dof):
        Ems, psims = np.linalg.eigh(HL_4d[:,:,alpha,alpha]);
        psims = psims.T[Ems+2*tLa < E_cutoff[alpha,alpha]];
        Ems = Ems[Ems+2*tLa < E_cutoff[alpha,alpha]];
        Emas.append(Ems);
        psimas.append(psims);
        n_bound_left = max(n_bound_left, len(Emas[alpha]));
    Emas_arr = np.empty((n_loc_dof,n_bound_left), dtype = complex); # make un-ragged
    psimas_arr = np.empty((n_loc_dof,n_bound_left,n_spatial_dof), dtype = complex);
    for alpha in range(n_loc_dof):# un-ragged the array by filling in highest Es
        Ems = Emas[alpha];
        Ems_arr = np.append(Ems, np.full((n_bound_left-len(Ems),), Ems[-1]));
        Emas_arr[alpha] = Ems_arr;
        psims = psimas[alpha];
        psims_arr = np.append(psims, np.full((n_bound_left-len(Ems),n_spatial_dof), psims[-1]),axis=0);
        psimas_arr[alpha] = psims_arr;
    print("Ems = ",Ems);
    print("Emas = ",Emas);
    print("Emas_arr = ",Emas_arr);
    assert False
    del Ems, psims, alpha
    Emas, psimas = Emas_arr, psimas_arr # shape is (n_loc_dof, n_bound_left)

    # right well eigenstates  
    HR_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HR_4d),n_loc_dof));
    Enbs, psinbs = [], []; # will index as Enbs[beta,n]
    n_bound_right = 0;
    for beta in range(n_loc_dof):
        Ens, psins = np.linalg.eigh(HR_4d[:,:,beta,beta]);
        psins = psins.T[Ens+2*tRa < E_cutoff[beta,beta]];
        Ens = Ens[Ens+2*tRa < E_cutoff[beta,beta]];
        Enbs.append(Ens.astype(complex));
        psinbs.append(psins);
        n_bound_right = max(n_bound_right, len(Ens));
    Enbs_arr = np.empty((n_loc_dof,n_bound_right), dtype = complex); # make un-ragged
    psinbs_arr = np.empty((n_loc_dof,n_bound_right,n_spatial_dof), dtype = complex);
    for beta in range(n_loc_dof):# un-ragged the array by filling in highest Es
        Ens = Enbs[beta];
        Ens_arr = np.append(Ens, np.full((n_bound_right-len(Ens),), Ens[-1]));
        Enbs_arr[beta] = Ens_arr;
        psins = psinbs[beta];
        psins_arr = np.append(psins, np.full((n_bound_right-len(Ens),n_spatial_dof), psins[-1]),axis=0);
        psinbs_arr[beta] = psins_arr;
    del Ens, psins;
    Enbs, psinbs = Enbs_arr, psinbs_arr # shape is (n_loc_dof, n_bound_right)

    # operator
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    jvals = np.array(range(len(Hsys_4d))) + offset;
    mid = len(jvals) // 2;
    if(verbose > 9): # plot hams
        print("np.shape(Emas) = ",np.shape(Emas));
        print("np.shape(Enbs) = ",np.shape(Enbs));
        for alpha in range(n_loc_dof):
            plot_ham(jvals, (Hsys_4d,HL_4d,Hsys_4d-HL_4d), alpha );
        raise NotImplementedError;

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    overlaps = np.empty_like(Emas,dtype=float); # <<<<<
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):               
            for beta in range(n_loc_dof):
                # inelastic means averaging over an interval
                Mns = [];
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval/2):
                        melement = matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]);
                        Mns.append(np.real(melement*np.conj(melement)));
                        overlaps[alpha,m] = np.real( np.conj(np.dot(np.conj(psinbs[beta,n]), psimas[alpha,m]))
                                                     *np.dot(np.conj(psinbs[beta,n]), psimas[alpha,m]));

                # update M with average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;
                
    return Emas, Mbmas;

def kernel_well_super(tinfty, tL, tR,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC,HCprime, alpha_mat, E_cutoff,
           interval=1e-9,eigval_tol=1e-9,verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nm averaged over n in a
    nearby interval
    
    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR
    
    For this kernel, the initial and final states are superpositions of
    the eigenstates of HL and HR. If the latter are in the basis |\alpha>,
    then the former are in the basis |\tilde{\alpha} >
    the alpha basis is chosen by the spin matrix alpha_mat
    the variable change_basis gives the basis transformation according to
    |\tilde{\alpha} > = \sum_\alpha change_basis[\alpha, |tilde{\alpha} ] |\alpha>

    # DO NOT combine with kernel_well b/c that controls "old way"
    
    Optional args:
    -E_cutoff, float, don't calculate T for eigenstates with energy higher 
        than this. That way we limit to bound states
    '''
    if(np.shape(HC) != np.shape(HCprime)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];
    E_cutoff_first = np.max(E_cutoff);
    print("E_cutoff, E_cutoff_first",E_cutoff, E_cutoff_first);

    # convert from matrices to spin-diagonal, spin-independent elements
    to_convert = [tL, tR];
    converted = [];
    for convert in to_convert:
        # check spin-diagonal
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ): raise ValueError("not spin diagonal"); 
        # check spin-independent
        diag = np.diagonal(convert);
        if(np.any(diag-diag[0])): raise ValueError("not spin independent");
        converted.append(convert[0,0]);
    tLa, tRa = tuple(converted);

                    ####
                    ####
    print(alpha_mat);
    #alpha_mat = alpha_mat - np.diagflat(np.diagonal(alpha_mat));
    #print(alpha_mat);

    # change of basis
    # alpha basis is eigenstates of alpha_mat
    alpha_eigvals_exact, alphastates = np.linalg.eigh(alpha_mat);
    alphastates = alphastates.T;
    tildestates = np.eye(n_loc_dof);
    # get coefs st \tilde{\alpha}\sum_alpha coefs[\alpha,\tilde{\alpha}] \alpha
    change_basis = np.empty_like(alphastates);
    for astatei in range(len(alphastates)):
        for tstatei in range(len(tildestates)):
            change_basis[astatei, tstatei] = np.dot( np.conj(alphastates[astatei]), tildestates[tstatei]);
    if False: # for checking change of basis
        print("\nalphastates = ",alphastates,"\ntildestates = ", tildestates,"\nchange_basis = ", change_basis);  
        raise NotImplementedError;

    # left well 
    HL_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    HL = fci_mod.mat_4d_to_2d(HL_4d);

    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);  
    jvals = np.array(range(len(Hsys_4d))) + offset;
    if(verbose > 9): # plot hams
        for alpha in range(n_loc_dof):
            plot_ham(jvals, (Hsys_4d,HL_4d,Hsys_4d-HL_4d), alpha );
        for h in (Hsys_4d,HL_4d):
            h_aa = h[:,:,0,0] ;
            h_bb = h[:,:,1,1] - np.diagflat(np.diagonal(0.087*np.ones_like(h_aa)));
            assert( not np.any(h_aa-h_bb));
        raise NotImplementedError;
    
    # left well eigenstates
    Ems, psims = np.linalg.eigh(HL);
    psims = psims.T[Ems+2*tLa < E_cutoff_first];
    Ems = Ems[Ems+2*tLa < E_cutoff_first].astype(complex);
    if(len(Ems) % n_loc_dof != 0): Ems, psims = Ems[:-1], psims[:-1]; # must be even

    # measure alpha val for each k_m
    alpha_mat_4d = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof),dtype=complex);
    for sitej in range(n_spatial_dof):
        alpha_mat_4d[sitej,sitej] = np.copy(alpha_mat);
    alpha_mat_2d = fci_mod.mat_4d_to_2d(alpha_mat_4d);
    alphams = np.empty((len(Ems),),dtype=complex);
    for m in range(len(Ems)):
        alphams[m] = np.dot( np.conj(psims[m]), np.matmul(alpha_mat_2d, psims[m]));
        if(verbose>5): print(m, Ems[m], alphams[m]);

    # get all unique alpha vals, should be exactly n_loc_dof of them
    alpha_eigvals = dict();
    for alpha in alphams:
        addin = True;
        for k in alpha_eigvals.keys():
            if(abs(alpha-k) < eigval_tol):
                alpha_eigvals[k] += 1;
                addin = False;
        if(addin):
            alpha_eigvals[alpha] = 1;
    print(">>>\nalpha_eigvals =\n",alpha_eigvals,"\nalpha_eigvals_exact =\n",{alpha_eigvals_exact[0]:0,alpha_eigvals_exact[1]:0});
    if(len(alpha_eigvals.keys()) != n_loc_dof): print(alpha_eigvals); raise Exception("alpha vals");
    n_bound_left = np.min(list(alpha_eigvals.values()));  print("n_bound_left = ", n_bound_left); # truncate
    alpha_eigvals = list(alpha_eigvals.keys());

    # classify left well eigenstates in the \alpha basis
    Emas = [];
    psimas = [];
    for eigvali in range(len(alpha_eigvals)):
        Es_this_a, psis_this_a = [], [];
        for m in range(len(Ems)):
            if(abs(np.real(alphams[m] - alpha_eigvals[eigvali])) < eigval_tol):
                Es_this_a.append(Ems[m]); psis_this_a.append(psims[m]);
        Emas.append(Es_this_a); psimas.append(psis_this_a);
    print("Emas[0] = ",np.real(Emas[0]));
    print("Emas[1] = ",np.real(Emas[1]));

    # classify again with cutoff
    Emas_arr = np.empty((n_loc_dof,n_bound_left),dtype=complex);
    psimas_arr = np.empty((n_loc_dof,n_bound_left,len(psims[0])),dtype=complex);
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):
            if(Emas[alpha][m] + 2*tLa < E_cutoff[alpha,alpha]):
                Emas_arr[alpha,m] = Emas[alpha][m];
                psimas_arr[alpha,m] = psimas[alpha][m];
    Emas, psimas = Emas_arr, psimas_arr;
    del Ems, psims;
    print("Emas[0] = ",np.real(Emas[0]));
    print("Emas[1] = ",np.real(Emas[1]));

    # right well 
    HR_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    HR = fci_mod.mat_4d_to_2d(HR_4d);

    # right well eigenstates
    Ens, psins = np.linalg.eigh(HR);
    psins = psins.T[Ens+2*tRa < E_cutoff_first];
    Ens = Ens[Ens+2*tRa < E_cutoff_first].astype(complex);
    if(len(Ens) % n_loc_dof != 0): Ens, psins = Ens[:-1], psins[:-1]; # must be even
    
    # measure alpha_val for each k_n
    alphans = np.empty((len(Ens),),dtype=complex);
    for n in range(len(Ens)):
        alphans[n] = np.dot( np.conj(psins[n]), np.matmul(alpha_mat_2d, psins[n]));
    n_bound_right= 1*n_bound_left; # bad
    
    # classify right well eigenstates in the \alpha basis
    Enbs = [];
    psinbs = [];
    for eigvali in range(len(alpha_eigvals)):
        Es_this_b, psis_this_b = [], [];
        for n in range(len(Ens)):
            if(abs(np.real(alphans[n] - alpha_eigvals[eigvali])) < eigval_tol):
                Es_this_b.append(Ens[n]); psis_this_b.append(psins[n]);
        Enbs.append(Es_this_b); psinbs.append(psis_this_b);

    # classify again with cutoff
    Enbs_arr = np.empty((n_loc_dof,n_bound_right),dtype=complex);
    psinbs_arr = np.empty((n_loc_dof,n_bound_right,len(psins[0])),dtype=complex);
    for beta in range(n_loc_dof):
        for n in range(n_bound_right):
            if(Enbs[beta][n] + 2*tRa < E_cutoff[beta,beta]):
                Enbs_arr[beta,n] = Enbs[beta][n];
                psinbs_arr[beta,n] = psinbs[beta][n];
    Enbs, psinbs = Enbs_arr, psinbs_arr;
    del Ens, psins;
    
    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=complex);
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):               
            for beta in range(n_loc_dof):
                # inelastic means averaging over an interval
                Mns = [];
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval/2):
                        melement = np.dot(np.conj(psinbs[beta,n]), np.matmul(Hdiff, psimas[alpha,m]));
                        if(np.real(melement) < 0):
                            print("\n\n\nWARNING: changing sign of melement");
                            melement *= (-1);
                        Mns.append(melement);

                # update M with average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;

    # get effective matrix elements
    Mbmas_tilde = np.zeros_like(Mbmas);
    for atilde in range(n_loc_dof):
        for btilde in range(n_loc_dof):
            for alpha in range(n_loc_dof):
                Mbmas_tilde[btilde,:,atilde] += change_basis[alpha,atilde]*change_basis[alpha,btilde]*Mbmas[alpha,:,alpha];
    Mbmas_dum = np.copy(Mbmas); # for plotting
    Mbmas_tilde = np.real(np.conj(Mbmas_tilde)*Mbmas_tilde);
    Mbmas_tilde = Mbmas_tilde.astype(float);
    del Mbmas

    return Emas, Mbmas_tilde;

#######################################################################
#### generate observables from matrix elements

def current(Emas, Mbmas, muR, eVb, kBT) -> np.ndarray:
    '''
    current as a function of bias voltage
    '''
    n_loc_dof, n_bound_left = np.shape(Emas);

    # bias voltage window
    stat_part = nFD(Emas, muR+eVb,kBT)*(1-nFD(Emas,muR,kBT)) - nFD(Emas,muR,kBT)*(1-nFD(Emas, muR+eVb,kBT));
    print(Emas.T,"\n",nFD(Emas,muR,kBT).T,"\n",stat_part.T);
    # sum over spin
    Iab = np.empty((n_loc_dof, n_loc_dof));
    for alpha in range(n_loc_dof):
        for beta in range(n_loc_dof):

            # sum over initial energy m
            #print( 2*np.pi*np.dot(stat_part[alpha],Mbmas[alpha,:,beta] ) ); assert False
            Iab[alpha,beta] = 2*np.pi*np.dot(stat_part[alpha],Mbmas[alpha,:,beta]);

    return Iab;

def Ts_bardeen(Emas, Mbmas, tL, tR, VL, VR, NL, NR, verbose = 0) -> np.ndarray:
    '''
    Using the n-averaged Oppenheimer matrix elements from bardeen.kernel,
    get the transmission coefficients.
    '''
    if(Mbmas.dtype != float): raise TypeError;
    n_loc_dof, n_bound_left = np.shape(Emas);
    if(len(np.shape(Mbmas)) != 3): raise ValueError;
    if(np.shape(Mbmas)[0] != n_loc_dof): raise ValueError;

    # convert from matrices to _{alpha alpha} elements
    to_convert = [tL, VL, tR, VR];
    converted = [];
    for convert in to_convert:
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ):
            raise ValueError; # VL must be diag
        converted.append(np.diagonal(convert));
    tLa, VLa, tRa, VRa = tuple(converted);

    # transmission probs
    Tbmas = np.empty_like(Mbmas);
    kmas = np.arccos((Emas-fci_mod.scal_to_vec(VLa,n_bound_left))
                    /(-2*fci_mod.scal_to_vec(tLa,n_bound_left))); # wavenumbers in the left well
    for alpha in range(n_loc_dof):
        for beta in range(n_loc_dof):
            Tbmas[alpha,:,beta] = NL/(kmas[alpha]*tLa[alpha]) * NR/(kmas[beta]*tRa[alpha]) *Mbmas[alpha,:,beta];

    return Tbmas;

def Ts_wfm(Hsys, Emas, tbulk, verbose=0) -> np.ndarray:
    '''
    Given bound state energies and Hsys, calculate the transmission
    probability for each energy using wfm code

    Used when the initial and final states have definite spin,
    and so CAN RESOLVE the spin -> spin transitions
    '''
    n_spatial_dof = np.shape(Hsys)[0];
    n_loc_dof = np.shape(Hsys)[-1];
    n_bound_left = np.shape(Emas)[-1];
    if(np.shape(Emas)[0] != n_loc_dof): raise ValueError;
    if(Hsys[0,0,0,0] != 0): raise Exception("Is for continuous leads not wells");

    # convert from Hsys to hblocks, tnn, tnnn 
    hblocks = np.empty((n_spatial_dof,n_loc_dof,n_loc_dof),dtype=complex);
    for sitei in range(n_spatial_dof):
        hblocks[sitei] = Hsys[sitei, sitei];
    tnn = np.empty((n_spatial_dof-1,n_loc_dof,n_loc_dof),dtype=complex);
    for sitei in range(n_spatial_dof-1):
        tnn[sitei] = Hsys[sitei, sitei+1];
    tnnn = np.empty((n_spatial_dof-2,n_loc_dof,n_loc_dof),dtype=complex);
    for sitei in range(n_spatial_dof-2):
        tnnn[sitei] = Hsys[sitei, sitei+2];
    for sitei in range(n_spatial_dof-3):
        assert(not np.any(Hsys[sitei, sitei+3]));
    if(verbose > 9):
        print(hblocks);
        print(tnn);
        print(tnnn);
        assert False;

    # get probabilities, final spin state resolved
    Tbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    for alpha in range(n_loc_dof):
        source = np.zeros((n_loc_dof,));
        source[alpha] = 1.0;
        for m in range(n_bound_left):
            Energy = Emas[alpha,m];
            if( np.isnan(Emas[alpha,m])):
                #Energy = Emas[alpha,m-1]
                Tdum = np.full( (n_loc_dof,), np.nan);
            else:
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tbulk, Energy, source, verbose = verbose);
            Tbmas[:,m,alpha] = Tdum;
            
    return Tbmas;

def Ts_wfm_well(tL, tR, VL, VR, HC, Emas, verbose=0) -> np.ndarray:
    '''
    Given bound state energies and HC from kernel, calculate the transmission
    probability for each energy using wfm code

    Used when the initial and final states have definite spin,
    and so CAN RESOLVE the spin -> spin transitions
    '''
    if(np.any(tL-tR)): raise NotImplementedError; # wfm code can't handle this case
    if(np.shape(Emas)[0] != np.shape(HC)[-1]): raise ValueError;
    n_spatial_dof = np.shape(HC)[0];
    n_loc_dof = np.shape(HC)[-1];
    n_bound_left = np.shape(Emas)[-1];

    ##### convert from HC to hblocks, tnn, tnnn
    # construct arrs
    hblocks = np.empty((n_spatial_dof+2,n_loc_dof,n_loc_dof),dtype=complex);
    hblocks[0] = VL*np.eye(n_loc_dof);
    tnn = np.empty((n_spatial_dof+1,n_loc_dof,n_loc_dof),dtype=complex);
    tnn[0] = -tL*np.eye(n_loc_dof);
    tnnn = np.empty((n_spatial_dof,n_loc_dof,n_loc_dof),dtype=complex);
    tnnn[0] = 0.0*np.eye(n_loc_dof);
    # convert
    for spacei in range(n_spatial_dof):
        for spacej in range(n_spatial_dof):
            if(spacei == spacej): # on-site
                hblocks[1+spacei] = HC[spacei,spacej];
            elif(spacei == spacej - 1): # nn hopping
                tnn[1+spacei] = HC[spacei,spacej];
            elif(spacei == spacej - 2): # next nn hopping
                tnnn[1+spacei] = HC[spacei,spacej];
            elif(spacei < spacej):
                assert(not np.any(HC[spacei,spacej]));
    hblocks[-1] = VR*np.eye(n_loc_dof);
    tnn[-1] = -tR*np.eye(n_loc_dof);
    tnnn[-1] = 0.0*np.eye(n_loc_dof);
    if(verbose > 9):
        print(hblocks);
        print(tnn);
        print(tnnn);
        assert False;

    # get probabilities, final spin state resolved
    Tbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    for alpha in range(n_loc_dof):
        source = np.zeros((n_loc_dof,));
        source[alpha] = 1.0;
        for m in range(n_bound_left):
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tL[alpha,alpha], Emas[alpha,m], source, verbose = verbose);
            Tbmas[:,m,alpha] = Tdum;
            
    return Tbmas;
    
############################################################################
#### Hamiltonian construction

def Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC, bound=True) -> np.ndarray:
    '''
    Make the TB Hamiltonian for the full system, general 1D case
    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR is hopping in these regions (2d arr describing local dofs)
    Vinfty, VL, VR is local potential in these regions (2d arr describing local dofs)
    Ninfty, NL, NR is number of sites in these regions
    HC is Hamiltonian of central region (4d arr describing spatial and local dofs)
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

    # whether L and R states are bound at ends or not
    if bound:
        VinftyL = Vinfty; VinftyR = Vinfty; del Vinfty;
    else:
        print("\n\nWARNING: NOT BOUND\n\n");
        VinftyL = VL; VinftyR = VR; del Vinfty;

    # Hamiltonian matrix
    Hmat = np.zeros((nsites,nsites,n_loc_dof,n_loc_dof),dtype=complex);
    for j in range(minusinfty, plusinfty+1):

        # diag outside HC
        if(j < -NL - littleNC):  # far left        
            Hmat[j-minusinfty,j-minusinfty] += VinftyL;
        elif(j >= -NL-littleNC and j < -littleNC): # left well
            Hmat[j-minusinfty,j-minusinfty] += VL;
        elif(j > littleNC and j <= littleNC+NR): # right well
            Hmat[j-minusinfty,j-minusinfty] += VR;
        elif(j > littleNC+NR): # far right
            Hmat[j-minusinfty,j-minusinfty] += VinftyR;

        # off diag outside HC
        if(j < -NL - littleNC):  # far left         
            Hmat[j-minusinfty,j+1-minusinfty] += -tinfty;
            Hmat[j+1-minusinfty,j-minusinfty] += -tinfty;
        elif(j >= -NL-littleNC and j < -littleNC): # left well
            Hmat[j-minusinfty,j+1-minusinfty] += -tL;
            Hmat[j+1-minusinfty,j-minusinfty] += -tL;
        elif(j > littleNC and j <= littleNC+NR): # right well
            Hmat[j-minusinfty,j-1-minusinfty] += -tR;
            Hmat[j-1-minusinfty,j-minusinfty] += -tR; 
        elif(j > littleNC+NR): # far right
            Hmat[j-minusinfty,j-1-minusinfty] += -tinfty;
            Hmat[j-1-minusinfty,j-minusinfty] += -tinfty;

    # HC
    Hmat[-littleNC-minusinfty:littleNC+1-minusinfty,-littleNC-minusinfty:littleNC+1-minusinfty] = HC;
            
    return Hmat, minusinfty;

##################################################################################
#### utils

def matrix_element(beta,psin,op,alpha,psim) -> complex:
    '''
    Take the matrix element of a
    -not in general alpha conserving 2d operator, with spin/spatial dofs mixed
    -alpha conserving 2d state vector, with spin/spatial dofs separated
    '''
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
    psinbeta[beta] = psin[beta]; # all zeros except for psi[beta]
    psinbeta = fci_mod.vec_2d_to_1d(psinbeta.T); # flatten
    assert(is_alpha_conserving(psinbeta,n_loc_dof));
    return np.dot(np.conj(psinbeta), np.dot(op,psimalpha));

def is_alpha_conserving(T,n_loc_dof,tol=1e-9) -> bool:
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
            alphas[ai] = np.any(abs(T[indices % n_loc_dof == ai]) > tol);
        return (sum(alphas) == 1 or sum(alphas) == 0);

    elif len(shape) == 2: #matrix
        for i in range(shape[0]):
            for j in range(shape[1]):
                if(abs(T[i,j]) > tol):
                    if(i % n_loc_dof != j % n_loc_dof):
                        return False;
        return True;

    else: raise NotImplementedError; 

def plot_ham(j, hams, alpha, label=True) -> None:
    '''
    '''
    if( not isinstance(hams, tuple)): raise TypeError;
    n_loc_dof = np.shape(hams[0])[-1];
    ham_strs = ["$H_{sys}$","$H_{L}$","$H_{sys}-H_L$"];
    if( len(hams) != len(ham_strs)): raise ValueError;

    # construct axes
    nax = len(hams);
    myfig,myaxes = plt.subplots(nax, sharex=True);
    if(nax == 1): myaxes = [myaxes];

    # iter over hams
    for hami in range(len(hams)):
        ham_4d = hams[hami];
        myaxes[hami].plot(j, np.diag(ham_4d[:,:,alpha,alpha]), color="cornflowerblue");
        myaxes[-1].set_xlabel("$j$");
        myaxes[hami].set_ylabel("$V_j$");
        myaxes[hami].set_title(ham_strs[hami]+"["+str(alpha)+""+str(alpha)+"]");

        # label
        if(label):
            textbase = -0.1;
            VL = ham_4d[len(j)//4,len(j)//4,alpha,alpha];
            VC = ham_4d[len(j)//2,len(j)//2,alpha,alpha];
            VR = ham_4d[len(j)*3//4,len(j)*3//4,alpha,alpha];
            Vinfty = ham_4d[-1,-1,alpha,alpha];            
            if(hami == 0):
                Vcoords = [j[len(j)//4],j[len(j)//2],j[len(j)*3//4],j[-1]];
                Vs = [VL, VC, VR, Vinfty];
                Vlabels = ["VL","VC","VR","Vinfty"];
            elif(hami == 1):
                Vcoords =  [j[len(j)//4],j[len(j)//2],j[len(j)*3//4],j[-1]];
                Vs = [VL, VC, VR, Vinfty];
                Vlabels = ["VL","VC","VRprime","Vinfty"];
            elif(hami == 2):
                Vcoords = [j[len(j)*3//4]];
                Vlabels = ["VR - VRprime"];
                Vs = [VR];
            for Vi in range(len(Vs)):
                myaxes[hami].annotate(Vlabels[Vi], xy=(Vcoords[Vi], Vs[Vi]), xytext=(Vcoords[Vi], textbase),arrowprops=dict(arrowstyle="->", relpos=(0,1)),xycoords="data", textcoords="data")

    # format
    plt.tight_layout();
    plt.show();

def nFD(epsilon,mu,kBT):
    '''
    Fermi-Dirac distribution function
    '''
    assert isinstance(mu,float);
    return 1/(np.exp((np.real(epsilon)-mu)/kBT )+1)

def get_self_energy(t, V, E) -> np.ndarray:
    if(not isinstance(t, float) or t < 0): raise TypeError;
    if(not isinstance(V, float)): raise TypeError;
    if(not isinstance(E, float)): raise TypeError;
    dummy = (E-V)/(-2*t);
    return-(dummy+np.lib.scimath.sqrt(dummy*dummy-1));

def couple_to_cont(H, E, alpha0) -> np.ndarray:
    '''
    Couple a 4d Hamiltonian H to a continuum state with energy E and spin alpha0
    by using absorbing/emitting bcs
    '''
    if(len(np.shape(H)) != 4): raise ValueError;
    n_loc_dof = np.shape(H)[-1];

    # right and left
    for sidei in [0,-1]:

        # get the self energy
        #print("----->",-np.real(H[sidei,sidei+1+sidei*2,alpha0,alpha0]),np.real(H[sidei,sidei,alpha0,alpha0]),np.real(E))
        selfenergy = get_self_energy(-np.real(H[sidei,sidei+1+sidei*2,alpha0,alpha0]),np.real(H[sidei,sidei,alpha0,alpha0]),np.real(E));

        # for all others just absorb
        for alpha in range(n_loc_dof):
            if(alpha == alpha0 and sidei == 0): # emit in alpha0 on left only
                H[sidei,sidei,alpha,alpha] += np.conj(selfenergy);
            else: # absorb
                H[sidei,sidei,alpha,alpha] += selfenergy;

    return H;

def get_eigs(h_4d, E_cutoff) -> tuple:
    '''
    Get eigenvalues and eigenvectors of a 4d (non hermitian) hamiltonian
    '''
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

def plot_wfs(h_4d, E_cutoff):
    '''
    '''
    raise NotImplementedError
    if(len(np.shape(h_4d)) != 4): raise ValueError;
    n_loc_dof = np.shape(h_4d)[-1];
    spatial_orbs = np.shape(h_4d)[0];
    assert(spatial_orbs % 2 == 1);
    mid = spatial_orbs // 2;
    jvals = np.array(range(-mid,mid+1));

    # eigenstates
    Es, psis = get_eigs(h_4d,E_cutoff);




#####################################################################################################
#### run code

if __name__ == "__main__":

    pass;
    

