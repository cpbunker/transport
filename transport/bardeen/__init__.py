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

    Returns:
    -Emas, complex 2d array, initial energies separated by spin and energy
    - Mbmas, real 3d array, NORM SQUARED of Oppenheimer matrix elements,
        after averaging over final energies so that n is not a free index
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
    HL_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
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
    del Ems, psims, alpha
    Emas, psimas = Emas_arr, psimas_arr # shape is (n_loc_dof, n_bound_left)

    # right well eigenstates  
    HR_4d = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
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
    Hsys_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    if(verbose > 9): # plot hams
        print("np.shape(Emas) = ",np.shape(Emas));
        print("np.shape(Enbs) = ",np.shape(Enbs));
        for alpha in range(n_loc_dof):
            plot_ham((Hsys_4d,HL_4d,Hsys_4d-HL_4d), ["$H_{sys}$","$H_{L}$","$H_{sys}-H_L$"], alpha );
        raise NotImplementedError;

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):               
            for beta in range(n_loc_dof):
                # inelastic means averaging over an energy interval
                Mns = [];
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval):
                        Mns.append( matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]));

                # "adaptive" interval 
                if( np.isnan(interval) and Mns==[]): 
                    n_nearest = np.argmin( abs(Emas[alpha,m] - Enbs[beta]) );
                    Mns.append( matrix_element(beta,psinbs[:,n_nearest],Hdiff,alpha,psimas[:,m]));    

                # update M with average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;

    # norm squared of Oppenheimer matrix elements               
    Mbmas = np.real(np.conj(Mbmas)*Mbmas);
    Mbmas = Mbmas.astype(float);               
    return Emas, Mbmas;

def kernel_well_super(tinfty, tL, tR,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC, HCprime, alpha_mat, E_cutoff,
           interval=1e-9,expval_tol=1e-9,verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nbma averaged over n in a
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
    -interval, float, the maximum allowed difference between initial and final
        state energies.
    -expval_tol, float, when classifying eigenstates into the alpha basis,
        there will be some deviation of <k_m \alpha | alpha_mat | k_m \alpha>
        around its true value due to symmetry breaking. This is the allowed
        tolerance of such deviation

    Returns:
    -Emas, complex 2d array, initial energies separated by spin and energy
    - Mbmas_tilde, real 3d array, NORM SQUARED of EFFECTIVE Oppenheimer
        matrix elements, after averaging over final energies (ie n is not
        a free index)
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

    # change of basis
    # alpha basis is eigenstates of alpha_mat, alpha still a good spin quant #
    _, alphastates = np.linalg.eigh(alpha_mat);
    alphastates = alphastates.T;
    tildestates = np.eye(n_loc_dof);
    # get change_basis matrix st
    # |\tilde{\alpha}> = \sum_alpha change_basis[\alpha,\tilde{\alpha}] |\alpha>
    change_basis = np.empty_like(alphastates);
    for astatei in range(len(alphastates)):
        for tstatei in range(len(tildestates)):
            change_basis[astatei, tstatei] = np.dot( np.conj(alphastates[astatei]), tildestates[tstatei]);

    # find left lead bound states, they will be in alpha basis
    HL_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    Emas, psimas = get_bound_states(HL_4d, tLa, alpha_mat, E_cutoff, expval_tol = expval_tol, verbose=verbose);

    # find right lead bound states, they will be in alpha basis
    HR_4d = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    Enbs, psinbs = get_bound_states(HR_4d, tRa, alpha_mat, E_cutoff, expval_tol = expval_tol, verbose=verbose);

    # physical system
    Hsys_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);  
    if(verbose > 9): # plot wfs, hams
        plot_wfs(HL_4d, psimas, Emas, which_m = 13);
        plot_wfs(HR_4d, psinbs, Enbs, which_m = 13);
        for alpha in range(n_loc_dof):
            plot_ham((Hsys_4d,HL_4d,Hsys_4d-HL_4d), ["$H_{sys}$","$H_{L}$","$H_{sys}-H_L$"], alpha );
        for h in (Hsys_4d,HL_4d):
            h_aa = h[:,:,0,0] ;
            h_bb = h[:,:,1,1] - np.diagflat(np.diagonal(0.02*np.ones_like(h_aa)));
            assert( not np.any(h_aa-h_bb));
        raise NotImplementedError;
    
    # average matrix elements over final states |k_n \beta>
    # with the energy sufficiently close to that of the
    # initial state |k_m \alpha>
    # keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mbmas = np.empty((n_loc_dof,np.shape(Emas)[-1],n_loc_dof),dtype=complex);
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(np.shape(Emas)[-1]):               
            for beta in range(n_loc_dof):
                # inelastic means averaging over an enegy interval
                Mns = [];
                for n in range(np.shape(Enbs)[-1]):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval):
                        melement = np.dot(np.conj(psinbs[beta,n]), np.matmul(Hdiff, psimas[alpha,m]));
                        if(np.real(melement) < 0):
                            if(verbose>5): print("\tWARNING: changing sign of melement");
                            melement *= (-1);
                        Mns.append(melement);

                # "adaptive" interval 
                if( np.isnan(interval) and Mns==[]): 
                    n_nearest = np.argmin( abs(Emas[alpha,m] - Enbs[beta]) );
                    melement = np.dot(np.conj(psinbs[beta,n_nearest]), np.matmul(Hdiff, psimas[alpha,m]));
                    if(np.real(melement) < 0):
                        if(verbose>5): print("\tWARNING: changing sign of melement");
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
    del Mbmas;
    
    # norm squared of effective matrix elements 
    Mbmas_tilde = np.real(np.conj(Mbmas_tilde)*Mbmas_tilde);
    Mbmas_tilde = Mbmas_tilde.astype(float);
    return Emas, Mbmas_tilde; #NB sometimes up and down get switched in Emas

def kernel_well_spinless(tinfty, tL, tR,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC, HCobs, defines_Sz,
            E_cutoff, interval=1e-12, expval_tol=1e-12, verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nm averaged over n in a
    nearby interval. NB there is no alpha and beta anymore because
    spin translational symmetry has been broken so that there are no more
    good spin quantum numbers.
    
    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR
    
    For this kernel, the initial and final states are superpositions of
    the eigenstates of HL and HR. The latter is the basis imposed on us
    by the physics, we cannot choose it, and energy is in general the only
    good quantum number in this basis.

    Optional args:
    -E_cutoff, float, don't calculate T for eigenstates with energy higher 
        than this. That way we limit to bound states
    -interval, float, the maximum allowed difference between initial and final
        state energies
    -expval_tol, float, when classifying eigenstates of the observable basis,
        we compare their expectation values of the defines_Sz operator with
        the eigenvalues of that operator. This gives the tolerance for binning
        a state as having a certain eigenvalue of that operator

    Returns:
    -Emas, complex 2d array, initial energies separated by spin and energy
    - Mbmas_eff, real 3d array, NORM SQUARED of EFFECTIVE Oppenheimer
        matrix elements, after averaging over final energies (ie n is not
        a free index)
    '''
    if(np.shape(HC) != np.shape(HCobs)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];
    if(n_loc_dof != len(defines_Sz)): raise ValueError;

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

    # physical system
    Hsys_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    
    # left lead bound states, energy is only good quantum number
    HL_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HC);
    Ems, psims = get_mstates(HL_4d, tLa, verbose=verbose);

    # left lead observable basis, where Sz is a good quantum number
    HLobs_4d = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCobs);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HLobs_4d),n_loc_dof));
    Emus, psimus = get_mstates(HLobs_4d, tLa, verbose=verbose);

    # right lead bound states, energy is only good quantum number
    HR_4d = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HC);
    Ens, psins = get_mstates(HR_4d, tRa, verbose=verbose);

    # right lead observable basis, where Sz is a good quantum number
    HRobs_4d = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCobs);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HRobs_4d),n_loc_dof));
    Enus, psinus = get_mstates(HRobs_4d, tRa, verbose=verbose);

    if False: # some properties of n and nu states
        print("Ens:",len(Ens), Ens[:10]);
        print("Enus:",len(Enus), Enus[:10]);
        for n in range(20):
            for nu in range(30):
                if(n%2==0 and nu in [n,n+1,n+2,n+3]): # abs(Ens[n]-Enus[nu])<1e-8 or abs(Ens[n]-Enus[nu]-0.02)<1e-8):
                    print("<n="+str(n)+", E_n="+str(Ens[n])+"|\\nu="+str(nu)+", E_\\nu>="+str(Enus[nu])+"> = ", np.dot(np.conj(psins[n]), psinus[nu]) );
                    #plot_wfs(HR_4d, np.array([psins,psins]), np.array([Ens,Ens]), which_m = n, title="$\psi_n$");
                    #plot_wfs(HRobs_4d, np.array([psinus,psinus]), np.array([Enus,Enus]), which_m = nu, title="$\psi_\\nu$");
                    # NB we want this to be \pm 1 but the sign does not matter
        assert False;

    if(verbose > 9): # plot wfs
        #plot_ham((Hsys_4d,), ["$H_{sys}$"], 0 );
        #plot_ham((HL_4d,), ["$H_{L}$"], 0 );
        #plot_ham((HR_4d,), ["$H_{R}$"], 0 );
        plot_wfs(Hsys_4d, np.array([psims,psims]), np.array([Ems,Ems]), which_m = None, title="$H_{sys}$");
        plot_wfs(HL_4d, np.array([psims[::n_loc_dof],psims[1::n_loc_dof]]), np.array([Ems[::n_loc_dof],Ems[1::n_loc_dof]]), which_m = 13, title="$H_L$");
        plot_wfs(HLobs_4d, np.array([psimus[::n_loc_dof],psimus[1::n_loc_dof]]), np.array([Emus[::n_loc_dof], Emus[1::n_loc_dof]]), which_m = 13, title="$\\tilde{H}_L$");
        plot_ham((Hsys_4d,HL_4d,HR_4d), ["$H_{sys}$","$H_{L}$","$H_R$"], 0 );
        raise NotImplementedError;

    # average matrix elements over final states |k_n >
    # with the energy sufficiently close to that of the
    # initial state |k_m \alpha>
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mms = np.empty_like(Ems);
    # initial energy and spin states
    for m in range(np.shape(Ems)[-1]):               
        # inelastic means averaging over an energy interval
        Mns = [];
        for n in range(np.shape(Ens)[-1]):
            if( abs(Ems[m] - Ens[n]) < interval):
                melement = np.dot(np.conj(psins[n]), np.matmul(Hdiff, psims[m]));
                if(np.real(melement) < 0):
                    if(verbose>5): print("\tWARNING: changing sign of melement");
                    melement *= (-1);
                Mns.append(melement);

        # "adaptive" interval 
        if( np.isnan(interval) and Mns==[]): 
            n_nearest = np.argmin( abs(Ems[m] - Ens[beta]) );
            melement = np.dot(np.conj(psins[n_nearest]), np.matmul(Hdiff, psims[m]));
            if(np.real(melement) < 0):
                if(verbose>5): print("\tWARNING: changing sign of melement");
                melement *= (-1);
            Mns.append(melement);

        # update M with average
        if(verbose): print("\tinterval = ",interval, len(Mns));
        #if(len(Mns) not in [0,1]): raise Exception("This destroys final state spin information");
        if Mns: Mns = sum(Mns)/len(Mns);
        else: Mns = 0.0;
        Mms[m] = Mns;

    ####
    #### averaging must be done in eigenbasis of system (ie above)
    #### but we want to keep n info if possible, so that we can
    #### resolve final spin info. So we do this:
    Mnms = np.diagflat(Mms);
    del Mms;

    # change of basis matrix from psims to psi_mus (for observables)
    # st psi_mu = \sum_m change_basis[m, \mu] \psi_m
    change_basis = np.empty((np.shape(Ems)[-1], np.shape(Emus)[-1]),dtype=complex);
    for m in range(np.shape(Ems)[-1]):
        for mu in range(np.shape(Emus)[-1]):
            change_basis[m, mu] = np.dot( np.conj(psims[m]), psimus[mu]);
            
    # get effective matrix elements by changing basis
    Mnumus = np.matmul( np.conj(change_basis.T), np.matmul(Mnms, change_basis) );
    del Mnms, Ems, Ens, psims, psins;

    # Norm squared of effective matrix elements               
    Mnumus = np.real(np.conj(Mnumus)*Mnumus);
    Mnumus = Mnumus.astype(float);

    # cutoff
    E_cutoff_first = np.max(E_cutoff);
    mu_cutoff = len(Emus[Emus+2*tLa < E_cutoff_first]);
    Emus = Emus[:(mu_cutoff//n_loc_dof)*n_loc_dof];
    psimus = psimus[:(mu_cutoff//n_loc_dof)*n_loc_dof];
    nu_cutoff = len(Enus[Enus+2*tRa < E_cutoff_first]);
    Enus = Enus[:(nu_cutoff//n_loc_dof)*n_loc_dof];
    psinus = psinus[:(nu_cutoff//n_loc_dof)*n_loc_dof];
    Mnumus = Mnumus[:(nu_cutoff//n_loc_dof)*n_loc_dof,:(mu_cutoff//n_loc_dof)*n_loc_dof];

    # classify the psi_mus and psi_nus by \sigma_z, and likewise break up M nu mus
    E_mualphas = np.zeros((n_loc_dof,len(Emus)),dtype=complex);
    psi_mualphas = np.zeros((n_loc_dof,len(Emus),len(psimus[0])),dtype=complex);
    E_nubetas = np.zeros((n_loc_dof,len(Enus)),dtype=complex);
    psi_nubetas = np.zeros((n_loc_dof,len(Enus),len(psinus[0])),dtype=complex);
    M_betanu_alphamus = np.empty((n_loc_dof,len(Enus),n_loc_dof,len(Emus)),dtype=float);

    # need to take exp vals of Sz to do this
    expvals_exact, _ = np.linalg.eigh(defines_Sz);
    print("expvals_exact = ", expvals_exact);
    defines_Sz_4d = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof),dtype=complex);
    for sitej in range(n_spatial_dof):
        defines_Sz_4d[sitej,sitej] = np.copy(defines_Sz);
    defines_Sz_2d = fci_mod.mat_4d_to_2d(defines_Sz_4d);

    # classify mu states
    mucounter = np.zeros((n_loc_dof,),dtype = int);
    for mu in range(np.shape(Emus)[-1]):
        expSz_mu = np.dot( np.conj(psimus[mu]), np.dot(defines_Sz_2d, psimus[mu]));
        Sz_index_mu = 2;
        for expvali_mu in range(n_loc_dof):
            if(abs(expSz_mu-expvals_exact[expvali_mu])<expval_tol):                
                Sz_index_mu = expvali_mu;
                E_mualphas[Sz_index_mu,mucounter[Sz_index_mu]] = Emus[mu];
                psi_mualphas[Sz_index_mu,mucounter[Sz_index_mu]] = psimus[mu];
                mucounter[Sz_index_mu] += 1;                                
        if(Sz_index_mu not in np.array(range(n_loc_dof))): raise Exception("<Sz>_mu = "+str(expSz_mu)+" not an eigenval of Sz");
        # classify nu states
        nucounter = np.zeros((n_loc_dof,),dtype = int);
        for nu in range(np.shape(Enus)[-1]):
            expSz_nu = np.dot( np.conj(psinus[nu]), np.dot(defines_Sz_2d, psinus[nu]));
            Sz_index_nu = 2;
            for expvali_nu in range(n_loc_dof):
                if(abs(expSz_nu-expvals_exact[expvali_nu])<expval_tol):
                    Sz_index_nu = expvali_nu;
                    E_nubetas[Sz_index_nu,nucounter[Sz_index_nu]] = Enus[nu];
                    psi_nubetas[Sz_index_nu,nucounter[Sz_index_nu]] = psinus[nu];
                    nucounter[Sz_index_nu] += 1;
            if(Sz_index_nu not in np.array(range(n_loc_dof))): raise Exception("<Sz>_nu = "+str(expSz_nu)+" not an eigenval of Sz");
             # classify M nu mus
            M_betanu_alphamus[Sz_index_nu,nucounter[Sz_index_nu]-1,Sz_index_mu,mucounter[Sz_index_mu]-1] = Mnumus[nu,mu];
    del Emus, psimus, Enus, psinus, Mnumus;

    # truncate again
    E_mualphas_trunc = np.empty((n_loc_dof,np.min(mucounter)),dtype=complex);
    E_nubetas_trunc = np.empty((n_loc_dof,np.min(nucounter)),dtype=complex);
    Mnbmas_eff = np.empty((n_loc_dof,np.min(nucounter),n_loc_dof,np.min(mucounter)),dtype=float);
    for alpha in range(n_loc_dof):
        E_mualphas_trunc[alpha,:] = E_mualphas[alpha,:np.min(mucounter)];
        for beta in range(n_loc_dof):
            E_nubetas_trunc[beta,:] = E_nubetas[beta,:np.min(nucounter)];
            Mnbmas_eff[alpha,:,beta,:] = M_betanu_alphamus[beta,:np.min(nucounter),alpha,:np.min(mucounter)];
    if(verbose > 9): # plot wfs
        print("E_mualphas = ", np.shape(E_mualphas),"\n",E_mualphas+2*tLa);
        print("E_mualphas_trunc = ", np.shape(E_mualphas_trunc),"\n",E_mualphas_trunc+2*tLa);
        for mymu in [18,19]:
            print("psi_mu = ",mymu);
            expSz_mu_alpha0 = np.dot( np.conj(psi_mualphas[0,mymu]), np.dot(defines_Sz_2d, psi_mualphas[0,mymu]));
            expSz_mu_alpha1 = np.dot( np.conj(psi_mualphas[1,mymu]), np.dot(defines_Sz_2d, psi_mualphas[1,mymu]));
            print("<Sz> of alpha0 = ", expSz_mu_alpha0);
            print("<Sz> of alpha1 = ", expSz_mu_alpha1);
            plot_wfs(HLobs_4d, psi_mualphas, E_mualphas_trunc, which_m = mymu);
        raise NotImplementedError;
    E_mualphas, E_nubetas = E_mualphas_trunc, E_nubetas_trunc;
    
    # get rid of n as free index
    Mbmas_eff = np.empty((n_loc_dof,np.shape(E_mualphas)[-1],n_loc_dof),dtype=float);
    for beta in range(n_loc_dof):
        for alpha in range(n_loc_dof):
            Mbmas_eff[beta,:,alpha] = np.diagonal(Mnbmas_eff[beta,:,alpha]);

    return E_mualphas, Mbmas_eff;

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
    if(abs(np.max(np.imag(kmas)))>1e-10): print(abs(np.max(np.imag(kmas)))); raise ValueError;
    kmas = np.real(kmas); # force to be real
    
    # check
    check = False;
    if(check):
        fig, axes = plt.subplots(n_loc_dof,n_loc_dof,sharex=True,sharey=True);
        if(n_loc_dof==1): axes = np.array([[axes]]);
    
    for alpha in range(n_loc_dof):
        for beta in range(n_loc_dof):
            factor_from_dos = 1/np.sqrt(1-np.power((Emas[alpha]-VRa[beta])/(2*tRa[beta]),2));
            if(abs(np.max(np.imag(factor_from_dos)))>1e-10): print(abs(np.max(np.imag(factor_from_dos)))); raise ValueError;
            factor_from_dos = np.real(factor_from_dos); # force to be real
            Tbmas[alpha,:,beta] = NL/(kmas[alpha]*tLa[alpha]) * NR/tRa[beta] *factor_from_dos*Mbmas[alpha,:,beta];

            if(check):
                print("*"*40,"\n",alpha,beta);
                for el in kmas[alpha]: print(el);
                print("*"*40,"\n",alpha,beta);
                for el in factor_from_dos: print(el);
                axes[alpha, beta].plot(Emas[alpha]+2*tLa[alpha], factor_from_dos,label="true term",linestyle="solid" );
                axes[alpha, beta].plot(Emas[alpha]+2*tLa[alpha], 1/kmas[beta], label="$1/k_m a$", linestyle="dashed");
                axes[alpha,beta].set_ylabel("$\\alpha, \\beta = $"+str(alpha)+", "+str(beta));

    if(check):
        axes[-1,-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $");
        axes[0,0].set_title("Checking validity of $k_m$ substitution");
        plt.legend();
        plt.show();
        assert False

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
            
    return Hmat;

##################################################################################
#### utils

def get_mstates(H_4d, ta, verbose=0) -> tuple:
    '''
    There is no cutoff because we need a complete basis!
    '''
    H_2d = fci_mod.mat_4d_to_2d(H_4d);
    Ems, psims = np.linalg.eigh(H_2d);
    return Ems.astype(complex), psims.T;

def get_bound_states(H_4d, ta, alpha_mat, E_cutoff, expval_tol = 1e-9, verbose=0) -> tuple:
    '''
    '''
    n_spatial_dof = np.shape(H_4d)[0];
    n_loc_dof = np.shape(H_4d)[-1];
    E_cutoff_first = np.max(E_cutoff);

    # all eigenstates
    H_2d = fci_mod.mat_4d_to_2d(H_4d);
    Ems, psims = np.linalg.eigh(H_2d);

    # cutoff
    num_cutoff = len(Ems[Ems+2*ta < E_cutoff_first]);
    Ems = Ems[:(num_cutoff//n_loc_dof)*n_loc_dof].astype(complex);
    psims = psims.T[:(num_cutoff//n_loc_dof)*n_loc_dof];
    if(verbose): print(">>>", np.shape(Ems));

    # measure alpha val for each k_m
    alpha_mat_4d = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof),dtype=complex);
    for sitej in range(n_spatial_dof):
        alpha_mat_4d[sitej,sitej] = np.copy(alpha_mat);
    alpha_mat_2d = fci_mod.mat_4d_to_2d(alpha_mat_4d);
    alphams = np.empty((len(Ems),),dtype=complex);
    for m in range(len(Ems)):
        alphams[m] = np.dot( np.conj(psims[m]), np.matmul(alpha_mat_2d, psims[m]));

    # commutator of alpha_mat with H
    # when alpha eigval classification fails, it is because they don't commute!
    commutator = np.matmul(alpha_mat_2d, H_2d) - np.matmul(H_2d, alpha_mat_2d);
    if(verbose): print("\ncommutator = ", np.max(abs(commutator)));

    # get all unique alpha vals, should be exactly n_loc_dof of them
    if(verbose):
        alpha_eigvals_exact, _ = np.linalg.eigh(alpha_mat);
        print("\nalpha_mat =\n", alpha_mat);
        print("alpha_eigvals_exact =\n",{el:0 for el in alpha_eigvals_exact});
    alpha_eigvals = dict();
    for alpha in alphams:
        addin = True;
        for k in alpha_eigvals.keys():
            if(abs(alpha-k) < expval_tol):
                alpha_eigvals[k] += 1;
                addin = False;
        if(addin):
            alpha_eigvals[alpha] = 1;
    if(verbose): print("\nalpha_eigvals = (expval_tol = "+str(expval_tol)+")\n",alpha_eigvals);   
    if(len(alpha_eigvals.keys()) != n_loc_dof): raise Exception("alpha vals");
    n_bound_left = np.min(list(alpha_eigvals.values()));
    alpha_eigvals = list(alpha_eigvals.keys());

    # classify left well eigenstates in the \alpha basis
    Emas = [];
    psimas = [];
    for eigvali in range(len(alpha_eigvals)):
        Es_this_a, psis_this_a = [], [];
        for m in range(len(Ems)):
            if(abs(np.real(alphams[m] - alpha_eigvals[eigvali])) < expval_tol):
                Es_this_a.append(Ems[m]); psis_this_a.append(psims[m]);
        Emas.append(Es_this_a); psimas.append(psis_this_a);

    # classify again with cutoff
    Emas_arr = np.empty((n_loc_dof,n_bound_left),dtype=complex);
    psimas_arr = np.empty((n_loc_dof,n_bound_left,len(psims[0])),dtype=complex);
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):
            if(Emas[alpha][m] + 2*ta < E_cutoff[alpha,alpha]):
                Emas_arr[alpha,m] = Emas[alpha][m];
                psimas_arr[alpha,m] = psimas[alpha][m];
    if(verbose): print(">>>", np.shape(Emas_arr));

    return Emas_arr, psimas_arr;

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

def plot_ham(hams, ham_strs, alpha, label=True) -> None:
    '''
    '''
    if( not isinstance(hams, tuple)): raise TypeError;
    n_loc_dof = np.shape(hams[0])[-1];
    if( len(hams) != len(ham_strs)): raise ValueError;

    # spatial dofs
    spatial_orbs = np.shape(hams[0])[0];    
    mid = spatial_orbs // 2;
    jvals = np.array(range(-mid,mid+1));

    # construct axes
    nax = len(hams);
    myfig,myaxes = plt.subplots(nax, sharex=True);
    if(nax == 1): myaxes = [myaxes];

    # iter over hams
    for hami in range(len(hams)):
        ham_4d = hams[hami];
        myaxes[hami].plot(jvals, np.real(np.diag(ham_4d[:,:,alpha,alpha])), color="cornflowerblue");
        myaxes[-1].set_xlabel("$j$");
        myaxes[hami].set_ylabel("$V_j$");
        myaxes[hami].set_title(ham_strs[hami]+"["+str(alpha)+""+str(alpha)+"]");

        # label
        if(label):
            textbase = -0.1;
            VL = ham_4d[len(jvals)//4,len(jvals)//4,alpha,alpha];
            VC = ham_4d[len(jvals)//2,len(jvals)//2,alpha,alpha];
            VR = ham_4d[len(jvals)*3//4,len(jvals)*3//4,alpha,alpha];
            Vinfty = ham_4d[-1,-1,alpha,alpha];            
            if(hami == 0):
                Vcoords = [jvals[len(jvals)//4],jvals[len(jvals)//2],jvals[len(jvals)*3//4],jvals[-1]];
                Vs = [VL, VC, VR, Vinfty];
                Vlabels = ["VL","VC","VR","Vinfty"];
            elif(hami == 1):
                Vcoords =  [jvals[len(jvals)//4],jvals[len(jvals)//2],jvals[len(jvals)*3//4],jvals[-1]];
                Vs = [VL, VC, VR, Vinfty];
                Vlabels = ["VL","VC","VRprime","Vinfty"];
            elif(hami == 2):
                Vcoords =  [jvals[len(jvals)//4],jvals[len(jvals)//2],jvals[len(jvals)*3//4],jvals[-1]];
                Vs = [VL, VC, VR, Vinfty];
                Vlabels = ["VLprime","VC","VR","Vinfty"];
            elif(hami == 3):
                Vcoords = [jvals[len(jvals)*3//4]];
                Vlabels = ["VR - VRprime"];
                Vs = [VR];
            for Vi in range(len(Vs)):
                myaxes[hami].annotate(Vlabels[Vi], xy=(Vcoords[Vi], Vs[Vi]), xytext=(Vcoords[Vi], textbase),arrowprops=dict(arrowstyle="->", relpos=(0,1)),xycoords="data", textcoords="data")

    # format
    plt.tight_layout();
    plt.show();

def plot_wfs(h_4d, psimas, Emas, which_m = 0, title = "$H$", imag_tol = 1e-12, reverse = False, fit_exp=False):
    '''
    '''
    if(len(np.shape(h_4d)) != 4): raise ValueError;
    n_loc_dof = np.shape(h_4d)[-1];
    fig, axes = plt.subplots(n_loc_dof);
    if(n_loc_dof==1): axes = [axes];

    # spatial dofs
    spatial_orbs = np.shape(h_4d)[0];    
    mid = spatial_orbs // 2;
    jvals = np.array(range(-mid,mid+1));
    
    # separate by up, down channels
    channel_strs = ["$| \\uparrow \\rangle $", "$|\\downarrow \\rangle$"];
    for channel in range(n_loc_dof):
        # plot spin-diagonal part of H
        axes[channel].plot(jvals, np.real(np.diagonal(h_4d[:,:,channel,channel])), color="black");
        axes[channel].axhline(0.0,color="gray",linestyle="dashed");
        # off-diagonal (spin-flip)
        axes[channel].fill_between(jvals, np.zeros_like(jvals), abs(np.real(np.diagonal(h_4d[:,:,channel,channel-1]))),color="green",alpha=0.5);

        # plot wfs
        if(which_m is not None): 
            alphas = np.array(range(n_loc_dof));
            if(reverse): alphas = alphas[::-1];
            scale = np.max(abs(psimas[:, which_m]));
            for alpha in alphas:
                psi = psimas[alpha, which_m][channel::n_loc_dof];
                energy = np.round(np.real(Emas[alpha, which_m])+2.0, decimals=10);
                if(scale < 1e-12): scale = 1.0;
                assert(np.max(np.imag(psi)) < imag_tol); # otherwise plot complex part
                axes[channel].plot(jvals, np.real((0.25/scale)*psi),
                                   label = "$|k_{m="+str(which_m)+"}, \\alpha = "+str(alpha)+"\\rangle (\\varepsilon = "+str(energy)+")$");  
        axes[channel].set_ylabel(channel_strs[channel]);

    # show
    axes[0].legend();
    axes[0].set_title(title);
    plt.tight_layout();
    plt.show();

    if(fit_exp):
        def fit_kappa(j, kappa):
            return np.exp(-kappa*j);

        # fit exponential
        colors = ["tab:blue", "tab:orange"];
        expfig, expax = plt.subplots();
        well_width = 11;
        exp_j = np.real(jvals[abs(jvals)<well_width//2]);
        exp_j = exp_j - exp_j[0];
        for alpha in range(n_loc_dof):
            exp_wf_a = np.real(psimas[alpha, which_m][1::n_loc_dof][abs(jvals)<well_width//2]);
            exp_wf_a = exp_wf_a/exp_wf_a[0];
            E_a = np.round(np.real(Emas[alpha, which_m])+2.0, decimals=10);
            expax.plot(exp_j, exp_wf_a, color=colors[alpha], label = "$|k_{m="+str(which_m)+"}, \\alpha = "+str(alpha)+"\\rangle (\\varepsilon = "+str(E_a)+")$");

            # interpolate kappa
            kappa_a = -np.log(exp_wf_a[1]/exp_wf_a[0]);
            V_kappa = kappa_a*kappa_a + E_a # in units of t
            exp_fit_a = np.real(np.exp(-kappa_a*exp_j));
            expax.scatter(exp_j, exp_fit_a, color=colors[alpha], marker='+', label = "$\kappa = "+str(kappa_a)+", V_{eff} = $"+str(V_kappa));

            # fit kappa
            if False:
                import scipy
                from scipy import optimize
                kappa_fitted, _ = scipy.optimize.curve_fit(fit_kappa, exp_j, exp_wf_a, p0=[kappa_a], bounds =[[0],[3*kappa_a]], verbose=2);
                V_kappa_fitted = kappa_fitted[0]*kappa_fitted[0] + E_a # in units of t
                exp_fitted = np.real(np.exp(-kappa_a*exp_j));
                expax.scatter(exp_j, exp_fitted, color=colors[alpha], marker='x', label = "$\kappa = "+str(kappa_fitted[0])+", V_{eff} = $"+str(V_kappa_fitted));
        plt.legend();
        plt.show();

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


#####################################################################################################
#### run code

if __name__ == "__main__":

    pass;
    

