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

def kernel(tinfty, tL, tLprime, tR, tRprime, Vinfty, VL, VLprime, VR, VRprime, Ninfty, NL, NR, HC,HCprime,E_cutoff=1.0,verbose=0) -> tuple:
    '''
    Calculate a transmission probability for each left well bound state
    as a function of the bound state energies

    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR

    This kernel requires the initial and final states to have definite spin,
    and so CAN RESOLVE the spin -> spin transitions

    Optional args:
    -E_cutoff, float, don't calculate T for eigenstates with energy higher 
        than this. That way we limit to bound states
    '''
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
    HL_4d, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HL_4d),n_loc_dof));
    Emas, psimas = [], []; # will index as Emas[alpha,m]
    n_bound_left = 0;
    interval = 3;
    for alpha in range(n_loc_dof):
        Ems, psims = np.linalg.eigh(HL_4d[:,:,alpha,alpha]);
        psims = psims.T[Ems+2*tLa[alpha] < E_cutoff];
        Ems = Ems[Ems+2*tLa[alpha] < E_cutoff];
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
    Emas, psimas = Emas_arr, psimas_arr # shape is (n_loc_dof, n_bound_left)
    kmas = np.arccos((Emas-fci_mod.scal_to_vec(VLa,n_bound_left))
                    /(-2*fci_mod.scal_to_vec(tLa,n_bound_left))); # wavenumbers in the left well
    
    # right well eigenstates  
    HR_4d, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HR_4d),n_loc_dof));
    Enbs, psinbs = [], []; # will index as Enbs[beta,n]
    n_bound_right = 0;
    for beta in range(n_loc_dof):
        Ens, psins = np.linalg.eigh(HR_4d[:,:,beta,beta]);
        psins = psins.T[Ens+2*tRa[alpha] < E_cutoff];
        Ens = Ens[Ens+2*tRa[alpha] < E_cutoff];
        Enbs.append(Ens.astype(complex));
        psinbs.append(psins);
        n_bound_right = max(n_bound_right, len(Ens));
    Enbs_arr = np.empty((n_loc_dof,n_bound_right), dtype = complex); # make un-ragged
    psinbs_arr = np.empty((n_loc_dof,n_bound_right,n_spatial_dof), dtype = complex);
    for alpha in range(n_loc_dof):# un-ragged the array by filling in highest Es
        Ens = Enbs[alpha];
        Ens_arr = np.append(Ens, np.full((n_bound_right-len(Ens),), Ens[-1]));
        Enbs_arr[alpha] = Ens_arr;
        psins = psinbs[alpha];
        psins_arr = np.append(psins, np.full((n_bound_right-len(Ens),n_spatial_dof), psins[-1]),axis=0);
        psinbs_arr[alpha] = psins_arr;
    Enbs, psinbs = Enbs_arr, psinbs_arr # shape is (n_loc_dof, n_bound_right)
    knbs = np.arccos((Enbs-fci_mod.scal_to_vec(VRa,n_bound_right))
                    /(-2*fci_mod.scal_to_vec(tRa,n_bound_right))); # wavenumbers in the left well

    # operator
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);

    # visualize
    jvals = np.array(range(len(Hsys_4d))) + offset;
    if(verbose > 9):
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL_4d,HR_4d,Hsys_4d,Hsys_4d-HL_4d,Hsys_4d-HR_4d];
            Hstrs = ["HL","HR","Hsys","Hsys-HL","Hsys-HR"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(jvals, Hi*0.001+np.diag(Hs[Hi][:,:,alpha,alpha]),label = Hstrs[Hi]);
        plt.legend();plt.show();assert False;

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Tbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):

            if False:
                print("******** E(",alpha,m,") = ",Emas[alpha,m]," ********")
                psima = psimas[:,m]
                fig, axes = plt.subplots(n_loc_dof);
                for sigma in range(n_loc_dof):
                    axes[sigma].plot(jvals, np.real(psima[sigma]));
                    axes[sigma].plot(jvals, np.imag(psima[sigma]),linestyle='dashed');
                    axes[sigma].set_xlim(0,5); axes[sigma].set_ylim(-0.05,0);
                plt.show();
                if(m>5): assert False;
                
            # final spin states
            myfig, myaxes = plt.subplots(2, sharex=True)
            for beta in range(n_loc_dof):
                # inelastic means averaging over an interval
                Mbmas = [];
                interval_width = abs(Enbs[alpha,-2]-Enbs[alpha,-1]);
                interval_width = 4 #1e-9;
                if(n_bound_left == n_bound_right):
                    if(not np.any(Enbs-Emas)): interval_width = 1e-9;
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval_width/2):
                        melement = matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]);
                        Mbmas.append(np.real(melement*np.conj(melement)));

                myaxes[beta].scatter(2+Enbs[beta],Mbmas);
                myaxes[beta].set_title(str(alpha)+"$\\rightarrow$"+str(beta)+" (N="+str(len(Mbmas))+")");
            #myaxes[-1].set_xscale('log', subs = []);
            plt.show();
            assert False;
            if False:

                # update T based on average
                print(interval_width, len(Mbma));
                if Mbmas: Mbmas = sum(Mbmas)/len(Mbmas);
                else: Mbmas = 0.0;
                Tbmas[beta,m,alpha] = NL/(kmas[alpha,m]*tLa[alpha]) *NR/(kmas[alpha,m]*tRa[alpha]) *Mbmas;

    return Emas, Tbmas;

def kernel_constructed(tinfty, tL, tLprime, tR, tRprime, Vinfty, VL, VLprime, VR, VRprime, Ninfty, NL, NR, HC,HCprime,E_cutoff=-1.9,verbose=0) -> tuple:
    '''
    Calculate a transmission probability for each left well bound state
    as a function of the bound state energies

    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR

    intended to RESOLVE initial and final spin
    simply posits a normalized plane wave with desired spin

    Optional args:
    -E_cutoff, float, don't calculate T for eigenstates with energy higher 
        than this. That way we limit to bound states
    '''
    if(np.shape(HC) != np.shape(HCprime)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];

    # convert from matrices to _{alpha alpha} elements
    to_convert = [tL, VL, tR, VR, Vinfty];
    converted = [];
    for convert in to_convert:
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ):
            raise ValueError; # VL must be diag
        converted.append(np.diagonal(convert));
    tLa, VLa, tRa, VRa, Vinfa = tuple(converted);

    # left well 
    HL_4d, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime, bound=False);
    HL = fci_mod.mat_4d_to_2d(HL_4d);
    # left well eigenstates
    Emas, _ = np.linalg.eigh(HL);
    Emas = Emas[Emas < E_cutoff].astype(complex);
    n_bound_left = len(Emas)//n_loc_dof;
    Emas = np.reshape(Emas, (n_bound_left, n_loc_dof)).T;
    #n_bound_left = 1000
    #Emas = np.array([np.linspace(-2.0,-1.0,n_bound_left),np.linspace(-2.0,-1.0,n_bound_left)], dtype=complex);
    kmas = np.arccos((Emas-fci_mod.scal_to_vec(VLa,n_bound_left))
                    /(-2*fci_mod.scal_to_vec(tLa,n_bound_left))); # wavenumbers in left well 
    kapmas = np.arccosh((-Emas+fci_mod.scal_to_vec(Vinfa,n_bound_left))
                    /(-2*fci_mod.scal_to_vec(tLa,n_bound_left))); # wavenumbers in left well 
    fig, axes = plt.subplots(2)
    axes[0].plot(Emas[0], kmas[0]);
    axes[0].plot(Emas[0], np.sqrt( (Emas[0]+2*tLa[0]-VLa[0])/2*tLa[0]) );
    axes[1].plot(Emas[0],kapmas[0]);
    axes[1].plot(Emas[0], np.sqrt( (Vinfa[0] - (Emas[0]+2*tLa[0]))/2*tLa[0]) );
    #plt.show();
    #assert False
    # right well 
    HR_4d, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime, bound=False);
    HR = fci_mod.mat_4d_to_2d(HR_4d);
    # right well eigenstates
    Enbs, _ = np.linalg.eigh(HR);
    Enbs = Enbs[Enbs < E_cutoff].astype(complex);
    n_bound_right = len(Enbs)//n_loc_dof;
    Enbs = np.reshape(Enbs, (n_bound_right, n_loc_dof)).T;
    knbs = np.arccos((Enbs-fci_mod.scal_to_vec(VRa,n_bound_right))
                    /(-2*fci_mod.scal_to_vec(tRa,n_bound_right))); # wavenumbers in right well 

    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC, bound=False);
    jvals = np.array(range(len(Hsys_4d))) + offset;
    if(verbose > 9):   # plot the potential
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL_4d,HR_4d,Hsys_4d,Hsys_4d-HL_4d,Hsys_4d-HR_4d]; Hstrs = ["HL","HR","Hsys","Hsys-HL","Hsys-HR"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(np.real(jvals), np.real(Hi*1e-4+np.diag(Hs[Hi][:,:,alpha,alpha])),label = Hstrs[Hi]);
        plt.legend();plt.show();      

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Tbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    # need ham with no left barrier and left lead self energies
    HLself_4d_base, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime); 
    for m in range(n_bound_left):
        for alpha in range(n_loc_dof):
            print("******** E(",alpha,m,") = ",Emas[alpha,m]," ********");
            kma_js = np.lib.scimath.arccos((Emas[alpha,m]-np.diag(HL_4d[:,:,alpha,alpha]))/(-2*tLa[alpha]));
            psima = np.exp(jvals*kma_js*complex(0,1));
            psima = psima/np.dot(psima,np.conj(psima));
            psima = np.array((psima,)*n_loc_dof);
            ####
            #### problem: exponential decay hardly changes with energy compared
            #### to HL eigenfunctions in regular kernel
            #### looking at the latter, see psi(j=0) varies a lot with energy
            #### by definition not the case here since psi ~ e^(-\kappa j)
            #### as a consequence of continuity with a propagating wave
            ####
            psim_prop = np.exp(jvals*kmas[alpha,m]*complex(0,1));
            psim_dec = np.exp(jvals*kapmas[alpha,m]*(-1));
            psim_dec = [el if abs(el) <= 1.0 else 0.0 for el in psim_dec];
            if(verbose > 9):
                fig, axes = plt.subplots(n_loc_dof);
                for sigma in range(n_loc_dof):
                    axes[sigma].plot(jvals, np.real(psima[sigma]));
                    axes[sigma].plot(jvals, np.imag(psima[sigma]),linestyle='dashed');
                    axes[sigma].plot(jvals, np.real(psim_dec) );
                    axes[sigma].set_xlim(0,5); axes[sigma].set_ylim(0,0.05);
                plt.show();
                if(m>5): assert False;

            # final spin states
            for beta in range(n_loc_dof):
                # average over final state energy
                Mbma = 0.0;

                # inelastic means averaging over an interval
                Nbma = 0; # num states in the interval
                interval_width = abs(Enbs[alpha,-2]-Enbs[alpha,-1]);
                interval_width = 1e-9;
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval_width/2):
                        Nbma += 1;

                        # construct psinb and get matrix element
                        knb_js = -np.lib.scimath.arccos((Enbs[beta,n]-np.diag(HR_4d[:,:,beta,beta]))/(-2*tRa[beta]));
                        psinb = np.exp(jvals*knb_js*complex(0,1));
                        psinb = psinb/np.dot(psinb, np.conj(psinb));
                        psinb = np.array((psinb,)*n_loc_dof);
                        if(verbose>9):
                            fig, axes = plt.subplots(n_loc_dof);
                            for sigma in range(n_loc_dof):
                                axes[sigma].plot(jvals, np.real(psinb[sigma]));
                                axes[sigma].plot(jvals, np.imag(psinb[sigma]),linestyle='dashed');
                            plt.show();
                            if(n>5): assert False
                        melement = matrix_element(beta,psinb,Hdiff,alpha,psima);
                        Mbma += np.real(melement*np.conj(melement));

                # update T based on average
                print(interval_width, Nbma);
                if Nbma == 0: Mbma = 0.0;
                else: Mbma = Mbma / Nbma;
                Tbmas[beta,m,alpha] = NL/(kmas[alpha,m]*tLa[alpha]) *NR/(kmas[alpha,m]*tRa[alpha]) *Mbma;

    return Emas, Tbmas;

def kernel_mixed(tinfty, tL, tLprime, tR, tRprime, Vinfty, VL, VLprime, VR, VRprime, Ninfty, NL, NR, HC,HCprime,E_cutoff=1.0,verbose=0) -> tuple:
    '''
    Calculate a transmission probability for each left well bound state
    as a function of the bound state energies

    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR

    This kernel allows the initial and final states to lack definite spin,
    but as a result CANNOT RESOLVE the spin -> spin transitions

    Optional args:
    -E_cutoff, float, don't calculate T for eigenstates with energy higher 
        than this. That way we limit to bound states
    '''
    if(np.shape(HC) != np.shape(HCprime)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];

    # convert from matrices to spin-diagonal, spin-independent elements
    to_convert = [tL, VL, tR, VR];
    converted = [];
    for convert in to_convert:
        # check spin-diagonal
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ): raise ValueError("not spin diagonal"); 
        # check spin-independent
        diag = np.diagonal(convert);
        if(np.any(diag-diag[0])): raise ValueError("not spin independent");
        converted.append(convert[0,0]);
    tLa, VLa, tRa, VRa = tuple(converted);

    # left well 
    HL_4d, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    HL = fci_mod.mat_4d_to_2d(HL_4d);
    interval = 2;
    interval_tup = (n_loc_dof*(n_spatial_dof//2-interval),n_loc_dof*(n_spatial_dof//2+interval+1) );
    if verbose: print("-HL[:,:] near barrier =\n",np.real(HL[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
    # left well eigenstates
    Ems, psims = np.linalg.eigh(HL);
    psims = psims.T[Ems < E_cutoff];
    Ems = Ems[Ems < E_cutoff].astype(complex);
    n_bound_left = len(Ems);
    kms = np.arccos((Ems-VLa)/(-2*tLa)); # wavenumbers in the left well

    # get Sx for each psim
    Sxms = np.zeros_like(Ems);
    Sx_op = np.zeros((len(psims[0]),len(psims[0]) ),dtype=complex);
    for eli in range(len(Sx_op)-1): Sx_op[eli,eli+1] = 1.0; Sx_op[eli+1,eli] = 1.0;
    for m in range(len(psims)):
        Sxms[m] = np.dot( np.conj(psims[m]), np.dot(Sx_op, psims[m]));
    
    # right well 
    HR_4d, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    HR = fci_mod.mat_4d_to_2d(HR_4d);
    if verbose: print("-HR[:,:] near barrier =\n",np.real(HR[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
    
    # right well eigenstates
    Ens, psins = np.linalg.eigh(HR);
    psins = psins.T[Ens < E_cutoff];
    Ens = Ens[Ens < E_cutoff].astype(complex);
    n_bound_right = len(Ens);
    knbs = np.arccos((Ens-VRa)/(-2*tRa)); # wavenumbers in the right well

    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    jvals = np.array(range(len(Hsys_4d))) + offset;
    if(verbose > 9):
        # plot the potential
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL_4d,HR_4d,Hsys_4d,Hsys_4d-HL_4d,Hsys_4d-HR_4d]; Hstrs = ["HL","HR","Hsys","Hsys-HL","Hsys-HR"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(np.real(jvals), np.real(Hi*1e-4+np.diag(Hs[Hi][:,:,alpha,alpha])),label = Hstrs[Hi]);
        plt.legend();plt.show();
        # plot the wfs
        for m in range(6):
            plot_wfs(HL_4d, Ems[m], 0, E_cutoff, E_tol=1e-9, fourier = False);
        assert False;

    # average matrix elements over final states |k_n >
    # with the same energy as the intial state |k_m >
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mnms = np.empty((n_bound_right,n_bound_left),dtype=float);
    Tms = np.empty((n_bound_left,),dtype=float);
    for m in range(n_bound_left):

        # average over final state energy
        Mm = 0.0;

        # inelastic means averaging over an interval
        Nm = 0; # num states in the interval
        interval_width = abs(Ems[-2]-Ems[-1]);
        interval_width = 1e-9;
        if(n_bound_left == n_bound_right):
            if(not np.any(Ens-Ems)): interval_width = 1e-9;       
        for n in range(n_bound_right):
            if( abs(Ems[m] - Ens[n]) < interval_width/2):
                Nm += 1;
                melement = np.dot(np.conj(psins[n]), np.dot(Hdiff,psims[m]));
                Mm += np.real(melement*np.conj(melement));
                print(interval_width, Nm);
                if False:
                    print("-Hdiff[:,:] near barrier =\n",np.real(Hdiff[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
                    print("- psim near barrier\n",np.real(psims[m,interval_tup[0]:interval_tup[1]]));
                    print("- psin near barrier\n",np.real(psins[n,interval_tup[0]:interval_tup[1]]));
                    print("- barrier overlap of psim and psin: ",np.dot( psims[m,interval_tup[0]:interval_tup[1]][-4:],psins[n,interval_tup[0]:interval_tup[1]][-4:]));
                    assert False

        # update T based on average
        #print(interval_width, Nm, psims[m]);
        if(Nm == 0): Mm = 0.0;
        else: Mm = Mm/Nm;
        Tms[m] = NL/(kms[m]*tLa) *NR/(kms[m]*tRa) *Mm;

    return Ems, Tms, Sxms;

def kernel_fourier(tinfty, tL, tLprime, tR, tRprime, Vinfty, VL, VLprime, VR, VRprime, Ninfty, NL, NR, HC,HCprime,E_cutoff=1.0,verbose=0) -> tuple:
    '''
    Calculate a transmission probability for each left well bound state
    as a function of the bound state energies

    Physical params are classified by region: infty, L, R.
    tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC are as in Hsysmat
    docstring below. Primed quantities represent the values given to
    the unperturbed Hamiltonians HL and HR

    This kernel allows the initial and final states to lack definite spin,
    but then fourier expands to separate right and left going parts by spin
    and so CAN RESOLVE the spin -> spin transitions

    Optional args:
    -E_cutoff, float, don't calculate T for eigenstates with energy higher 
        than this. That way we limit to bound states
    '''
    if(np.shape(HC) != np.shape(HCprime)): raise ValueError;
    n_spatial_dof = Ninfty+NL+len(HC)+NR+Ninfty;
    n_loc_dof = np.shape(HC)[-1];

    # convert from matrices to spin-diagonal, spin-independent elements
    to_convert = [tL, VL, tR, VR];
    converted = [];
    for convert in to_convert:
        # check spin-diagonal
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ): raise ValueError("not spin diagonal"); 
        # check spin-independent
        diag = np.diagonal(convert);
        if(np.any(diag-diag[0])): raise ValueError("not spin independent");
        converted.append(convert[0,0]);
    tLa, VLa, tRa, VRa = tuple(converted);

    # left well 
    HL_4d, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    HL = fci_mod.mat_4d_to_2d(HL_4d);
    interval = 2;
    interval_tup = (n_loc_dof*(n_spatial_dof//2-interval),n_loc_dof*(n_spatial_dof//2+interval+1) );
    if verbose: print("-HL[:,:] =\n",np.real(HL[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
    # left well eigenstates
    Ems, psims = np.linalg.eigh(HL);
    psims = psims.T[Ems < E_cutoff];
    Ems = Ems[Ems < E_cutoff].astype(complex);
    n_bound_left = len(Ems);
    kms = np.arccos((Ems-VLa)/(-2*tLa)); # wavenumbers in the left well
    
    # right well 
    HR_4d, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    HR = fci_mod.mat_4d_to_2d(HR_4d);
    if verbose: print("-HR[:,:] =\n",np.real(HR[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
    
    # right well eigenstates
    Ens, psins = np.linalg.eigh(HR);
    psins = psins.T[Ens < E_cutoff];
    Ens = Ens[Ens < E_cutoff].astype(complex);
    n_bound_right = len(Ens);
    knbs = np.arccos((Ens-VRa)/(-2*tRa)); # wavenumbers in the right well


    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    jvals = np.array(range(len(Hsys_4d))) + offset;
    if(verbose > 9):
        # plot the potential
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL_4d,HR_4d,Hsys_4d,Hsys_4d-HL_4d,Hsys_4d-HR_4d]; Hstrs = ["HL","HR","Hsys","Hsys-HL","Hsys-HR"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(np.real(jvals), np.real(Hi*1e-4+np.diag(Hs[Hi][:,:,alpha,alpha])),label = Hstrs[Hi]);
        plt.legend();plt.show();
        # plot the wfs
        for m in range(2):
            plot_wfs(HL_4d, Ems[m], 0, E_cutoff, E_tol=1e-9);      

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Tbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    # initial energy and spin states
    for m in range(n_bound_left):
        for alpha in range(n_loc_dof):
                
            # final spin states
            for beta in range(n_loc_dof):
                # average over final state energy
                Mbma = 0.0;

                # inelastic means averaging over an interval
                Nbma = 0; # num states in the interval
                interval_width = abs(Ens[-2]-Ens[-1]);
                interval_width = 1e-9;
                if(n_bound_left == n_bound_right):
                    if(not np.any(Ens-Ems)): interval_width = 1e-9;
                for n in range(n_bound_right):
                    if( abs(Ems[m] - Ens[n]) < interval_width/2):
                        Nbma += 1;
                        
                        # convert psim and psin
                        if False:
                            psim = psims[m].reshape(n_spatial_dof, n_loc_dof).T;
                            psim4d = fourfold_decompose(psim, ninf=100);
                            psin = psins[n].reshape(n_spatial_dof, n_loc_dof).T;
                            psin4d = fourfold_decompose(psim);
                            for sigma in range(n_loc_dof):
                                fig, ax = plt.subplots();
                                ax.plot(jvals,psim[alpha]);
                                ax.plot(jvals,psim4d[0,sigma]+psim4d[1,sigma]);
                                ax.set_xlim(0,20); ax.set_ylim(-0.01,0.0);
                                plt.show();

                            assert False

                        # get matrix element
                        psi_i = np.array([psims[m][eli] if eli % n_loc_dof == alpha else 0.0 for eli in range(len(psims[m]))]);
                        psi_f = np.array([psims[n][eli] if eli % n_loc_dof == beta else 0.0 for eli in range(len(psins[n]))]);
                        melement = np.dot(np.conj(psi_f), np.dot(Hdiff,psi_i));
                        if(m==n_bound_left-1 and is_alpha_conserving(Hdiff, n_loc_dof)): print("WARNING: overlap of different spin components is strictly zero without off diagonal elements in Hdiff");
                        Mbma += np.real(melement*np.conj(melement));

                        if False:
                            if(m>20): assert False
                            print("--->",alpha,beta);
                            print(sum(psi_i.reshape(n_spatial_dof, n_loc_dof)[:,0]),sum(psi_i.reshape(n_spatial_dof, n_loc_dof)[:,1]) );
                            print(sum(psi_f.reshape(n_spatial_dof, n_loc_dof)[:,0]),sum(psi_f.reshape(n_spatial_dof, n_loc_dof)[:,1]) );
                            print(melement)

                # update T based on average
                print(interval_width, Nbma);
                if Nbma == 0: Mbma = 0.0;
                else: Mbma = Mbma / Nbma;
                Tbmas[beta,m,alpha] = NL/(kms[m]*tLa) *NR/(kms[m]*tRa) *Mbma;

    return np.array([Ems, Ems]), Tbmas;

def benchmark(tL, tR, VL, VR, HC, Emas, verbose=0) -> np.ndarray:
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

def benchmark_mixed(tL, tR, VL, VR, HC_4d, Ems, verbose=0) -> np.ndarray:
    '''
    Given bound state energies and HC from kernel, calculate the transmission
    probability for each energy using wfm code

    Used when the initial and final states don't definite spin,
    and so CANNOT RESOLVE the spin -> spin transitions
    '''
    if(np.any(tL-tR)): raise NotImplementedError; # wfm code can't handle this case
    n_spatial_dof = np.shape(HC_4d)[0];
    n_loc_dof = np.shape(HC_4d)[-1];
    n_bound_left = np.shape(Ems)[-1];

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
                hblocks[1+spacei] = HC_4d[spacei,spacej];
            elif(spacei == spacej - 1): # nn hopping
                tnn[1+spacei] = HC_4d[spacei,spacej];
            elif(spacei == spacej - 2): # next nn hopping
                tnnn[1+spacei] = HC_4d[spacei,spacej];
            elif(spacei < spacej):
                assert(not np.any(HC_4d[spacei,spacej]));
    hblocks[-1] = VR*np.eye(n_loc_dof);
    tnn[-1] = -tR*np.eye(n_loc_dof);
    tnnn[-1] = 0.0*np.eye(n_loc_dof);
    if(verbose > 9):
        print(hblocks);
        print(tnn);
        print(tnnn);
        assert False;

    # got total transmission prob for a given initial state
    Tmas = np.empty((n_loc_dof,n_bound_left),dtype=float);
    Tms = np.empty_like(Ems);
    for m in range(n_bound_left):
        for alpha in range(n_loc_dof):
            source = np.zeros((n_loc_dof,));
            source[alpha] = 1.0;
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tL[alpha,alpha], Ems[m], source, verbose = verbose);
            Tmas[alpha,m] = np.sum(Tdum);
        # sum over initial spin states
        Tms[m] = np.sum(Tmas[:,m])/n_loc_dof;
            
    return Tms; 
    
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

    else: raise Exception; # not supported

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

def fourfold_decompose(psi2d, ninf=10):
    '''
    decompose a wf into up right going, up left going, down right, down left parts
    '''
    if(not isinstance(psi2d, np.ndarray)): raise TypeError;
    if(not isinstance(ninf,int)): raise TypeError;
    n_loc_dof, n_spatial_dof = np.shape(psi2d);
    assert(n_spatial_dof % 2 == 1);

    # 4d wf
    psi4d = np.zeros((2,n_loc_dof, n_spatial_dof));
    cmas = np.empty((n_loc_dof,2*ninf+1),dtype=complex);
    for alpha in range(n_loc_dof):

        # fourier expand
        psi = psi2d[alpha];
        L = n_spatial_dof // 2;
        jvals = np.array(range(-L,L+1));
        psi_right, psi_left = np.zeros_like(psi), np.zeros_like(psi);

        # get fourier coef c_n for n \in {-ninf,...+ninf}
        for n in range(-ninf,ninf+1):
            cmas[alpha,n] = np.trapz(psi*np.exp(-complex(0,1)*n*np.pi*jvals/L)/(2*L), jvals);

        # break up into right moving and left moving parts
        for n in range(-ninf,ninf+1):
            if(n>0): # right moving
                psi_right += cmas[alpha,n]*np.exp(complex(0,1)*n*np.pi*jvals/L);
            elif(n<=0): # left moving
                psi_left += cmas[alpha,n]*np.exp(complex(0,1)*n*np.pi*jvals/L);

        # put together
        psi4d[0,alpha] = psi_right;
        psi4d[1,alpha] = psi_left;

    return psi4d;
 

def get_fourier_coefs(wf_full,n_loc_dof,ninf=10) -> np.ndarray:
    '''
    Get the comple fourier coefficients for a wf
    '''
    if(not isinstance(wf_full, np.ndarray)): raise TypeError;
    if(not isinstance(ninf,int)): raise TypeError;
    n_spatial_dof = len(wf_full) // n_loc_dof;
    assert(n_spatial_dof % 2 == 1);

    # decompose left and right
    mycolors= ['tab:blue','tab:orange','tab:green','tab:red']; # differentiates spin comps
    mystyles=['solid','dashed']; # differentiates real vs imaginary

    # color by spin
    cmas = np.empty((n_loc_dof,2*ninf+1),dtype=complex);
    for alpha in range(n_loc_dof):
        figtest, (axright, axleft, axcomp) = plt.subplots(3, sharex=True);
        wf = wf_full[alpha::n_loc_dof];
        L = n_spatial_dof // 2;
        jvals = np.array(range(-L,L+1));

        # get fourier coef c_n for n \in {-ninf,...+ninf}
        for n in range(-ninf,ninf+1):
            cmas[alpha,n] = np.trapz(wf*np.exp(-complex(0,1)*n*np.pi*jvals/L)/(2*L), jvals);

        # break up into right moving and left moving parts
        wf_right, wf_left = np.zeros_like(wf), np.zeros_like(wf);
        for n in range(-ninf,ninf+1):
            if(n>0): # right moving
                wf_right += cmas[alpha,n]*np.exp(complex(0,1)*n*np.pi*jvals/L);
            elif(n<=0): # left moving
                wf_left += cmas[alpha,n]*np.exp(complex(0,1)*n*np.pi*jvals/L);

        if False:
            axright.plot(jvals,np.real(wf),linestyle='solid',color='tab:blue');
            axright.plot(jvals,np.imag(wf),linestyle='dashed',color='tab:blue');
            axleft.plot(jvals,np.real(wf_left+wf_right),linestyle='solid',color='tab:blue');
            axleft.plot(jvals,np.imag(wf_left+wf_right),linestyle='dashed',color='tab:blue');
            plt.show();
            assert False

        else:
            axright.plot(jvals, np.real(wf_right),linestyle=mystyles[0],color=mycolors[alpha]);
            axright.plot(jvals, np.imag(wf_right),linestyle=mystyles[1],color=mycolors[alpha]);
            axleft.plot(jvals, np.real(wf_left),linestyle=mystyles[0],color=mycolors[alpha]);
            axleft.plot(jvals, np.imag(wf_left),linestyle=mystyles[1],color=mycolors[alpha]);
            axcomp.plot(jvals, np.real(np.append(wf_right[1:],wf_right[0])/wf_right),linestyle=mystyles[0],color=mycolors[alpha]);
            axcomp.plot(jvals, np.imag(np.append(wf_right[1:],wf_right[0])/wf_right),linestyle=mystyles[1],color=mycolors[alpha]);

        # show
        axright.set_ylabel("Right going");
        axright.set_title("Bound state Fourier decomposition");
        axleft.set_ylabel("Left going");
        axcomp.set_ylabel("Right going $\psi_{j+1}/\psi_j$");
        axcomp.set_ylim(-2,2);
        plt.show();
    
    return cmas;

def plot_wfs(H, E0, alpha0, E_cutoff, E_tol = 1e-2, fourier = True) -> None:
    '''
    '''
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
    mycolors= ['tab:blue','tab:orange','tab:green','tab:red']; # differentiates spin comps
    mystyles=['solid','dashed']; # differentiates real vs imaginary
    print("Plotting bound states with absorbing/emitting bcs.");

    # look for the bound state coupled to the continuum
    V_cont, t_cont = np.real(H[0,0,alpha0,alpha0]), -np.real(H[0,1,alpha0,alpha0]);
    ks = np.arccos((Es - V_cont)/(-2*t_cont));
    k_cont = np.arccos((E0 - V_cont)/(-2*t_cont));
    print("continuum E, k =",E0, k_cont)
    coupled_continuum = False;
    for m in range(len(Es)):
        psim = psis[m];
        print(m,Es[m].round(4),ks[m].round(4));

        # only plot the coupled state
        if(abs(np.real(Es[m]-E0)) < E_tol):
            coupled_continuum = True;
            Szm = np.dot(np.conj(psim),np.dot(Sz_op,psim));
            Sxm = np.dot(np.conj(psim),np.dot(Sx_op,psim));

            # plot spin components in different colors
            myfig, (hamax, wfax, derivax) = plt.subplots(3);
            for alpha in range(n_loc_dof):
                # plot ham
                hamax.plot(jvals,np.real(np.diag(H[:,:,alpha,alpha])),color=mycolors[alpha],linestyle=mystyles[0]);
                hamax.plot(jvals,np.imag(np.diag(H[:,:,alpha,alpha])),color=mycolors[alpha],linestyle=mystyles[1]);
                # plot wf
                psimup = psim[alpha::n_loc_dof];
                wfax.plot(np.real(jvals), 1e-6*alpha+np.real(psimup),color=mycolors[alpha],linestyle=mystyles[0]);
                wfax.plot(np.real(jvals), 1e-6*alpha+np.imag(psimup),color=mycolors[alpha],linestyle=mystyles[1]);
                derivax.plot(np.real(jvals), 1e-6*alpha+np.real(np.append(psimup[1:],psimup[0])/psimup),color=mycolors[alpha],linestyle=mystyles[0]);
                derivax.plot(np.real(jvals), 1e-6*alpha+np.imag(np.append(psimup[1:],psimup[0])/psimup),color=mycolors[alpha],linestyle=mystyles[1]); 

            # show
            hamax.set_ylabel('$H_L$');
            wfax.set_ylabel('$\psi$');
            wfax.set_title("Bound state: <S_z> = "+str(int(1000*Szm)/1000)+", <S_x> = "+str(int(1000*Sxm)/1000));
            derivax.set_ylabel("$\psi_{j+1}/\psi_j$");
            derivax.set_ylim(-2,2)
            plt.tight_layout();
            plt.show();

            # fourier decomp
            if fourier: get_fourier_coefs(psim,n_loc_dof);

    # check
    if(not coupled_continuum): raise Exception("bound state energy not coupled to continuum");

#####################################################################################################
#### run code

if __name__ == "__main__":

    pass;
    

