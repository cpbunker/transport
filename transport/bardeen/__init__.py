'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

from transport import fci_mod, wfm

import numpy as np
import matplotlib.pyplot as plt

####
#### spin flip has to happen through the matrix element !!!
####

##################################################################################
#### driver of transmission coefficient calculations

def kernel(tinfty, tL, tLprime, tR, tRprime,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC,HCprime,
           interval=1e-9,E_cutoff=1.0,HT_perturb=False,verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nbma averaged over n in a
    nearby interval

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
    del Ems, psims
    Emas, psimas = Emas_arr, psimas_arr # shape is (n_loc_dof, n_bound_left)

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
    del Ens, psins;
    Enbs, psinbs = Enbs_arr, psinbs_arr # shape is (n_loc_dof, n_bound_right)

    # operator
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);

    # filter left and right
    jvals = np.array(range(len(Hsys_4d))) + offset;
    mid = len(jvals) // 2;
    if(HT_perturb):
        for alpha in range(n_loc_dof):
            for m in range(n_bound_left):
                psim = psimas[alpha,m];
                weight_left = np.dot( np.conj(psim[:mid]), psim[:mid]);
                weight_right = np.dot( np.conj(psim[mid:]), psim[mid:]);
                if(weight_left < weight_right):
                    Emas[alpha,m] = 0.0;
                    psimas[alpha,m] = np.zeros_like(psim);
        for beta in range(n_loc_dof):
            for n in range(n_bound_right):
                psin = psinbs[beta,n];
                weight_left = np.dot( np.conj(psin[:mid]), psin[:mid]);
                weight_right = np.dot( np.conj(psin[mid:]), psin[mid:]);
                if(weight_left > weight_right):
                    Enbs[beta,n] = 0.0
                    psinbs[beta,n] = np.zeros_like(psin);

    # visualize
    if(verbose > 9):

        # plot hams
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL_4d,HR_4d,Hsys_4d,Hsys_4d-HL_4d,Hsys_4d-HR_4d];
            Hstrs = ["$H_L$","$H_R$","$H_{sys}$","$H_{sys}-H_L$","$H_{sys}-H_{R}$"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(jvals, Hi*0.001+np.diag(Hs[Hi][:,:,alpha,alpha]),label = Hstrs[Hi]);
            myaxes[alpha].set_xlabel("$j$"); myaxes[alpha].set_ylabel("$V_j$");
        plt.legend();plt.show();

        # plot left wfs
        energy_off = 0;
        if HT_perturb: energy_off = 1;
        for m in range(min(n_bound_left,6)):
            fig, wfax = plt.subplots();
            alpha_colors=["tab:blue","tab:orange"];
            for alpha in range(n_loc_dof):
                print(Emas[alpha,m]);
                print(Enbs[alpha,m+energy_off]);
                wfax.set_title("Left: "+str(Emas[alpha,m].round(4)));
                wfax.plot(jvals,np.diag(HL_4d[:,:,alpha,alpha]),color="black");
                wfax.plot(jvals, np.real(psimas[alpha,m]),color=alpha_colors[alpha],linestyle="solid");
                wfax.plot(jvals, np.real(psinbs[alpha,m+1]),color=alpha_colors[alpha],linestyle="dotted");
                wfax.plot(jvals[mid-1:mid+len(HC)+1], np.matmul((Hsys_4d-HL_4d)[:,:,alpha,alpha], psimas[alpha,m])[mid-1:mid+len(HC)+1], color=alpha_colors[alpha],linestyle="solid",marker='s')               
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
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval/2):
                        melement = matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]);
                        Mns.append(np.real(melement*np.conj(melement)));

                        if False:
                            print(np.real(melement*np.conj(melement)));
                            fig, wfax = plt.subplots();
                            wfax.plot(jvals, np.matmul((Hsys_4d-HL_4d)[:,:,alpha,alpha], psimas[alpha,m]), color='red');
                            wfax.plot(jvals, np.real(psinbs[beta,n]),color="tab:orange");
                            wfax.plot(jvals, np.real(psimas[alpha,m]),color="tab:blue");
                            plt.show(); assert False;

                # update T based on average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;

    return Emas, Mbmas;

def kernel_mixed(tinfty, tL, tLprime, tR, tRprime,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC,HCprime,
           interval=1e-9,E_cutoff=1.0,HT_perturb=False,verbose=0) -> tuple:
    '''
    Calculate the Oppenheimer matrix elements M_nm averaged over n in a
    nearby interval
    
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
    my_interval = 2;
    my_interval_tup = (n_loc_dof*(n_spatial_dof//2-my_interval),n_loc_dof*(n_spatial_dof//2+my_interval+1) );
    if verbose: print("-HL[:,:] near barrier =\n",np.real(HL[my_interval_tup[0]:my_interval_tup[1],my_interval_tup[0]:my_interval_tup[1]]));
    # left well eigenstates
    Ems, psims = np.linalg.eigh(HL);
    psims = psims.T[Ems+2*tLa < E_cutoff];
    Ems = Ems[Ems+2*tLa < E_cutoff].astype(complex);
    n_bound_left = len(Ems);
    kms = np.arccos((Ems-VLa)/(-2*tLa)); # wavenumbers in the left well

    # get Sx val for each psim
    Sx_op = np.zeros((len(psims[0]),len(psims[0]) ),dtype=complex);
    for eli in range(len(Sx_op)-1): Sx_op[eli,eli+1] = 1.0; Sx_op[eli+1,eli] = 1.0;
    Sxms = np.zeros_like(Ems);
    for m in range(n_bound_left):
        Sxms[m] = np.dot( np.conj(psims[m]), np.dot(Sx_op, psims[m]));
    
    # right well 
    HR_4d, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    HR = fci_mod.mat_4d_to_2d(HR_4d);
    if verbose: print("-HR[:,:] near barrier =\n",np.real(HR[my_interval_tup[0]:my_interval_tup[1],my_interval_tup[0]:my_interval_tup[1]]));
    
    # right well eigenstates
    Ens, psins = np.linalg.eigh(HR);
    psins = psins.T[Ens+2*tRa < E_cutoff];
    Ens = Ens[Ens+2*tRa < E_cutoff].astype(complex);
    n_bound_right = len(Ens);
    kns = np.arccos((Ens-VRa)/(-2*tRa)); # wavenumbers in the right well
    # get Sx val for each psin
    Sxns = np.zeros_like(Ens);
    for n in range(n_bound_right):
        Sxns[n] = np.dot( np.conj(psins[n]), np.dot(Sx_op, psins[n]));

    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);

    # filter left and right
    jvals = np.array(range(len(Hsys_4d))) + offset;
    mid = len(jvals) // 2;
    if(HT_perturb):
        for m in range(n_bound_left):
            psim = psims[m];
            weight_left = np.dot( np.conj(psim[:2*mid]), psim[:2*mid]);
            weight_right = np.dot( np.conj(psim[2*mid:]), psim[2*mid:]);
            if(weight_left < weight_right):
                Ems[m] = 0.0;
                psims[m] = np.zeros_like(psim);
        for n in range(n_bound_right):
            psin = psins[n];
            weight_left = np.dot( np.conj(psin[:2*mid]), psin[:2*mid]);
            weight_right = np.dot( np.conj(psin[2*mid:]), psin[2*mid:]);
            if(weight_left > weight_right):
                Ens[n] = 0.0
                psins[n] = np.zeros_like(psin);


    # filter by Sx val
    Ems, psims = Ems[Sxms > 0], psims[Sxms > 0];
    n_bound_left = n_bound_left // 2;
    Ens, psins = Ens[Sxns > 0], psins[Sxns > 0];
    n_bound_right = n_bound_right // 2;

    # reshape
    print("n_spatial_dof, n_bound_left, n_bound_right = ",n_spatial_dof, n_bound_left, n_bound_right);
    Emas = np.array([np.copy(Ems), np.copy(Ems)]);
    Enbs = np.array([np.copy(Ens), np.copy(Ens)]);
    assert(n_loc_dof == 2);
    psimas = np.empty((n_loc_dof, n_bound_left, n_spatial_dof), dtype=complex);
    for m in range(n_bound_left):
        psimas[0,m] = psims[m,0::2];
        psimas[1,m] = psims[m,1::2];
    psinbs = np.empty((n_loc_dof, n_bound_right, n_spatial_dof), dtype=complex);
    for n in range(n_bound_right):
        psinbs[0,n] = psins[n,0::2];
        psinbs[1,n] = psins[n,1::2];
    print("psims -> psimas = ",np.shape(psims),np.shape(psimas));
    print("psins -> psinbs = ",np.shape(psins),np.shape(psinbs));
    print("E shapes = ",np.shape(Ems), np.shape(Emas), np.shape(Ens), np.shape(Enbs));
    del Ems, psims, Ens, psins;
    assert False
    
    # visualize
    if(verbose > 9):
        
        # plot hams
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL_4d,HR_4d,Hsys_4d,Hsys_4d-HL_4d,Hsys_4d-HR_4d]; Hstrs = ["HL","HR","Hsys","Hsys-HL","Hsys-HR"];
            for Hi in range(len(Hs)):
                print(Hstrs[Hi]);
                print(Hs[Hi][0-offset,0-offset]);
                myaxes[alpha].plot(np.real(jvals), np.real(Hi*1e-4+np.diag(Hs[Hi][:,:,alpha,alpha])),label = Hstrs[Hi]);
        plt.legend();plt.show();

        # plot left wfs
        for m in range(min(n_bound_left,6)):
            fig, wfax = plt.subplots();
            alpha_colors=["tab:blue","tab:orange"];
            for alpha in range(n_loc_dof):
                print(Emas[alpha,m]);
                print(Enbs[alpha,m+1]);
                wfax.set_title("Left: "+str(Emas[alpha,m].round(4)));
                wfax.plot(jvals,np.diag(HL_4d[:,:,alpha,alpha]),color="black");
                wfax.plot(jvals, alpha*0.01+np.real(psimas[alpha,m]),color=alpha_colors[alpha],linestyle="solid");
                wfax.plot(jvals, alpha*0.01+np.real(psinbs[alpha,m+1]),color=alpha_colors[alpha],linestyle="dotted");
                wfax.plot(jvals[mid-1:mid+1], np.matmul((Hsys_4d-HL_4d)[:,:,alpha,alpha], psimas[alpha,m])[mid-1:mid+1], color=alpha_colors[alpha],linestyle="solid",marker='s')               
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
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval/2):
                        melement = matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]);
                        Mns.append(np.real(melement*np.conj(melement)));

                # update T based on average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;

    return Emas, Mbmas;

    # average matrix elements over final states |k_n >
    # with the same energy as the intial state |k_m >
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mms = np.empty((n_bound_left,),dtype=float);
    for m in range(n_bound_left):
        
        # inelastic means averaging over an interval
        Mns = [];      
        for n in range(n_bound_right):
            if( abs(Ems[m] - Ens[n]) < interval/2):
                melement = np.dot(np.conj(psins[n]), np.dot(Hdiff,psims[m]));
                Mns.append( np.real(melement*np.conj(melement)) );
                if False:
                    print("-Hdiff[:,:] near barrier =\n",np.real(Hdiff[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
                    print("- psim near barrier\n",np.real(psims[m,interval_tup[0]:interval_tup[1]]));
                    print("- psin near barrier\n",np.real(psins[n,interval_tup[0]:interval_tup[1]]));
                    print("- barrier overlap of psim and psin: ",np.dot( psims[m,interval_tup[0]:interval_tup[1]][-4:],psins[n,interval_tup[0]:interval_tup[1]][-4:]));
                    assert False

        # update T based on average
        if verbose: print("\tinterval = ",interval, len(Mns));
        if(Mns): Mns = sum(Mns)/len(Mns);
        else: Mns = 0.0;
        Mms[m] = Mns;

    return Ems, Mms

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
    #n_spatial_dof = Ninfty+NL+len(HCprime)+NR+Ninfty;
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

def Ts_wfm(tL, tR, VL, VR, HC, Emas, verbose=0) -> np.ndarray:
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

    else: raise Exception; # not supported

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
    

