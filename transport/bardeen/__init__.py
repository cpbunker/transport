'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

from transport import fci_mod, wfm

import numpy as np

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
    HL, _ = Hsysmat(tinfty, tL, tRprime, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HL),n_loc_dof));
    Emas, psimas = [], []; # will index as Emas[alpha,m]
    n_bound_left = 0;
    interval = 3;
    for alpha in range(n_loc_dof):
        Ems, psims = np.linalg.eigh(HL[:,:,alpha,alpha]);
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
    HR, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    assert(is_alpha_conserving(fci_mod.mat_4d_to_2d(HR),n_loc_dof));
    Enbs, psinbs = [], []; # will index as Enbs[beta,n]
    n_bound_right = 0;
    for beta in range(n_loc_dof):
        Ens, psins = np.linalg.eigh(HR[:,:,beta,beta]);
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
    Hsys, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    Hdiff = Hsys - HL;

    # visualize
    if(verbose > 9):
        import matplotlib.pyplot as plt
        jvals = np.array(range(len(Hsys))) + offset;
        myfig,myaxes = plt.subplots(n_loc_dof,sharex=True);
        if n_loc_dof == 1: myaxes = [myaxes];
        for alpha in range(n_loc_dof):
            Hs = [HL,HR,Hsys,Hdiff,Hsys-HR];
            Hstrs = ["HL","HR","Hsys","Hsys-HL","Hsys-HR"];
            for Hi in range(len(Hs)):
                myaxes[alpha].plot(jvals, Hi*0.001+np.diag(Hs[Hi][:,:,alpha,alpha]),label = Hstrs[Hi]);
        plt.legend();plt.show();assert False;

    # average matrix elements over final states |k_n \beta>
    # with the same energy as the intial state |k_m \alpha>
    # average over energy but keep spin separate
    Hdiff = fci_mod.mat_4d_to_2d(Hsys - HL);
    Tbmas = np.empty((n_loc_dof,n_bound_left,n_loc_dof),dtype=float);
    # initial energy and spin states
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):
            # final spin states
            for beta in range(n_loc_dof):
                # average over final state energy
                Mbma = 0.0;
                Nbma = 0; # number of final states averaged over
                interval_width = 1e-9;
                #interval_width = abs(Enbs[alpha,-2]-Enbs[alpha,-1]); # arbitrary for now
                for n in range(n_bound_right):
                    if( abs(Emas[alpha,m] - Enbs[beta,n]) < interval_width/2):
                        Nbma += 1;
                        melement = matrix_element(beta,psinbs[:,n],Hdiff,alpha,psimas[:,m]);
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
    if verbose: print("-HL[:,:] =\n",np.real(HL[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));

    # left well eigenstates
    Ems, psims = np.linalg.eigh(HL);
    psims = psims.T[Ems+2*tLa < E_cutoff];
    Ems = Ems[Ems+2*tLa < E_cutoff].astype(complex);
    n_bound_left = len(Ems);
    kms = np.arccos((Ems-VLa)/(-2*tLa)); # wavenumbers in the left well
    
    # right well 
    HR_4d, _ = Hsysmat(tinfty, tLprime, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    HR = fci_mod.mat_4d_to_2d(HR_4d);
    if verbose: print("-HR[:,:] =\n",np.real(HR[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
    
    # right well eigenstates
    Ens, psins = np.linalg.eigh(HR);
    psins = psins.T[Ens+2*tRa < E_cutoff];
    Ens = Ens[Ens+2*tRa < E_cutoff].astype(complex);
    n_bound_right = len(Ens);
    knbs = np.arccos((Ens-VRa)/(-2*tRa)); # wavenumbers in the right well

    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);
    if(verbose > 9):
        for m in range(10):
            psi0 = psims[m];
            print(m,Ems[m],np.dot(np.conj(psi0[::2]),psi0[::2]),np.dot(np.conj(psi0[1::2]),psi0[1::2]));
        assert False
        import matplotlib.pyplot as plt
        jvals = np.array(range(len(Hsys_4d))) + offset;
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
    Hdiff = fci_mod.mat_4d_to_2d(Hsys_4d - HL_4d);
    Mnms = np.empty((n_bound_right,n_bound_left),dtype=float);
    Tms = np.empty((n_bound_left,),dtype=float);
    for m in range(n_bound_left):

        # average over final states
        Mm = 0.0;
        Nm = 0; # number of final states averaged over
        interval_width = 1e-9;
        #interval_width = abs(Ens[-1]-Ens[-2]); # arbitrary for now       
        for n in range(n_bound_right):
            if( abs(Ems[m] - Ens[n]) < interval_width/2):
                Nm += 1;
                melement = np.dot(np.conj(psins[n]), np.dot(Hdiff,psims[m]));
                Mm += np.real(melement*np.conj(melement));
                print(interval_width, Nm);
                #print(np.real(Hdiff[interval_tup[0]:interval_tup[1],interval_tup[0]:interval_tup[1]]));
                print("->",np.real(psims[m,interval_tup[0]:interval_tup[1]]));
                print("->",np.real(psins[n,interval_tup[0]:interval_tup[1]]));
                print("->",np.dot( psims[m,interval_tup[0]:interval_tup[1]][-4:],psins[n,interval_tup[0]:interval_tup[1]][-4:]));
                Sz = np.zeros_like(HL_4d);
                for Szi in range(np.shape(HL_4d)[0]):
                    for Szj in range(np.shape(HL_4d)[0]):
                        if Szi==Szj:
                            Sz[Szi,Szj] = np.array([[0,1],[1,0]]);
                Sz = fci_mod.mat_4d_to_2d(Sz);
                print("->",np.dot(psims[m],np.dot(Sz,psins[n])));
                assert False

        # update T based on average
        #print(interval_width, Nm, psims[m]);
        if(Nm == 0): Mm = 0.0;
        else: Mm = Mm/Nm;
        Tms[m] = NL/(kms[m]*tLa) *NR/(kms[m]*tRa) *Mm;

    return Ems, Tms;

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

def Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC) -> np.ndarray:
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

def Hwellmat(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR) -> np.ndarray:
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

def matrix_element(beta,psin,op,alpha,psim) -> complex:
    '''
    Take the matrix element of a
    -not in general alpha conserving 2d operator, with spin/spatial dofs mixed
    -alpha conserving 2d state vector, with spin/spatial dofs separated
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
    psinbeta[beta] = psin[beta]; # all zeros except for psi[beta]
    psinbeta = fci_mod.vec_2d_to_1d(psinbeta.T); # flatten
    assert(is_alpha_conserving(psinbeta,n_loc_dof));
    return np.dot(np.conj(psinbeta), np.dot(op,psimalpha));

##################################################################################
#### test code

def plot_wfs(tinfty, tL, tC, tR, Vinfty, VL, VC, VR, Ninfty, NL, NC, NR, tLprime = None, VLprime = None, tRprime = None, VRprime = None) -> None:
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

if __name__ == "__main__":

    pass;
    


    








