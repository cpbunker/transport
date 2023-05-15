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
    if(verbose):
        print("Hsys = "+str(np.shape(Hsys_4d))+"\n",Hsys_4d[mid-2:mid+2,mid-2:mid+2,0,0]);
        print("HL = "+str(np.shape(HL_4d))+"\n",HL_4d[mid-2:mid+2,mid-2:mid+2,0,0]);
        print("HR = "+str(np.shape(HL_4d))+"\n",HL_4d[mid-2:mid+2,mid-2:mid+2,0,0]);

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
    Emas_arr, psimas_arr # shape is (n_loc_dof, n_bound_left);

    # flag initial vs final states
    mid = len(Hsys_4d)//2;
    flags = np.zeros_like(Emas_arr,dtype=int)
    for alpha in range(n_loc_dof):
        for m in range(n_bound_left):
            psim = psimas_arr[alpha,m];
            weight_left = np.dot( np.conj(psim[:mid]), psim[:mid]);
            weight_right = np.dot( np.conj(psim[mid:]), psim[mid:]);
            if(weight_left > weight_right): # this is an initial state
                flags[alpha,m] = 1;
    Emas = np.where(flags==1,Emas_arr,np.nan);
    Enbs = np.where(flags==0,Emas_arr,np.nan);
                    
    # visualize
    jvals = np.linspace(-mid, -mid +len(Hsys_4d)-1,len(Hsys_4d), dtype=int);
    if(verbose > 9):

        # energies
        print("Emas_arr "+str(np.shape(Emas_arr))+"\n",Emas_arr/tbulk);
        print("Emas "+str(np.shape(Emas))+"\n",Emas/tbulk);
        print("Enbs "+str(np.shape(Enbs))+"\n",Enbs/tbulk);

        # plot left wfs
        for m in range(6): #n_bound_left):
            wffig, wfax = plt.subplots();
            alpha_colors=["tab:blue","tab:orange"];
            for alpha in range(n_loc_dof):
                if(not np.isnan(Emas[alpha,m])):
                    wfax.set_title("$"+str(np.real(Emas[alpha,m].round(4)))+" \\rightarrow "+str(np.real(Enbs[alpha,m+1].round(4)))+"$");
                    wfax.plot(jvals,np.diag(HL_4d[:,:,alpha,alpha]),color="black");
                    wfax.plot(jvals[:-1], np.diagonal(HL_4d[:,:,alpha,alpha], offset=1), color="black", linestyle="dotted")
                    wfax.plot(jvals, np.real(psimas_arr[alpha,m]),color=alpha_colors[alpha],linestyle="solid");
                    wfax.plot(jvals, np.real(psimas_arr[alpha,m+1]),color=alpha_colors[alpha],linestyle="dotted");
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
           Ninfty, NL, NR, HC,HCprime,
           interval=1e-9,E_cutoff=1.0,HT_perturb=False,verbose=0) -> tuple:
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

    # convert from matrices to _{alpha alpha} elements
    to_convert = [tL, VL, tR, VR];
    converted = [];
    for convert in to_convert:
        if( np.any(convert - np.diagflat(np.diagonal(convert))) ):
            raise ValueError; # VL must be diag
        converted.append(np.diagonal(convert));
    tLa, VLa, tRa, VRa = tuple(converted);

    # left well eigenstates
    HL_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
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
    HR_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
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
        raise(Exception("Don't do this!"));
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
        for m in range(10,10+min(n_bound_left,2)):
            fig, wfax = plt.subplots();
            alpha_colors=["tab:blue","tab:orange"];
            for alpha in range(n_loc_dof):
                print(Emas[alpha,m]);
                print(Enbs[alpha,m+energy_off]);
                wfax.set_title("Left: "+str(Emas[alpha,m].round(4)));
                wfax.plot(jvals,np.diag(HL_4d[:,:,alpha,alpha])/max(abs(np.diag(HL_4d[:,:,alpha,alpha]))),color="black");
                wfax.plot(jvals, np.real(psimas[alpha,m])/max(abs(np.real(psimas[alpha,m]))),color=alpha_colors[alpha],linestyle="solid");
                wfax.plot(jvals, np.real(psinbs[alpha,m+energy_off])/max(abs(np.real(psinbs[alpha,m+energy_off])))*(-1),color=alpha_colors[alpha],linestyle="dotted");
                wfax.plot(jvals[mid-len(HC)-10:mid+len(HC)+10], abs((np.matmul((Hsys_4d-HL_4d)[:,:,alpha,alpha], psimas[alpha,m])/max(abs(np.matmul((Hsys_4d-HL_4d)[:,:,alpha,alpha], psimas[alpha,m])))) )[mid-len(HC)-10:mid+len(HC)+10], color=alpha_colors[alpha],linestyle="solid",marker='s')               
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

                # update M with average
                if(verbose): print("\tinterval = ",interval, len(Mns));
                if Mns: Mns = sum(Mns)/len(Mns);
                else: Mns = 0.0;
                Mbmas[beta,m,alpha] = Mns;

    return Emas, Mbmas;

def kernel_well_super(tinfty, tL, tR,
           Vinfty, VL, VLprime, VR, VRprime,
           Ninfty, NL, NR, HC,HCprime, change_basis,
           interval=1e-9,E_cutoff=1.0,verbose=0) -> tuple:
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
    the arg coefs gives the change of basis:
    |\tilde{\alpha} > = \sum_\alpha coefs[\alpha, |tilde{\alpha} ] |\alpha>
    
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
    HL_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VL, VRprime, Ninfty, NL, NR, HCprime);
    HL = fci_mod.mat_4d_to_2d(HL_4d);
    
    # left well eigenstates
    Ems, psims = np.linalg.eigh(HL);
    psims = psims.T[Ems+2*tLa < E_cutoff];
    Ems = Ems[Ems+2*tLa < E_cutoff].astype(complex);
    if(len(Ems) % n_loc_dof != 0): Ems, psims = Ems[:-1], psims[:-1]; # must be even
    n_bound_left = len(Ems);

    # get Sx val for each psim
    Sx_op = np.zeros((len(psims[0]),len(psims[0]) ),dtype=complex);
    for eli in range(len(Sx_op)-1): Sx_op[eli,eli+1] = 1.0; Sx_op[eli+1,eli] = 1.0;
    Sxms = np.zeros_like(Ems);
    for m in range(n_bound_left):
        Sxms[m] = np.dot( np.conj(psims[m]), np.dot(Sx_op, psims[m]));

    # recall \alpha basis is eigenstates of HC[j=0,j=0]
    alpha_eigvals, _ = np.linalg.eigh(HC[len(HC)//2,len(HC)//2]);
    eigval_tol = 1e-9;

    # measure HC[j=0,j=0] for each k_m
    HC00_op_4d = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof),dtype=complex);
    for sitej in range(n_spatial_dof):
        HC00_op_4d[sitej,sitej] = HC[len(HC)//2,len(HC)//2];
    HC00_op = fci_mod.mat_4d_to_2d(HC00_op_4d);
    alphams = np.empty((n_bound_left,),dtype=complex);
    for m in range(n_bound_left):
        alphams[m] = np.dot( np.conj(psims[m]), np.matmul(HC00_op, psims[m]));
        if(verbose>5): print(m, Ems[m], alphams[m], Sxms[m]);
    n_bound_left = n_bound_left // n_loc_dof;

    # classify left well eigenstates in the \alpha basis
    Emas = np.empty((n_loc_dof,n_bound_left),dtype=complex);
    psimas = np.empty((n_loc_dof,n_bound_left,len(psims[0])),dtype=complex);
    for eigvali in range(len(alpha_eigvals)):
        Es_this_a, psis_this_a = [], [];
        for m in range(n_bound_left*n_loc_dof):
            if(abs(np.real(alphams[m] - alpha_eigvals[eigvali])) < eigval_tol):
                Es_this_a.append(Ems[m]); psis_this_a.append(psims[m]);
        Emas[eigvali], psimas[eigvali] = Es_this_a, psis_this_a;
    del Ems, psims;
    
    # right well 
    HR_4d, _ = Hsysmat(tinfty, tL, tR, Vinfty, VLprime, VR, Ninfty, NL, NR, HCprime);
    HR = fci_mod.mat_4d_to_2d(HR_4d);

    # right well eigenstates
    Ens, psins = np.linalg.eigh(HR);
    psins = psins.T[Ens+2*tRa < E_cutoff];
    Ens = Ens[Ens+2*tRa < E_cutoff].astype(complex);
    if(len(Ens) % n_loc_dof != 0): Ens, psins = Ens[:-1], psins[:-1]; # must be even
    n_bound_right = len(Ens);

    # measure HC[j=0,j=0] for each k_n
    alphans = np.empty((n_bound_right,),dtype=complex);
    for n in range(n_bound_right):
        alphans[n] = np.dot( np.conj(psins[n]), np.matmul(HC00_op, psins[n]));
    n_bound_right= n_bound_right // n_loc_dof;
    
    # classify right well eigenstates in the \alpha basis
    Enbs = np.empty((n_loc_dof,n_bound_right),dtype=complex);
    psinbs = np.empty((n_loc_dof,n_bound_right,len(psins[0])),dtype=complex);
    for eigvali in range(len(alpha_eigvals)):
        Es_this_b, psis_this_b = [], [];
        for n in range(n_bound_right*n_loc_dof):
            if(abs(np.real(alphans[n] - alpha_eigvals[eigvali])) < eigval_tol):
                Es_this_b.append(Ens[n]); psis_this_b.append(psins[n]);
        Enbs[eigvali], psinbs[eigvali] = Es_this_b, psis_this_b;
    del Ens, psins;

    # physical system
    Hsys_4d, offset = Hsysmat(tinfty, tL, tR, Vinfty, VL, VR, Ninfty, NL, NR, HC);  

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
    del Mbmas
    Mbmas_tilde = np.real(np.conj(Mbmas_tilde)*Mbmas_tilde);
    Mbmas_tilde = Mbmas_tilde.astype(float);
    
    # visualize
    if(verbose > 9):

        # compare matrix elements
        if False:
            ps = np.array(range(len(Emas[0])),dtype=int);
            Efig, Eax = plt.subplots();
            which = 0;
            Eax.plot(ps,Enbs[0]-Emas[0,which],label="$\\varepsilon_{p+}-\\varepsilon_{0+}$",color="darkblue");
            Eax.plot(ps,Enbs[1]-Emas[1,which],label="$\\varepsilon_{p-}-\\varepsilon_{0-}$",color="darkblue",linestyle="dashed");
            Eax.scatter(ps, (Enbs[0]-Emas[0,which])-(Enbs[1]-Emas[1,which]),label="$(\\varepsilon_{p+}-\\varepsilon_{0+})-(\\varepsilon_{p-}-\\varepsilon_{0-})$",color="darkblue",marker='s',linestyle="solid");
            plt.legend();
            plt.show();
            assert False;
        
        # plot hams
        jvals = np.array(range(len(Hsys_4d))) + offset;
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
            wfax.plot(jvals,np.diag(Hsys_4d[:,:,0,0]),color="black");
            alpha_colors = ["tab:blue", "tab:orange"];
            for alpha in range(len(alpha_colors)):
                wfax.plot(jvals, 1e-3*alpha+np.real(psimas[alpha,m]),color=alpha_colors[alpha],linestyle="solid");
                wfax.plot(jvals, 1e-3*alpha+np.real(psinbs[alpha,m]),color=alpha_colors[alpha],linestyle="dotted");
            wfax.set_title("Ema = {:.4f}, alpha = {:.2f},\nEnb = {:.4f}, beta = {:.2f},\nM_bma = {:.2e}"
                           .format(np.real(Emas[alpha,m]),alpha,np.real(Enbs[alpha,m]),beta, Mbmas[alpha,m,alpha]));
            plt.show();
        assert False;

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
            if( np.isnan(Emas[alpha,m])): Energy = Emas[alpha,m-1];
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
    

