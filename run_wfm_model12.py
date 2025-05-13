'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 2:
Scattering of single electron off of two localized spin-1/2 impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
'''

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import sys
    
#################################################################################
#### utils

# load data
def load_data(fname, nind, nloc):
    print("Loading data from "+fname);
    data = np.load(fname);
    
    # params and spin channels
    myk = data[0,0];
    myJ = data[0,1];
    mysigmavals = data[0,2:];
    mysigmavals = np.extract(np.isfinite(mysigmavals), mysigmavals).astype(int);
    
    # different flavors of independent variables
    myxvals = data[1:1+nind];
    
    # transmission and reflection coefficients
    myTvals = data[1+nind:1+nind+nloc];
    myRvals = data[1+nind+nloc:1+nind+2*nloc]; # this is also sometimes overloaded
    assert(1+nind+2*nloc == np.shape(data)[0]);
    
    # return
    print("- sigmavals = ", mysigmavals);
    print("- shape xvals = ", np.shape(myxvals));
    print("- shape Tvals = ", np.shape(myTvals));
    return myk, myJ, mysigmavals, myxvals, myRvals, myTvals;

# colormap
def get_color(colori,numcolors):
    cm_reds = matplotlib.cm.get_cmap("seismic");
    denominator = 2*numcolors
    assert(colori >=0 and colori < numcolors);
    if colori <= numcolors // 2: # get a blue
        return cm_reds((1+colori)/denominator);
    else:
        return cm_reds((denominator-(numcolors-(colori+1)))/denominator);

# compute p2
def get_p2(Ti,Tp,theta):
    assert isinstance(Ti,float) and isinstance(Tp,float); # vectorized in thetas only
    return Ti*Tp/(Tp*np.cos(theta/2)*np.cos(theta/2)+Ti*np.sin(theta/2)*np.sin(theta/2));

# figure of merit
def FOM(Ti,Tp, grid=100000):
    thetavals = np.linspace(0,np.pi,grid);
    p2vals = p2(Ti,Tp,thetavals);
    fom = np.trapz(p2vals, thetavals)/np.pi;
    return fom;

def entangle(H,bi,bj):
    '''
    Perform a change of basis on a matrix such that basis vectors bi, bj become entangled (unentangled)
    in new ham, index bi -> + entangled state, bj -> - entangled state
    '''

    # check inputs
    assert(bi < bj);
    assert(bj < max(np.shape(H)));

    # rotation matrix
    R = np.zeros_like(H);
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            if( i != bi and i != bj):
                if(i == j):
                    R[i,j] = 1; # identity
            elif( i == bi and j == bi):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j == bj):
                R[i,j] = -1/np.sqrt(2);
            elif( i == bi and j == bj):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j== bi):
                R[i,j] = 1/np.sqrt(2);

    return np.matmul(np.matmul(R.T,H),R);
    
def h_cicc_eff(J, t, i1, i2, Nsites, pair_to_entangle):
    '''
    Construct tight binding blocks (each block has many body dofs) to implement
    cicc model in quasi many body GF method

    Args:
    - J, float, eff heisenberg coupling
    - t, float, hopping btwn sites - corresponds to t' in my setup
    - i1, list of sites for first spin. Must start at 0
    - i2, list of sites for second spin. Last site is N -> N sites in the SR
    - Nsites, int, total number of sites (typically N sites in SR, plus 1 from each lead, for N+2 total sites)
    - pair_to_entangle, tuple,  channels that form entangled states
    '''

    # check inputs\
    assert(isinstance(i1, list) and isinstance(i2, list));
    assert(i1[0] == 1);
    assert(i1[-1] < i2[0]);
    N = i2[-1];
    
    # heisenberg interaction matrices
    Se_dot_S1 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,1,0,0,0,0,0,0],   # channel 1 = up up down
                        [0,0,-1, 0, 2, 0,0,0],  # channel 2 = up down up
                        [0,0, 0,-1, 0, 2,0,0],
                        [0,0, 2, 0,-1, 0,0,0],  # channel 4 = down up up
                        [0,0, 0, 2, 0,-1,0,0],
                        [0,0, 0, 0, 0, 0,1,0],
                        [0,0, 0, 0, 0, 0,0,1] ]);

    Se_dot_S2 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-1, 0, 0, 2, 0, 0,0],
                        [0, 0, 1, 0, 0, 0, 0,0],
                        [0, 0, 0,-1, 0, 0, 2,0],
                        [0, 2, 0, 0,-1, 0, 0,0],
                        [0, 0, 0, 0, 0, 1, 0,0],
                        [0, 0, 0, 2, 0, 0,-1,0],
                        [0, 0, 0, 0, 0,0 , 0,1] ]);
    Se_dot_S1 = entangle(Se_dot_S1,*pair_to_entangle); # verified correct
    Se_dot_S2 = entangle(Se_dot_S2,*pair_to_entangle); # verified correct
    
    # insert these local interactions
    h_cicc =[];
    for sitei in range(Nsites): # iter over all sites
        if(sitei in i1 and sitei not in i2):
            h_cicc.append(Se_dot_S1);
        elif(sitei in i2 and sitei not in i1):
            h_cicc.append(Se_dot_S2);
        elif(sitei not in i1 and sitei not in i2):
            h_cicc.append(np.zeros_like(Se_dot_S1) );
        else:
            raise Exception;
    h_cicc = np.array(h_cicc, dtype = complex);
    
    # hopping connects like spin orientations only, ie is identity
    tblocks = [];
    for sitei in range(Nsites-1):
        tblocks.append(-t*np.eye(*np.shape(Se_dot_S1)) );
    tblocks = np.array(tblocks);
    
    return h_cicc, tblocks;
    
def h_cicc_hacked(J, t, i1, i2, Nsites, pair_to_entangle, unitcell):
    '''
    Construct tight binding blocks (each block has many body dofs) to implement
    cicc model in quasi many body GF method

    Args:
    - J, float, eff heisenberg coupling
    - t, float, hopping btwn sites - corresponds to t' in my setup
    - i1, list of sites for first spin. Must start at 0
    - i2, list of sites for second spin. Last site is N -> N sites in the SR
    - Nsites, int, total number of sites (typically N sites in SR, plus 1 from each lead, for N+2 total sites)
    - pair_to_entangle, tuple,  channels that form entangled states
    - unitcell, int, how many matrix elements define a unit cell
    '''

    # check inputs\
    assert(isinstance(i1, list) and isinstance(i2, list));
    assert(i1[0] == 1);
    assert(i1[-1] < i2[0]);
    N = i2[-1];
    
    # heisenberg interaction matrices
    Se_dot_S1 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,1,0,0,0,0,0,0],   # channel 1 = up up down
                        [0,0,-1, 0, 2, 0,0,0],  # channel 2 = up down up
                        [0,0, 0,-1, 0, 2,0,0],
                        [0,0, 2, 0,-1, 0,0,0],  # channel 4 = down up up
                        [0,0, 0, 2, 0,-1,0,0],
                        [0,0, 0, 0, 0, 0,1,0],
                        [0,0, 0, 0, 0, 0,0,1] ]);

    Se_dot_S2 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-1, 0, 0, 2, 0, 0,0],
                        [0, 0, 1, 0, 0, 0, 0,0],
                        [0, 0, 0,-1, 0, 0, 2,0],
                        [0, 2, 0, 0,-1, 0, 0,0],
                        [0, 0, 0, 0, 0, 1, 0,0],
                        [0, 0, 0, 2, 0, 0,-1,0],
                        [0, 0, 0, 0, 0,0 , 0,1] ]);
    Se_dot_S1 = entangle(Se_dot_S1,*pair_to_entangle); # verified correct
    Se_dot_S2 = entangle(Se_dot_S2,*pair_to_entangle); # verified correct
    
    # expand into unit cell space
    n_spin_dof = len(Se_dot_S1);
    Se_dot_S1_unit = np.zeros((unitcell*n_spin_dof, unitcell*n_spin_dof),dtype=float);
    Se_dot_S1_unit[:n_spin_dof,:n_spin_dof] = Se_dot_S1[:,:];
    Se_dot_S2_unit = np.zeros((unitcell*n_spin_dof, unitcell*n_spin_dof),dtype=float);
    Se_dot_S2_unit[:n_spin_dof,:n_spin_dof] = Se_dot_S2[:,:];
    
    # insert these local interactions
    h_cicc =[];
    for sitei in range(Nsites): # iter over all sites
        if(sitei in i1 and sitei not in i2):
            h_cicc.append(Se_dot_S1_unit);
        elif(sitei in i2 and sitei not in i1):
            h_cicc.append(Se_dot_S2_unit);
        elif(sitei not in i1 and sitei not in i2):
            h_cicc.append(np.zeros_like(Se_dot_S1_unit) );
        else:
            raise Exception;
    h_cicc = np.array(h_cicc, dtype = complex);
    return h_cicc;
    
def dos_RiceMele(u,v,w,Es):
    '''
    *Vectorized in Es* density of states of the *infinite* system (no relation to surface green's funct)
    '''
    raise NotImplementedError("replace with wfm universal");
    
    # invert Es to get ks
    ks = np.arccos(1/(2*v*w)*(Es*Es - u*u - v*v - w*w));
    band_deriv = np.gradient(Es,ks);
    
    # 1D tight-binding DOS is 2/pi * 1/|dE/dK|
    return 2/(np.pi*abs(band_deriv)); # this is a function of Es argument since we got ks by inverting it

if(__name__=="__main__"):
    # top level
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 5;
    case = sys.argv[1];

    # fig standardizing
    myxvals = 99; # number of pts on the x axis
    myfontsize = 14;
    mylinewidth = 1.0;
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"text.usetex": True})
    
##################################################################################
#### entanglement generation (cicc Fig 6)

if(case in ["Nx"]): # compare T vs rhoJa for N not fixed

    # iter over E, getting T
    logElims = -4,-1
    Evals = np.logspace(*logElims,myxvals,dtype=complex);
    Rvals = np.empty((len(Evals),len(source)), dtype = float);
    Tvals = np.empty((len(Evals),len(source)), dtype = float);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
        
        # location of impurities, fixed by kx0 = pi
        k_rho = np.arccos(Energy/(-2*tl)); # = ka since \varepsilon_0ss = 0
        kx0 = 2.0*np.pi;
        N0 = max(1,int(kx0/(k_rho))); #N0 = (N-1)
        print(">>> N0 = ",N0);

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = 1, 1+N0;
        hblocks, tnn = h_cicc_eff(Jeff, tl, i1, i2, i2+2, pair);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(Evali==0): 
            print("shape(hblocks) = ",np.shape(hblocks));
            print("sourcei = ",sourcei);
            
        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
        Rvals[Evali] = Rdum;
        Tvals[Evali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(Evals)));
    data[0,0] = tl;
    data[0,1] = Jeff;
    data[1,:] = np.real(Evals);
    data[2:10,:] = Tvals.T;
    data[10:,:] = Rvals.T;
    fname = "data/model12/Nx/"+str(int(kx0*100)/100);
    print("Saving data to "+fname);
    np.save(fname, data);


elif(case in ["N2","N2_k"]): # compare T vs rhoJa for N=2 fixed

    # channels
    pair = (1,2); # following entanglement change of basis, pair[0] = |+> channel
    source = np.zeros(8); 
    sourcei = 4; # down up up - the initial state when *generating* entanglement
    sigmas = [pair[0],pair[1],sourcei]; # all the channels of interest to generating entanglement
    source[sourcei] = 1; 

    # tight binding params
    tl = 1.0;
    Jval = -0.5*tl/100;
    Esplit = 0.0;
    Delta_zeeman = -Esplit; # Zeeman is only way to have energy difference btwn channels for spin-1/2
    
    # set number of lattice constants between MSQ 1 and MSQ
    # here it is fixed to be 1 bc N (number of sites in SR) is fixed at 2
    Distval = 1;
    
    # energy of the incident electron
    K_indep = True; # puts energy above the bottom of the band (logarithmically) on x axis
    if(case in ["N2_k"]): K_indep = False; # puts wavenumber on the x axis
    if(K_indep):               
        logKlims = -6, -4
        Kvals = np.logspace(*logKlims,myxvals, dtype = complex); # K > 0 always
        knumbers = np.arccos((Kvals-2*tl)/(-2*tl));
        indep_vals = np.real(Kvals);
    else:
        knumberlims = 0.1*(np.pi/Distval), 0.9*(np.pi/Distval);
        # ^ since we cannot exceed K = 4t, we cannot exceed k = \pi
        if(Distval == 1): knumberlims = 1e-3, 1e-2; # agrees with logKlims
        knumbers = np.linspace(knumberlims[0], knumberlims[1], myxvals, dtype=complex);
        Kvals = 2*tl - 2*tl*np.cos(knumbers);
        indep_vals = np.real(knumbers)/(np.pi/Distval);
    print("Kvals \in ",Kvals[0], Kvals[-1]);
    print("knumbers \in ",knumbers[0], knumbers[-1]);
    print("indep_vals \in ",indep_vals[0], indep_vals[-1]);

    # iter over E, getting T
    Rvals = np.empty((len(Kvals),len(source)), dtype = float);
    Tvals = np.empty((len(Kvals),len(source)), dtype = float);
    for Kvali in range(len(Kvals)):

        # energy
        Kval = Kvals[Kvali]; # Kval > 0 always, what I call Ki in paper
        Energy = Kval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc

        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [Distval+1];
        hblocks, tnn = h_cicc_eff(Jval, tl, i1, i2, i2[-1]+2, pair); # full 8 by 8, not reduced
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(Kvali==0): 
            print("shape(hblocks) = ",np.shape(hblocks));
            print("sourcei = ",sourcei);
        
        # Zeeman splitting effects. NB s=1/2 so 2s-1=0
        hzeeman = np.zeros_like(hblocks[0]);
        hzeeman[sourcei, sourcei] = Delta_zeeman;
        for hbi in range(len(hblocks)): hblocks[hbi] += np.copy(hzeeman);
        # shift so hblocks[0,i,i] = 0
        Eshift = hblocks[0,sourcei, sourcei];
        for hbi in range(len(hblocks)): hblocks[hbi] += -Eshift*np.eye(len(hblocks[0]));
        if(verbose > 3 and Kvali == 0): print("hblocks =\n",np.real(hblocks));

        # get R, T coefs
        Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, False, False, all_debug = False);
        Rvals[Kvali] = Rdum;
        Tvals[Kvali] = Tdum;

    # save data to .npy
    data = np.zeros((2+2*len(source),len(indep_vals)),dtype=float);
    data[0,0] = tl;
    data[0,1] = Jval;
    data[0,2:2+len(sigmas)] = sigmas[:];
    data[0,2+len(sigmas):] = np.full((len(indep_vals)-2-len(sigmas),), np.nan);
    data[1,:] = np.real(indep_vals);
    data[2:10,:] = Tvals.T; # 8 spin dofs
    data[10:,:] = Rvals.T;
    fname = "data/model12/"+case+"/"+str(int(Jval*1000)/1000);
    print("Saving data to "+fname);
    np.save(fname, data);
    
elif(case in ["rhoJ"]): # entanglement *preservation* at fixed rhoJa, N variable

    # channels
    pair = (1,2); # following entanglement change of basis, pair[0] = |+> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement 

    # tight binding params
    tl = 1.0;
    Jval = -0.5*tl/10; # /100
    
    # rhoJa = fixed throughout
    rhoJval = float(sys.argv[2]);
    fixed_Kval = abs(1/(np.pi*np.pi)*(Jval*Jval/tl)*1/(rhoJval*rhoJval));
    fixed_knumber = np.arccos((fixed_Kval-2*tl)/(-2*tl));
    fixed_Energy = complex(-2*tl*np.cos(fixed_knumber),0);
    fixed_rhoJ = (1/np.pi)*np.sqrt(abs(Jval*Jval/(tl*fixed_Kval)));
    
    # d = number of lattice constants between MSQ 1 and MSQ
    kdalims = 0.1*np.pi, 1.1*np.pi
    kdavals = np.linspace(*kdalims, myxvals);
    indep_vals = kdavals/np.pi;
    Distvals = (kdavals/fixed_knumber).astype(int);
    
    # iter over d, getting T
    # here T is *diagonal* in that we only save T in the channel of the source!
    # we don't compute for all channels so we leave the rest as NaNs
    Tvals = np.full((len(Distvals),8), np.nan, dtype = float);
    Tsummed = np.full((len(Distvals),8), np.nan, dtype = float);
    TpRsummed = np.full((len(Distvals),8), np.nan, dtype = float);
    for Distvali in range(len(Distvals)):
    
        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [Distvals[Distvali]+1];
        hblocks, tnn = h_cicc_eff(Jval, tl, i1, i2, i2[-1]+2, pair); # full 8 by 8, not reduced
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(Distvali==0): 
            #if(verbose > 3): print("hblocks =\n",np.real(hblocks));
            print("fixed_Energy = {:.6f}".format(np.real(fixed_Energy)));
            print("J = {:.4f}".format(Jval));
            print("rhoJ = {:.4f}".format(fixed_rhoJ));
            print("fixed_knumber = {:.6f}".format(fixed_knumber));
            print("\nmax N = {:.0f}".format(np.max(Distvals)+2));
            print("shape(hblocks) = ",np.shape(hblocks));

        for sigmai in range(len(sigmas)):
        
            # sourcei is one of the pairs always 
            source = np.zeros(8);
            source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet

            # get  T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl,fixed_Energy,"g_closed", source, 
                         False, False, all_debug = False);
            Tvals[Distvali,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
            Tsummed[Distvali,sigmas[sigmai]] = np.sum(Tdum);
            TpRsummed[Distvali,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);

    # save data to .npy
    data = np.zeros((2+2*len(source),len(indep_vals)),dtype=float);
    data[0,0] = fixed_rhoJ;
    data[0,1] = Jval;
    data[0,2:2+len(sigmas)] = sigmas[:];
    data[0,2+len(sigmas):] = np.full((len(indep_vals)-2-len(sigmas),), np.nan);
    data[1,:] = np.real(indep_vals);
    data[2:2+len(source),:] = Tvals.T; # 8 spin dofs
    data[2+len(source):2+2*len(source),:] = Tsummed.T;
    fname = "data/model12/"+case+"/{:.1f}".format(rhoJval);
    print("Saving data to "+fname);
    np.save(fname, data);

#################################################################
#### **DIATOMIC HAMILTONIAN**

elif(case in ["CB","VB"]): # entanglement *preservation* at fixed rhoJa, N variable
    my_unit_cell = 2; # since diatomic
    
    # tight binding params
    tl = 1.0;
    Jval = -0.5*tl/1; # /100

    # Rice-Mele tight binding
    vval = float(sys.argv[3]);
    uval = float(sys.argv[4]); 
    # w is always -tl;
    band_edges = np.array([np.sqrt(uval*uval+(-tl+vval)*(-tl+vval)),
                           np.sqrt(uval*uval+(-tl-vval)*(-tl-vval))]);

    # Rice-Mele matrices
    n_loc_dof = 8;
    diag_base_RM_spin = np.zeros((my_unit_cell*n_loc_dof, my_unit_cell*n_loc_dof),dtype=float);
    diag_base_RM_spin[:n_loc_dof,:n_loc_dof] = uval*np.eye(n_loc_dof);
    diag_base_RM_spin[n_loc_dof:,n_loc_dof:] = -uval*np.eye(n_loc_dof);
    diag_base_RM_spin[:n_loc_dof,n_loc_dof:] = vval*np.eye(n_loc_dof);
    diag_base_RM_spin[n_loc_dof:,:n_loc_dof] = vval*np.eye(n_loc_dof);
    offdiag_base_RM_spin = np.zeros((my_unit_cell*n_loc_dof, my_unit_cell*n_loc_dof),dtype=float);
    offdiag_base_RM_spin[n_loc_dof:,:n_loc_dof] = -tl*np.eye(n_loc_dof);
                                   
    # channels
    pair = (1,2); # following entanglement change of basis, pair[0] = |+> channel
    sigmas = np.array([pair[0],pair[1]]); # all the channels of interest to generating entanglement
                                          # in this case, MSQs in singlet or triplet, A orbital 
    
    # rhoJa = fixed throughout, thus fixing energy and wavenumber
    rhoJval = float(sys.argv[2]);

    # graphical solution for fixed energy
    fig, (dispax, dosax) = plt.subplots(ncols=2, sharey = True); myfontsize = 14;
    Ks_for_solution = np.logspace(-6,1,499);
    if(case in ["CB"]): 
        this_band = np.min(band_edges)+Ks_for_solution;
        this_band = this_band[this_band < np.max(band_edges)]; # stay w/in band
    else: 
        this_band = np.min(-band_edges)+Ks_for_solution;
        this_band = this_band[this_band < np.max(-band_edges)]; # stay w/in band
    bandks = np.linspace(-np.pi, np.pi,len(this_band));
    disp = dispersion_RiceMele(uval, vval, -tl, bandks);
    for dispvals in disp: dispax.plot(bandks/np.pi, dispvals,color="cornflowerblue");
    dispax.plot(np.arccos(1/(-2*vval*tl)*(this_band**2 - uval**2 - vval**2 - tl**2))/np.pi, this_band, color="black");
    dispax.set_xlabel("$ka/\pi$", fontsize = myfontsize);
    dispax.set_ylabel("$E(k)$", fontsize = myfontsize);
    
    banddos = dos_RiceMele(uval, vval, -tl, this_band);
    dosax.plot(banddos,this_band, color="black");
    dosline_from_rhoJ = rhoJval/abs(Jval);
    dosax.axvline(dosline_from_rhoJ, color="red");
    # solve graphically for E *specifically in VB/CB* that gives desired rhoJa
    fixed_Energy = complex(this_band[np.argmin(abs(banddos-dosline_from_rhoJ))],0);
    fixed_knumber = np.arccos(1/(-2*vval*tl)*(np.real(fixed_Energy)**2 - uval**2 - vval**2 - tl**2));
    fixed_rhoJ = banddos[np.argmin(abs(banddos-dosline_from_rhoJ))]*abs(Jval);
    print("fixed_Energy = {:.6f}".format(np.real(fixed_Energy)));
    print("fixed_knumber = {:.6f}".format(fixed_knumber));
    dosax.axhline(np.real(fixed_Energy), color="gray", linestyle="dashed");
    dispax.axhline(np.real(fixed_Energy), color="gray", linestyle="dashed");
     
    #
    RiceMele_numbers = np.arccos(1/(2*vval*(-tl))*(np.array([this_band[0],this_band[-1]])**2 - uval**2 - vval**2 - tl**2));
    RiceMele_shift_str = "$, E_{min}^{(VB)}=$"+"{:.2f}".format(np.min(-band_edges))
    if(case=="CB"): RiceMele_shift_str="$,  E_{min}^{(CB)}=$"+"{:.2f}".format(np.min(band_edges))
    RiceMele_shift_str += ",  $ka/\pi \in $[{:.2f},{:.2f}]".format(np.real(RiceMele_numbers[0]/np.pi), np.real(RiceMele_numbers[-1]/np.pi))
    dispax.set_ylabel("$E(k)$"+RiceMele_shift_str, fontsize = myfontsize);
    dosax.set_xlabel("$\\rho, \\rho Ja =${:.2}, $J =${:.2f}".format(rhoJval, Jval), fontsize = myfontsize);
    
    # show
    plt.tight_layout();
    plt.show();
    stopflag = False;
    try: 
        if(sys.argv[5]=="stop"): stopflag = True;
    except: print("Not flagged to stop");
    assert(not stopflag); 

    
    # d = number of lattice constants between MSQ 1 and MSQ
    kdalims = 0.01*np.pi, 1.1*np.pi
    widelimsflag = False;
    try: 
        if(sys.argv[5]=="widelims"): widelimsflag = True;
    except: print("Not flagged to widen kda limits");
    if(widelimsflag): kdalims = 0.01*np.pi, 100*np.pi; 
    
    # determine the number of lattice constants across this range
    kdavals = np.linspace(*kdalims, myxvals);
    Distvals = (kdavals/fixed_knumber).astype(int); 
    kdavals = kdavals[Distvals > 0];
    indep_vals = kdavals/np.pi;
    Distvals = Distvals[Distvals > 0];
    print(Distvals)
    
    # iter over d, getting T
    # here T is *diagonal* in that we only save T in the channel of the source!
    # we don't compute for all channels so we leave the rest as NaNs
    Tvals = np.full((len(Distvals),n_loc_dof), np.nan, dtype = float); # autom'ly extract only B orbital coefs
    Tsummed = np.full((len(Distvals),n_loc_dof), np.nan, dtype = float);
    TpRsummed = np.full((len(Distvals),n_loc_dof), np.nan, dtype = float);
    for Distvali in range(len(Distvals)):
        
        # construct hams
        # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
        i1, i2 = [1], [Distvals[Distvali]+1];
        hblocks_noRM = h_cicc_hacked(Jval, tl, i1, i2, i2[-1]+2, pair, my_unit_cell); 
        #print("shape hblocks = "+str(np.shape(hblocks_noRM))+", should be ({:.0f}, 16,16)".format(i2[-1]+2));
        # +2 for each lead site
        hblocks = 1*hblocks_noRM;
        tnn = np.zeros_like(hblocks);
        for blocki in range(len(hblocks)):
            hblocks[blocki] += diag_base_RM_spin;
            tnn[blocki] += offdiag_base_RM_spin;
        tnn = tnn[:-1];
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
        if(Distvali==0): 
            print("hblocks =\n");
            blockstoprint = 0
            for blocki in range(blockstoprint):
                print("\n\n");
                for chunki in range(my_unit_cell):
                    print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,chunki, chunki))
                    print(np.real(hblocks[blocki][chunki*n_loc_dof:(chunki+1)*n_loc_dof,chunki*n_loc_dof:(chunki+1)*n_loc_dof]));
                print("h(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,0, 1))
                print(np.real(hblocks[blocki][0*n_loc_dof:(0+1)*n_loc_dof,1*n_loc_dof:(1+1)*n_loc_dof]));
                print("t(j = {:.0f}, mu = {:.0f}, muprime = {:.0f})".format(blocki,1, 0))
                print(np.real(tnn[blocki][n_loc_dof:,:n_loc_dof]));
            print("J = {:.4f}".format(Jval));
            print("rhoJ = {:.4f}".format(fixed_rhoJ));
            print("\nmax N = {:.0f}".format(np.max(Distvals)+2));
            #assert False

        for sigmai in range(len(sigmas)):   
            # sourcei is one of the pairs always 
            source = np.zeros(my_unit_cell*n_loc_dof);
            source[sigmas[sigmai]] = 1;  # MSQs in singlet or triplet, A orbital

            # get  T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, fixed_Energy, "g_RiceMele", source, 
                         False, False, all_debug = False);
            Tdum = Tdum[n_loc_dof:];
            Rdum = Rdum[n_loc_dof:]; # autom'ly extract only B orbital coefs !!
            Tvals[Distvali,sigmas[sigmai]] = Tdum[sigmas[sigmai]];
            Tsummed[Distvali,sigmas[sigmai]] = np.sum(Tdum);
            TpRsummed[Distvali,sigmas[sigmai]] = np.sum(Tdum) + np.sum(Rdum);

    # save data to .npy
    nind = 2; # number of x values saved -- Tvals needs to skip this many rows
    data = np.zeros((1+nind+2*n_loc_dof,len(indep_vals)),dtype=float);
    data[0,0] = fixed_knumber;
    data[0,1] = Jval;
    
    # tell which channel to plot
    data[0,2:2+len(sigmas)] = sigmas[:];
    data[0,2+len(sigmas):] = np.full((len(indep_vals)-2-len(sigmas),), np.nan);
    
    # x values:
    data[1,:] = np.real(indep_vals);
    data[2,:] = Distvals;
    
    # saved sigma-> sigma and sigma-> \sum_\sigma Transmission coefs
    data[1+nind:1+nind+n_loc_dof,:] = Tvals.T; # 8 spin dofs
    data[1+nind+n_loc_dof:1+nind+2*n_loc_dof,:] = Tsummed.T;
    fname = "data/model12/"+case+"/{:.1f}_{:.1f}_{:.1f}".format(rhoJval, vval, uval);
    print("Saving data to "+fname);
    np.save(fname, data);


########################################################################
#### plot data


#### plot transmission in same state like in Cicc fig 2
elif(case in ["rhoJ_visualize"]):

    # set up plots
    num_plots = 1;
    height_mult = 1;
    fig, axes = plt.subplots(num_plots, 1, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7,6);
    
    # get data
    dataf = sys.argv[2];
    datacase = dataf.split("/")[-2];
    if(datacase in ["CB", "VB"]):
        xretrievalinteger = 2;
        kval, Jval, sigmavals, xlist, Tsummed, Tvals = load_data(dataf, xretrievalinteger, 8);
        xvals, dvals = xlist[0], xlist[1];
    else: 
        xretrievalinteger = 1;
        kval, Jval, sigmavals, xvals, Tsummed, Tvals = load_data(dataf, xretrievalinteger, 8);
        kval = np.nan; # because I have not run new data yet
        xvals = xvals[0];
        dvals = np.nan*np.ones_like(xvals);
    print(dvals)
        
    
    # get channels
    assert(sigmavals[1]-sigmavals[0] == 1); # poor mans verification of pair assignment

    # plot T(|+> -> |+>) and T(|-> -> |->) 
    for sigmai in range(len(sigmavals)):
        axes[0].plot(xvals, Tvals[sigmavals[sigmai]], color = mycolors[sigmai], marker =
                mymarkers[sigmai+1],markevery=mymarkevery,linewidth = mylinewidth,
                label = "$|"+["+","-"][sigmai]+"\\rangle \\rightarrow |"+["+","-"][sigmai]+"\\rangle$");
        axes[0].plot(xvals, Tsummed[sigmavals[sigmai]], color = mycolors[sigmai], linestyle="dashed",
                label = "$|"+["+","-"][sigmai]+"\\rangle \\rightarrow \sum_\sigma|\sigma\\rangle$");
    print(">>> T+ max = ",np.max(Tvals[sigmavals[0]])," at indep = ",xvals[np.argmax(Tvals[sigmavals[0]])]);
    print(">>> T- max = ",np.max(Tvals[sigmavals[1]])," at indep = ",xvals[np.argmax(Tvals[sigmavals[1]])]);
    
    # format T plot
    offset_y = 0.08;
    axes[0].set_ylim(0-offset_y, 1.0+offset_y);
    
    # format
    if(datacase in ["CB","VB"]): datarhoJ, datav, datau = tuple(dataf.split("/")[-1][:-4].split("_"));
    else: datarhoJ = dataf.split("/")[-1][:-4];
    title_str = "$\\rho J a = ${:.1f}, $J =${:.4f}".format(float(datarhoJ),Jval);
    if(datacase in ["VB"]): title_str = "$\mathbf{Valence\,Band}$"+", $u =${:.2f}, $v =${:.2f}, ".format(float(datau), float(datav))+title_str;
    if(datacase in ["CB"]): title_str = "$\mathbf{Conduction\,Band}$"+", $u =${:.2f}, $v =${:.2f}, ".format(float(datau), float(datav))+title_str;
    axes[0].set_title(title_str, fontsize = myfontsize) #, x=0.5, y = 0.0);
    axes[-1].set_xlabel('$k_i d/\pi, k_i =${:.4f}, $d \in [${:.0f}$a$,{:.0f}$a]$'.format(kval, dvals[0], dvals[-1]),fontsize = myfontsize);
    for _ in range(int(np.floor(np.max(xvals)))+1): axes[-1].axvline(_, linestyle="dotted", color="gray");
    axes[0].legend(fontsize=myfontsize);
    
    # show
    plt.tight_layout();
    if False:
        plt.savefig('figs/double/model12.pdf');
    else:
        plt.show();

#### plot T+ = entanglement generation prob like in Cicc fig 6
elif(case in ["gen_visualize"]):

    # set up plots
    num_plots = 2;
    height_mult = 1;
    fig, axes = plt.subplots(num_plots, 1, gridspec_kw={'height_ratios':[1,height_mult]}, sharex=True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*(1+height_mult)/2);
    
    # get data
    dataf = sys.argv[2];
    datacase = dataf.split("/")[-2];
    if("_k" in case): K_indep = False;
    else: K_indep = True;
    _, Jval, sigmavals, xvals, Rvals, Tvals, totals = load_data(dataf); 
    # typically xvals are Ki, i.e. the energy above zero
    logElims = np.log10(xvals[0]), np.log10(xvals[-1]);
    
    # get channels
    pairvals, sourceival = (sigmavals[0], sigmavals[1]), sigmavals[-1];
    assert(pairvals[1]-pairvals[0] == 1); # poor mans verification of pair assignment

    # plot Ti, T+, T-
    for sigmai in range(len(sigmavals)):
        factor = 1;
        if sigmavals[sigmai] == pairvals[1]: factor = 10**5; # blow up T-
        axes[0].plot(xvals, factor*Tvals[sigmavals[sigmai]],color = mycolors[sigmai],marker = mymarkers[sigmai+1],markevery=mymarkevery,linewidth = mylinewidth);
    print(">>> T+ max = ",np.max(Tvals[pairvals[0]])," at Ki = ",xvals[np.argmax(Tvals[pairvals[0]])]);
    print(">>> T- max = ",np.max(Tvals[pairvals[1]])," at Ki = ",xvals[np.argmax(Tvals[pairvals[1]])]);
    
    # format T plot
    lower_y = 0.08;
    axes[0].set_ylim(-lower_y,1.0);
    axes[0].set_ylabel(r'$T_\alpha$', fontsize = myfontsize);
    
    # plot p2 at diff theta
    numtheta = 9;
    thetavals = np.linspace(0,np.pi,numtheta);
    thetais = [0,1,2,8];
    endthetavals = [];
    for thetai in thetais:
        yvals = [];
        for xi in range(len(xvals)):
            yvals.append(get_p2(Tvals[sourceival,xi],Tvals[pairvals[0],xi],thetavals[thetai]));
        axes[1].plot(xvals, yvals,color = get_color(thetai,numtheta),linewidth = mylinewidth);
        endthetavals.append(np.copy(yvals)[-1]);
        print(thetavals[thetai]);
    print(endthetavals);

    # plot analytical FOM
    axes[1].plot(xvals, np.sqrt(Tvals[sourceival]*Tvals[pairvals[0]]), color = accentcolors[0], marker=mymarkers[0],markevery=mymarkevery, linewidth = mylinewidth)
    print(">>> p2 max = ",np.max(np.sqrt(Tvals[sourceival]*Tvals[pairvals[0]]))," at Ki = ",xvals[np.argmax(np.sqrt(Tvals[sourceival]*Tvals[pairvals[0]]))]);

    # label LHS with p2 values
    ax1ylim = (0,1.0);
    axes[1].set_ylim(*ax1ylim);
    axes[1].set_ylabel('$p^2(\\tilde{\\theta})$', fontsize = myfontsize);
    # label thetavals with RHS yticks
    if True:
        axRHS = axes[1].twinx();
        axRHS.tick_params(axis='y');
        axRHS.set_ylim(*ax1ylim);
        axRHS.set_yticks(endthetavals);
        axRHS.set_yticklabels(['0','$\pi/8$','$\pi/4$','$\pi$']);


    # format
    if(K_indep): 
        axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
        axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    else: axes[-1].set_xlabel('$k_i d/\pi$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    
    # show
    plt.tight_layout();
    #plt.savefig('figs/double/model12.pdf');
    plt.show();
    
else: raise NotImplementedError("case = "+case);
    
