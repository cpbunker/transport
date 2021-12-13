'''
Christian Bunker
M^2QM at UF
June 2021

ops.py

Representation of operators, hamiltonian and otherwise, in pySCF fci friendly
form, i.e. as numpy arrays corresponding to 2nd quantized hamiltonians

pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr


'''

import numpy as np

#######################################################
#### 1e operators

def occ(site_i, norbs):
    '''
    Operator for the occupancy of sites specified by site_i
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    # create op array
    o = np.zeros((norbs,norbs));

    # iter over sites, = 1 for ones we measure occ of
    for i in range(site_i[0], site_i[-1]+1, 1):
        o[i,i] = 1.0;

    return o;


def Sx(site_i, norbs):
    '''
    Operator for the x spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    # create op array
    sx = np.zeros((norbs,norbs));

    # iter over all given sites
    for i in range(site_i[0], site_i[-1]+1, 2): # doti is up, doti+1 is down 
        sx[i,i+1] = 1/2; # spin up
        sx[i+1,i] = 1/2; # spin down

    return sx;


def Sy(site_i, norbs):
    '''
    Operator for the y spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    # create op array
    sy = np.zeros((norbs,norbs),dtype = complex);

    # iter over all given sites
    for i in range(site_i[0], site_i[-1]+1, 2): # doti is up, doti+1 is down 
        sy[i,i+1] = -1/2; # spin up
        sy[i+1,i] = 1/2; # spin down

    return sy;


def Sz(site_i, norbs):
    '''
    Operator for the z spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    # create op array
    sz = np.zeros((norbs,norbs));

    # iter over all given sites
    for i in range(site_i[0], site_i[-1]+1, 2): # doti is up, doti+1 is down 
        sz[i,i] = 1/2; # spin up
        sz[i+1, i+1] = -1/2; # spin down

    return sz;


def Jup(site_i, norbs):
    '''
    Current of up spin e's thru sitei
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( len(site_i) == 2); # should be only 1 site ie 2 spatial orbs

    # current operator (1e only)
    JL = np.zeros((norbs,norbs)); # leftwards
    JR = np.zeros((norbs, norbs)); # rightwards

    # even spin index is up spins
    upiL = site_i[0];
    upiR = site_i[-1] - 1;
    assert(upiL % 2 == 0 and upiR % 2 == 0); # check even
    JL[upiL-2,upiL] = -1;  # dot up spin to left up spin #left moving is negative current
    JL[upiL,upiL-2] =  1; # left up spin to dot up spin # hc of above # right moving is +
    JR[upiR+2,upiR] = 1;  # up spin to right up spin
    JR[upiR,upiR+2] =  -1; # hc

    return [JL, JR];


def Jdown(site_i, norbs):
    '''
    Current of down spin e's thru sitei
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( len(site_i) == 2); # should be only 1 site ie 2 spatial orbs

    # current operator (1e only)
    JL = np.zeros((norbs,norbs)); # leftwards
    JR = np.zeros((norbs, norbs)); # rightwards

    # odd spin index is down spins
    dwiL = site_i[0] + 1;
    dwiR = site_i[-1];
    assert(dwiL % 2 == 1 and dwiR % 2 == 1); # check odd
    JL[dwiL-2,dwiL] = -1;  # dot dw spin to left dw spin #left moving is negative current
    JL[dwiL,dwiL-2] =  1; # left dw spin to dot dw spin # hc of above # right moving is +
    JR[dwiR+2,dwiR] = 1;  # dot dw spin to right dw spin
    JR[dwiR,dwiR+2] =  -1; # hc

    return [JL, JR];


#######################################################
#### 1e hamiltonians

def h_leads(V, N):
    '''
    create 1e hamiltonian for leads alone
    V is hopping between leads
    N tuple of number of lead sites on left, right lead
    '''

    assert(isinstance(N, tuple));
    
    n_lead_sos = 2*N[0] + 2*N[1]; # 2 spin orbs per lead site
    h = np.zeros((n_lead_sos,n_lead_sos));
    
    # iter over lead sites
    for i in range(2*N[0]-2): # i is spin up orb on left side, i+1 spin down

        h[i,i+2] += -V; # left side
        h[i+2,i] += -V; # h.c.
        
    for i in range(2*N[1]-2):
        
        h[n_lead_sos-1-i,n_lead_sos-1-(i+2)] += -V; # right side
        h[n_lead_sos-1-(i+2),n_lead_sos-1-i] += -V; # h.c.

    if False:
        h[0,2] = 0.0;
        h[2,0] = 0.0;
        
    return h; # end h_leads;


def h_chem(mu,N):
    '''
    create 1e hamiltonian for chem potential of leads
    mu is chemical potential of leads
    N tuple of number of lead sites on left, right lead
    '''

    assert(isinstance(N, tuple));
    
    n_lead_sos = 2*N[0] + 2*N[1]; # 2 spin orbs per lead site
    h = np.zeros((n_lead_sos,n_lead_sos));
    
    # iter over lead sites
    for i in range(2*N[0]): # i is spin up orb on left side, i+1 spin down

        h[i,i] += mu; # left side
        
    for i in range(1,2*N[1]+1):
        h[n_lead_sos-i,n_lead_sos-i] += mu; # right side
        
    return h; # end h chem
    
    
def h_imp_leads(V,N,Ncoupled):
    '''
    create 1e hamiltonian for e's hopping on and off impurity levels
    V is hopping between impurity, leads
    N is total number of impurity levels
    Ncoupled is number of impurities that couple to the leads (first Ncoupled are coupled)
    '''

    assert(Ncoupled <= N);
    
    h = np.zeros((2+2*N+2,2+2*N+2)); # 2N spin orbs on imp, 1st, last 2 are neighboring lead sites

    for impi in range(1,Ncoupled+1):

        impup = 2*impi
        impdown = 2*impi + 1;
        
        # couple to left lead
        h[0, impup] = -V; # up e's
        h[impup, 0] = -V; 
        h[1, impdown] = -V; # down e's
        h[impdown, 1] = -V;

        # couple to right lead 
        h[-2, impup] = -V; # up e's
        h[impup, -2] = -V;
        h[-1, impdown] = -V; # down e's
        h[impdown, -1] = -V;
        
    return h; # end h imp leads


def h_bias(V, dot_is, norbs, verbose = 0):
    '''
    Manipulate a full siam h1e  (ie stitched already) by
    turning on bias on leads

    Args:
    - V is bias voltage
    - dot_is is list of first and last spin orb indices which are part of dot
    - norbs, int, num spin orbs in whole system

    Returns 2d np array repping bias voltage term of h1e
    '''

    assert(isinstance(dot_is, list) or isinstance(site_i, np.ndarray));

    hb = np.zeros((norbs, norbs));
    for i in range(norbs): # iter over diag of h1e

        # pick out lead orbs
        if i < dot_is[0]: # left leads
            hb[i,i] = V/2;
        elif i > dot_is[-1]: # right leads
            hb[i,i] = -V/2;

    if(verbose > 4): print("h_bias:\n", hb)
    return hb;


def h_B(B, theta, phi, site_i, norbs, verbose=0):
    '''
    Turn on a magnetic field of strength B in the theta hat direction, on site i
    This has the effect of preparing the spin state of the site
    e.g. large, negative B, theta=0 yields an up electron

    Args:
    - B, float, mag field strength
    - theta, float, mag field direction
    - norbs, int, num spin orbs in whole system
    - site_i, list, first and last spin indices that feel mag field

    Returns 2d np array repping magnetic field on given sites
    '''

    assert(isinstance(site_i, list) or isinstance(site_i, np.ndarray));
    assert(phi == 0.0);

    hB = np.zeros((norbs,norbs));
    for i in range(site_i[0],site_i[-1],2): # i is spin up, i+1 is spin down
        hB[i,i+1] = B*np.sin(theta); # implement the mag field, x part
        hB[i+1,i] = B*np.sin(theta);
        hB[i,i] = B*np.cos(theta)/2;    # z part
        hB[i+1,i+1] = -B*np.cos(theta)/2;
        
    if (verbose > 4): print("h_B:\n", hB);
    return hB;
    
    
def h_dot_1e(V,t,N):
    '''
    create 1e part of dot hamiltonian
    dot is simple model of impurity
    V is gate voltage (ie onsite energy of dot sites)
    t is hopping between the dots (N/A unless N > 1)
    N is number of dot sites
    '''

    # create h array
    h = np.zeros((2*N,2*N));
    
    # gate voltage for each dot site
    for i in range (2*N):
        h[i,i] = V; # gate voltage
        if i >= 2: # more than one dot, so couple this dot to previous one
            h[i, i-2] = -t;
            h[i-2, i] = -t;
        
    return h; # end h dot 1e


def h_hub_1e(V, t):
    '''
    1e part of two site hubbard hamiltonian, with 2nd site energy diff V relative to 1st
    downfolds into J S dot S
    '''

    h=np.zeros((4,4));

    # hopping
    h[0,2] = -t;
    h[2,0] = -t;
    h[1,3] = -t;
    h[3,1] = -t;

    # gate voltage
    h[2,2] = V;
    h[3,3] = V;

    return h;


#######################################################
#### 2e operators

def spinflip(site_i, norbs):
    '''
    define the "spin flip operator \sigma_y x \sigma_y for two qubits
    abs val of exp of spin flip operator gives concurrence
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));
    #assert( len(site_i) == 4); # concurrence def'd for 2 qubits

    # create op array (2 body!)
    sf = np.zeros((norbs,norbs, norbs, norbs));
    sf[site_i[0],site_i[0]+3,site_i[0]+2,site_i[0]+1] += -1;
    sf[site_i[0],site_i[0]+2,site_i[0]+1,site_i[0]+3] += 1;
    sf[site_i[0]+1,site_i[0]+3,site_i[0]+2,site_i[0]] += 1;
    sf[site_i[0]+1,site_i[0]+2,site_i[0]+3,site_i[0]] += -1;

    # (pq|rs) = (qp|sr) switch particle labels
    sf[site_i[0]+3,site_i[0],site_i[0]+1,site_i[0]+2] += -1;
    sf[site_i[0]+2,site_i[0],site_i[0]+3,site_i[0]+1,] += 1;
    sf[site_i[0]+3,site_i[0]+1,site_i[0],site_i[0]+2] += 1;
    sf[site_i[0]+2,site_i[0]+1,site_i[0],site_i[0]+3,] += -1;

    return sf;


#######################################################
#### 2e hams
    
def h_dot_2e(U,N):
    '''
    create 2e part of dot hamiltonian
    dot is simple model of impurity
    U is hubbard repulsion
    N is number of dot sites
    '''
    
    h = np.zeros((2*N,2*N,2*N,2*N));
    
    # hubbard repulsion when there are 2 e's on same MO
    for i in range(0,2*N,2): # i is spin up orb, i+1 is spin down
        h[i,i,i+1,i+1] = U;
        h[i+1,i+1,i,i] = U; # switch electron labels
        
    return h; # end h dot 2e


def h_hub_2e(U1, U2):
    '''
    2e part of two site hubbard ham (see h_hub_1e)   
    '''

    assert(isinstance(U2, float));
    h = np.zeros((4,4,4,4)); # 2 site
    Us = [U1,0, U2,0];
    
    # hubbard terms
    for i in [0, 2]: # i is spin up orb, i+1 is spin down
        h[i,i,i+1,i+1] = Us[i];
        h[i+1,i+1,i,i] = Us[i]; # switch electron labels
        
    return h; # end h dot 2e



#######################################################
#### stitch seperate ham arrays together


def stitch_h1e(h_imp, h_imp_leads, h_leads, h_bias, n_leads, verbose = 0):
    '''
    stitch together the various parts of the 1e hamiltonian
    the structure of the final should be block diagonal:
    (Left leads)  V_dl
            V_il  (1e imp ham) V_il
                          V_dl (Right leads)
    '''
    
    # number of spin orbs
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads[0] + 2*n_leads[1];
    n_spin_orbs = (n_lead_sos + n_imp_sos);
    
    # combine pure lead ham states
    assert(np.shape(h_leads) == np.shape(h_bias) );#should both be lead sites only
    h_leads = h_leads + h_bias;
    
    # widened ham has leads on outside, dot sites in middle
    h = np.zeros((n_spin_orbs,n_spin_orbs));
    
    # put pure lead elements on top, bottom block diag
    for i in range(2*n_leads[0]):
        for j in range(2*n_leads[0]):
            
            # the first 2*n leads indices are the left leads
            h[i,j] += h_leads[i,j];
            
    for i in range(2*n_leads[1]):
        for j in range(2*n_leads[1]):
        
            # last 2n_lead indices are right leads
            h[n_spin_orbs-1-i,n_spin_orbs-1-j] += h_leads[n_lead_sos-1-i, n_lead_sos-1-j];
      
    # fill in imp and imp-lead elements in middle
    assert(n_imp_sos+4 == np.shape(h_imp_leads)[0]); # 2 spin orbs to left, right
    assert(n_lead_sos >= 4); # assumed by later code
    for i in range(n_imp_sos + 4):
        for j in range(n_imp_sos + 4):
            
            h[2*n_leads[0] - 2 + i, 2*n_leads[0] - 2 + j] += h_imp_leads[i,j];
            if(i>1 and j>1 and i<n_imp_sos+2 and j< n_imp_sos+2): #skip first two, last two rows, columns
                h[2*n_leads[0] - 2 + i, 2*n_leads[0] - 2 + j] += h_imp[i-2,j-2];
            
    if(verbose > 4):
        print("- h_leads + h_bias:\n",h_leads,"\n- h_imp_leads:\n",h_imp_leads,"\n- h_imp:\n",h_imp);
    return h; # end stitch h1e
    
    
def stitch_h2e(h_imp,n_leads,verbose = 0):
    '''
    Put the 2e impurity hamiltonian in the center of the full leads+imp h2e matrix
    h_imp, 4D array, 2e part of impurity hamiltonian
    '''
    
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads[0] + 2*n_leads[1];
    i_imp = 2*n_leads[0]; # index where imp orbs start
    n_spin_orbs = n_imp_sos + n_lead_sos
    
    h = np.zeros((n_spin_orbs,n_spin_orbs,n_spin_orbs,n_spin_orbs));
    
    for i1 in range(n_imp_sos):
        for i2 in range(n_imp_sos):
            for i3 in range(n_imp_sos):
                for i4 in range(n_imp_sos):
                    h[i_imp+i1,i_imp+i2,i_imp+i3,i_imp+i4] = h_imp[i1,i2,i3,i4];
                    if(verbose > 2): # check 4D tensor by printing nonzero elems
                        if(h_imp[i1,i2,i3,i4] != 0):
                            print("  h_imp[",i1,i2,i3,i4,"] = ",h_imp[i1,i2,i3,i4]," --> h2e[",i_imp+i1,i_imp+i2,i_imp+i3,i_imp+i4,"]");
                        
    return h; # end stitch h2e

#####################################
#### det ops

def heisenberg(J,s2):
    '''
    Determinantal operator form of J S_e dot S_2
    2nd particle operator S_2 has spin s2
    '''

    # m2 states
    ms = [];
    m2 = s2;
    while(m2 >= -s2):
        ms.append(m2);
        m2 -= 1;

    # combined states
    states = []
    for m in ms:
        states.append([0.5,m]);
        states.append([-0.5,m]);
    states = np.array(states);

    # fill in
    H = np.zeros((len(states),len(states)));
    for si in range(len(states)):
        for sj in range(len(states)):

            # diagonal
            if(si == sj):
                H[si,sj] = states[si,0]*states[si,1];

    print(H);


#####################################
#### full systems with leads

def dot_hams(nleads, ndots, physical_params, spinstate = "", verbose = 0):
    '''
    Converts physical params into 1e and 2e parts of siam model hamiltonian
    for use with td-fci. Also does spin state preparation
    which consists of dot(s) coupled to leads:
    H_imp = H_dot = -V_g sum_i n_i + U n_{i uparrow} n_{i downarrow}
        where i are impurity sites
        for ndots > 1 have linear chain of such dots forming impurity
    
    Args:
    - nleads, tuple of ints of lead sites on left, right
    - ndots, int, num impurity sites
    - physical params, tuple of tleads, thyb, tdots, Vbias, mu, Vgate, U, B, theta
    
    Returns:
    h1e, 2d np array, 1e part of dot ham
    h2e, 2d np array, 2e part of dot ham
    input_str, string with info on all the phy params
    '''

    assert(isinstance(nleads, tuple) );
    assert(isinstance(ndots, int) );
    assert( isinstance(physical_params, tuple) );

    # unpack inputs
    norbs = 2*(sum(nleads)+ndots);
    dot_i = [nleads[0]*2, nleads[0]*2 + 2*ndots - 1 ]; # imp sites start and end, inclusive
    t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, B, theta = physical_params;
    
    input_str = "\nInputs:\n- Num. leads = "+str(nleads)+"\n- Num. impurity sites = "+str(ndots)+"\n- t_leads = "+str(t_leads)+"\n- t_hyb = "+str(t_hyb)+"\n- t_dots = "+str(t_dots)+"\n- V_bias = "+str(V_bias)+"\n- mu = "+str(mu)+"\n- V_gate = "+str(V_gate)+"\n- Hubbard U = "+str(U)+"\n- B = "+str(B)+"\n- theta = "+str(theta);
    if verbose: print(input_str);

    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = h_leads(t_leads, nleads); # leads only
    hc = h_chem(mu, nleads);   # can adjust lead chemical potential
    hdl = h_imp_leads(t_hyb, ndots, ndots); # leads talk to dot
    hd = h_dot_1e(V_gate, t_dots, ndots); # dot
    h1e = stitch_h1e(hd, hdl, hl, hc, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias
    h1e += h_bias(V_bias, dot_i, norbs , verbose = verbose); # turns on bias

    # prepare spin states
    if( spinstate == ""): # default, single dot case
        h1e += h_B(B, theta, 0.0, dot_i, norbs, verbose = verbose); # spin(theta) on dot(s)
    elif( spinstate == "aa" ): # up on LL, up on dot
        assert( ndots == 1);
        h1e += h_B(-B, theta, 0.0, np.array(range(dot_i[0]) ), norbs, verbose = verbose);
        h1e += h_B(-B, theta, 0.0, dot_i, norbs, verbose = verbose);
    elif( spinstate == "ab" ): # up on entire LL, down on dot
        assert( ndots == 1);
        h1e += h_B(-B, theta, 0.0, np.array(range(dot_i[0]) ), norbs, verbose = verbose);
        h1e += h_B(B, theta, 0.0, dot_i, norbs, verbose = verbose);
    elif( spinstate == "ab1"): # up on LL first site, down on dot
        assert( ndots == 1);
        h1e += h_B(-B, theta, 0.0, [0,1], norbs, verbose = verbose);
        h1e += h_B(B, theta, 0.0, dot_i, norbs, verbose = verbose);
    elif( spinstate == "ab-1"): # up on LL last site, down on dot
        assert( ndots == 1);
        h1e += h_B(-B, theta, 0.0, [dot_i[0]-2,dot_i[0]-1], norbs, verbose = verbose);
        h1e += h_B(B, theta, 0.0, dot_i, norbs, verbose = verbose);
    elif( spinstate == "aaa"):
        assert(ndots == 2 and theta == 0.0);
        h1e += h_B(B, np.pi, 0.0, np.array(range(dot_i[0]) ), norbs, verbose = verbose); # spin up on first lead
        h1e += h_B(-B, theta, 0.0, dot_i, norbs, verbose = verbose); # spin(theta) on dot(s)
    elif( spinstate == "abb"): # itinerant up e, spin down e's on dots
        assert(ndots == 2 and theta == 0.0);
        h1e += h_B(B, np.pi, 0.0, np.array(range(dot_i[0]) ), norbs, verbose = verbose); # spin up on first lead
        h1e += h_B(B, theta, 0.0, dot_i, norbs, verbose = verbose); # spin(theta) on dot(s)
    elif( spinstate == "a00"): # itinerant up e, singlet on dots
        assert(ndots == 2 and theta == 0.0);
        h1e += h_B(B, np.pi, 0.0, np.array(range(dot_i[0]) ), norbs, verbose = verbose); # spin up on first lead
    else: assert(False); # invalid option
        
    # 1e ham finished now
    if(verbose > 2):
        np.set_printoptions(precision = 4, suppress = True);
        print("\n- Full one electron hamiltonian = \n",h1e);
        np.set_printoptions();
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        np.set_printoptions(precision = 4, suppress = True);
        print("\n- Nonzero h2e elements = ");
    hd2e = h_dot_2e(U,ndots);
    h2e = stitch_h2e(hd2e, nleads, verbose = verbose);

    return h1e, h2e, input_str; #end dot hams


def hub_hams(nleads, nelecs, physical_params, verbose = 0):
    '''
    Converts physical params into 1e and 2e parts of two site hubbard
    with leads, and with spin preparation, if B nonzero
    ie intended for td-fci
    
    Two site hubbard model maps onto two level spin impurity, with U
    This then downfolds into Jeff S1 dot S2 spin impurity ham
    
    Args:
    - nleads, tuple of ints of lead sites on left, right
    - nelecs, tuple of number es, 0 due to All spin up formalism
    - physical params, tuple of tleads, thyb, tdots, Vbias, mu, Vgate, U, B, theta
    
    Returns:
    h1e, 2d np array, 1e part
    g2e, 2d np array, 2e part
    input_str, string with info on all the phy params
    '''

    assert(isinstance(nleads, tuple) );
    assert(isinstance(nelecs, tuple) );
    assert(isinstance(physical_params, tuple) );

    # unpack inputs
    ndots = 2;
    norbs = 2*(sum(nleads)+ndots);
    dot_i = [nleads[0]*2, nleads[0]*2 + 2*ndots - 1 ]; # imp sites start and end, inclusive
    t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, B, theta = physical_params;
    
    input_str = "\nInputs:\n- Num. leads = "+str(nleads)+"\n- Num. impurity sites = "+str(ndots)+"\n- nelecs = "+str(nelecs)+"\n- t_leads = "+str(t_leads)+"\n- t_hyb = "+str(t_hyb)+"\n- t_dots = "+str(t_dots)+"\n- V_bias = "+str(V_bias)+"\n- mu = "+str(mu)+"\n- V_gate = "+str(V_gate)+"\n- Hubbard U = "+str(U)+"\n- B = "+str(B)+"\n- theta = "+str(theta);
    if verbose: print(input_str);

    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = h_leads(t_leads, nleads); # leads only
    hc = h_chem(mu, nleads);   # can adjust lead chemical potential
    hdl = h_imp_leads(t_hyb, ndots, ndots -1); # leads talk to 1st dot only
    hd = h_hub_1e(V_gate, t_hyb); # 2 site hubbard <-> 2 level spin imp
    h1e = stitch_h1e(hd, hdl, hl, hc, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias
    h1e += h_bias(V_bias, dot_i, norbs , verbose = verbose); # turns on bias

    # prep spin
    h1e += h_B(-B, theta, 0.0, [0,1], norbs, verbose = verbose);
    h1e += h_B(B, theta, 0.0, dot_i, norbs, verbose = verbose); # spin(theta) on dot(s)

    # 1e ham finished now
    if(verbose > 2):
        np.set_printoptions(precision = 4, suppress = True);
        print("\n- Full one electron hamiltonian = \n",h1e);
        np.set_printoptions();
        
    # 2e terms: U on imp level 2 only
    g2e = h_hub_2e(0.0,U);
    if(verbose > 2):
        np.set_printoptions(precision = 4, suppress = True);
        print("\n- Nonzero h2e elements = ");
        print(np.shape(g2e));
    g2e = stitch_h2e(g2e,nleads,verbose = verbose);

    return h1e, g2e, input_str;


    
#####################################
#### exec code

if(__name__ == "__main__"):

    pass;


