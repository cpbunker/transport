'''
Christian Bunker
M^2QM at UF
October 2021

Expression for the surface density of states around a particular tight binding
site using the resolvent Green's function G00 for a nearest neighbor
hamiltonian. Calculate G00 using recursive formulation of Haydock
'''

import numpy as np

class site(object):
    '''
    Represents tight binding basis vector, in discrete 1d chain
    1d chain is truncated by ends but these can be made arbitrarily large
    
    Computes how such vectors add, multiply
    Computes action of 1d nearest neighbor hopping hamiltonian H on such vectors
    '''

    def __init__(self, j, c, ends, ham):
        '''
        j = sites
        c = coefficients multiplying sites, defaults to 1
        ends = first and last allowed sites in the chain
        '''

        # defaults
        if(c is None): c = np.ones_like(j, dtype = float);
        if( ends is None): ends = (0,int(1e6) );
        if(ham is None): ham = "defH";

        # inputs
        assert(isinstance(j,np.ndarray));
        assert(isinstance(c,np.ndarray));
        assert(len(j) == len(c) );
        assert(len(ends) == 2 and isinstance(ends[0], int) );
        
        # screen out anything beyond ends
        self.j = j[j >= ends[0]][ j[j >= ends[0]] <= ends[1] ];

        # coefficients of nonzero sites, similarly screened
        self.c = c[j >= ends[0]][ j[j >= ends[0]] <= ends[1] ];

        # endpts
        self.ends = ends;

        # hamiltonian
        self.ham = ham; # keyword for calling a function via H() method


    #### overloads

    def __str__(self):
        stri = "";
        for ji in range(len(self.j)):
            stri += str(self.c[ji])+"|"+str(self.j[ji])+"> + ";
        return stri;
    

    def __add__(self, other): # add two sites, not in place

        assert(self.ends == other.ends); # need to correspond to same chain

        return site(np.append(self.j, other.j), np.append(self.c, other.c), self.ends, self.ham);


    def __mul__(self, other): # inner product as orthonormal basis

        if( isinstance(other, int) or isinstance(other, float) ): # multiply coefficients, not in place
            return site(self.j, self.c*other, self.ends, self.ham);

        else:
            
            assert(self.ends == other.ends); # need to correspond to same chain
            
            mysum = 0.0;
            for ji in range(len(self.j)):
                for ii in range(len(other.j)):
                    if(self.j[ji] == other.j[ii]):
                        mysum += self.c[ji]*other.c[ii];
            return mysum;

    #### hamiltonians

    def H(self):
        if self.ham == "defH":
            return self.defH();
        else: raise(TypeError);
        
    def defH(self):
        '''
        hamiltonian for uniform nearest neighbor hopping, mu=0
        default H option
        not in place

        assume hopping between all sites is same
        assume onsite energy of all sites is same
        Acts as if mu = 0 and t = -1, but this can be generalized by treating
        E as a scaled energy, E -> (E-mu)/t
        
        '''
        
        newj = [];
        newc = [];
        for ji in range(len(self.j)):
            newj.append(self.j[ji]-1);
            newc.append(self.c[ji]);
            newj.append(self.j[ji]+1);
            newc.append(self.c[ji]);
        return site(np.array(newj), -np.array(newc), self.ends, self.ham);
    
        
    #### misc
    
    def condense(self):
        '''
        Combine repeated states, in place
        '''

        newj = [];
        newc = [];
        for ji in range(len(self.j)):
            if self.j[ji] not in newj: # add site and coef
                newj.append(self.j[ji]);
                newc.append(self.c[ji]);
            else: # only update coef
                whichi = newj.index(self.j[ji]);
                newc[whichi] += self.c[ji];

        self.j = np.array(newj);
        self.c = np.array(newc);
        return self;
        

def gen_as_bs(site0, depth, verbose = 0):
    '''
    generate coefficients a_n', b_n' of fictitious states
    fictitious represent local environment of site of interest, site0

    local environment encoded in hamiltonian, which is represented by the
    site.H method. This method assumse nearest neighbor tight binding
    hamiltonian char'd only by:
    - onsite energy mu, same on each site, set to 0
    - hopping t

    stops iterating at n' = depth
    '''

    # check inputs
    assert(isinstance(site0, site) );

    # return variables
    a_s = [];
    b_s = [];

    # zeroth iteration
    mu = 0.0; # choose energy shift
    a_s.append(mu); # by definition
    b_s.append(1); # b_(-1) set to one by definition

    # first iteration
    nmin1prime = site0; # |0'>
    nprime = nmin1prime.H() + nmin1prime*(-a_s[0]); # |1'>
    if(False):
        print("|0'> = ",nmin1prime);
        print("|1'> = ",nprime);
        a1p = -4*mu*t*t/(2*t*t + mu*mu);
        print("predicted a1' = ", a1p);
        a2p = -4*t*t*mu*a1p*(mu+a1p)
        a2p = a2p/(2*t*t*np.power(mu+a1p,2) + mu*mu*a1p*a1p);
        print("predicted a2 = ", a2p);

    # now iterate
    for itr in range(depth-1):

        # a_n'
        divisor = nprime*nprime;
        if( divisor == 0.0): divisor = 1.0; # to avoid nans
        a_s.append((nprime*nprime.H())/divisor );

        #print(">>>", nprime);

        # b_(n-1)'
        divisor = nmin1prime*nmin1prime;
        if( divisor == 0.0): divisor = 1.0; # to avoid nans
        b_s.append((nmin1prime*nprime.H())/divisor);

        # update states
        nplus1prime = nprime.H() + nprime*(-a_s[-1]) + nmin1prime*(-b_s[-1]);
        nmin1prime = nprime*1; # to copy
        nprime = nplus1prime.condense()*1;

    return np.array(a_s), np.array(b_s);


def surface_gf(site0, E, depth, verbose = 0):
    '''
    Calculate resolvent green's function according to Haydock 2.6
    (continued fraction form)
    '''

    # check inputs
    assert(isinstance(site0, site) );

    # get coefs
    a_s, b_s = gen_as_bs(site0, depth, verbose = verbose);

    # start from bottom
    bG = (E-a_s[-1])/2 *(1- np.lib.scimath.sqrt(1-4*b_s[-1]/((E-a_s[-1])*(E-a_s[-1])) ) );

    # iter over rest
    for i in range(2, depth+1):

        bG = b_s[-i]/(E - a_s[-i] - bG); # update recursively
        if(verbose > 2): print("G (n = "+str(depth - i)+") = ",bG);

    return bG;


def surface_dos(length, hamkw, depth, Evals, iE, verbose = 0):
    '''
    For a chain of given length, with physics specified by hamkw,
    which selects the hamiltonian method of the site object to use,
    use the haydock method of the resolvent function to calculate G

    Then use G to get the surface dos g(E) at the far end
    E char'd by E1 < E < E2 with + i epsilon small imag part
    '''

    # energy sweep
    Evals += complex(0, iE); # add small imag part

    # site object
    site0 = site(np.array([0]), None, (0,length-1), hamkw);

    # green's function, vectorized
    G = surface_gf(site0, Evals, depth, verbose = verbose);

    # dos, vectorized
    gE = (-1/np.pi)*np.imag(G);

    return gE, Evals;


def junction_gf(g_L, t_L, g_R, t_R, E, H_SR):
    '''
    Given the surface green's function in the leads, as computed above,
    compute the gf at the junction between the leads, aka scattering region.
    NB the junction has its own local physics def'd by H_SR

    Args:
    - g_L, 1d arr, left lead noninteracting gf at each E
    - t_L, 2d arr, left lead coupling, constant in E
    - g_R, 1d arr, right lead noninteracting gf at each E
    - t_R, 2d arr, right lead coupling, constant in E
    - E, 1d array, energy values
    '''

    # check inputs
    assert(np.shape(t_L) == np.shape(H_SR) );
    assert(len(g_L) == len(E) );

    # vectorize by hand
    G = [];
    for Ei in range(len(E)): # do for each energy

        # gL, gR as of now are just numbers at each E, but should be matrices
        # however since leads are defined to be noninteracting, just identity matrices
        g_Lmat = g_L[Ei]*np.eye(*np.shape(H_SR));
        g_Rmat = g_R[Ei]*np.eye(*np.shape(H_SR));

        # integrate out leads, using self energies
        Sigma_L = np.dot(np.linalg.inv(g_Lmat),-t_L);
        Sigma_L = np.dot( -t_L, Sigma_L);
        Sigma_R = np.dot(np.linalg.inv(g_Rmat),-t_R);
        Sigma_R = np.dot( -t_R, Sigma_R);

        # local green's function
        G.append(np.linalg.inv( E[Ei]*np.eye(*np.shape(H_SR)) - H_SR - Sigma_L - Sigma_R));

    return np.array(G);


if __name__ == "__main__":

    verbose = 5;

    # test site objects
    if False:
        endpts = (-1,1);
        interest = site(np.array([0]), np.array([0.1]), endpts, None);
        print("|0> = ",interest);
        print("H|0> = ",interest.H() );
        print("|0> = ",interest);
        print("2|0> = ", (interest+ interest).condense());
        print("|0> = ",interest);
        x = interest + interest.H() + interest*10
        print(x);
        print(x.condense() )

    # site of interest = central site in 3 site chain
    interest = site(np.array([0]), None, (0,7), None);
    print(interest.ham )
    G00 = resolvent(interest, 1/0.3, 10, verbose = verbose);
    print(">>",G00);

    for eps in [0.001, 0.01, 0.1,0.5]:

        # test the dos
        # args = chain length, hamiltonian keyword, recursion depth, start E, stop E, epsilon
        gvals, Evals = surface_dos(5, None, 10, -3.0, 3.0, eps);

        # visualize
        import matplotlib.pyplot as plt
        plt.plot(Evals, gvals);
        plt.show();
    
   








