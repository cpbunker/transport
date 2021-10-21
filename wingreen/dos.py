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

    def __init__(self, j, c, ends):
        '''
        j = sites
        c = coefficients multiplying sites, defaults to 1
        ends = first and last allowed sites in the chain
        '''

        # defaults
        if(c is None): c = np.ones_like(j, dtype = float);
        if( ends is None): ends = (int(-1e6),int(1e6) );

        # inputs
        assert(isinstance(j,np.ndarray));
        assert(isinstance(c,np.ndarray));
        assert(len(j) == len(c) );
        assert(len(ends) == 2 and isinstance(ends[0], int) );

        # endpts
        self.ends = ends;
        
        # screen out anything beyond ends
        self.j = j[j >= ends[0]][ j[j >= ends[0]] <= ends[1] ];

        # coefficients of nonzero sites, similarly screened
        self.c = c[j >= ends[0]][ j[j >= ends[0]] <= ends[1] ];


    def __str__(self):
        stri = "";
        for ji in range(len(self.j)):
            stri += str(self.c[ji])+"|"+str(self.j[ji])+"> + ";
        return stri;
    

    def __add__(self, other): # add two sites, not in place

        assert(self.ends == other.ends); # need to correspond to same chain

        return site(np.append(self.j, other.j), np.append(self.c, other.c), self.ends);


    def __mul__(self, other): # inner product as orthonormal basis

        if( isinstance(other, int) or isinstance(other, float) ): # multiply coefficients, not in place
            return site(self.j, self.c*other, self.ends);

        else:
            
            assert(self.ends == other.ends); # need to correspond to same chain
            
            mysum = 0.0;
            for ji in range(len(self.j)):
                for ii in range(len(other.j)):
                    if(self.j[ji] == other.j[ii]):
                        mysum += self.c[ji]*other.c[ii];
            return mysum;


    def H(self,t, verbose = 0):
        '''
        hamiltonian takes site to nearest neighbors
        not in place

        assume hopping same between all sites
        assume onsite energy of all sites = same = 0
        '''
        newj = [];
        newc = [];
        for ji in range(len(self.j)):
            newj.append(self.j[ji]-1);
            newc.append(self.c[ji]);
            newj.append(self.j[ji]+1);
            newc.append(self.c[ji]);
        return site(np.array(newj), t*np.array(newc), self.ends);

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
        

def gen_as_bs(site0, t, depth, verbose = 0):
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
    assert(isinstance(t, float));

    # return variables
    a_s = [];
    b_s = [];

    # zeroth iteration
    mu = 0.0; # choose energy shift
    a_s.append(mu); # by definition
    b_s.append(1); # b_(-1) set to one by definition

    # first iteration
    nmin1prime = site0; # |0'>
    nprime = nmin1prime.H(t) + nmin1prime*(-a_s[0]); # |1'>
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
        a_s.append((nprime*nprime.H(t))/divisor );

        #print(">>>", nprime);

        # b_(n-1)'
        divisor = nmin1prime*nmin1prime;
        if( divisor == 0.0): divisor = 1.0; # to avoid nans
        b_s.append((nmin1prime*nprime.H(t))/divisor);

        # update states
        nplus1prime = nprime.H(t) + nprime*(-a_s[-1]) + nmin1prime*(-b_s[-1]);
        nmin1prime = nprime*1; # to copy
        nprime = nplus1prime.condense()*1;

    return np.array(a_s), np.array(b_s);


def resolvent(site0, E, t, depth, verbose = 0):
    '''
    Calculate resolvent green's function according to Haydock 2.6
    (continued fraction form)
    '''

    # get coefs
    a_s, b_s = gen_as_bs(site0, t, depth, verbose = verbose);

    # start from bottom
    bG = (E-a_s[-1])/2 *(1- np.lib.scimath.sqrt(1-4*b_s[-1]/((E-a_s[-1])*(E-a_s[-1])) ) );

    # iter over rest
    for i in range(2, depth+1):

        bG = b_s[-i]/(E - a_s[-i] - bG); # update recursively
        if(verbose > 2): print("G (n = "+str(depth - i)+") = ",bG);

    return bG;


if __name__ == "__main__":

    verbose = 5;

    # test site objects
    if False:
        endpts = (-1,1);
        interest = site(np.array([0]), np.array([0.1]), endpts);
        print("|0> = ",interest);
        print("H|0> = ",interest.H(-0.5));
        print("|0> = ",interest);
        print("2|0> = ", (interest+ interest).condense());
        print("|0> = ",interest);
        x = interest + interest.H(-0.5) + interest*10
        print(x);
        print(x.condense() )
        assert(False);

    # site of interest = central site in 3 site chain
    interest = site(np.array([0]), None, (-1,1));

    # determine G at site of interest
    # args: site of interest, energy, t, depth
    tl = -0.3;
    epsilon = abs(1/2); # imag broadening
    Evals = np.linspace(-4.5,4.5,100) + np.complex(0, epsilon)
    G00 = resolvent(interest, Evals, tl, 6, verbose = verbose);

    # confirm by plotting
    import matplotlib.pyplot as plt
    plt.scatter(Evals, (-1/np.pi)*np.imag(G00), marker = 's');
    plt.plot(Evals, Evals/(Evals*Evals - 2*tl*tl ) );
    plt.axvline(np.sqrt(2)*abs(tl), color = "black", linestyle = "dashed");
    plt.show();

    nstates = np.trapz((-1/np.pi)*np.imag(G00), Evals);
    print(nstates);
    
   








