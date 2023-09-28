'''
'''

from utils import plot_fit, load_dIdV, fit_wrapper
from landauer import En, dI_of_Vb_zero

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as scipy_integ

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkred", "darkgreen", "darkorange", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","s","^","d","*","+","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# units
kelvin2eV =  8.617e-5; # units eV/K
e2overh= 7.748e-5 *1e9/2; # units nA/volt
muBohr = 5.788e-5;     # units eV/T
gfactor = 2;

###############################################################
#### fitting dI/dV with background and oscillations

def make_EC_list(EC):
    '''
    '''
    return np.array([EC]); 

def make_EC_dist(EC, EC_not, n=2):
    '''
    '''
    if(not isinstance(EC, np.ndarray)): raise TypeError;

    delta = n/EC_not;
    return (delta*EC)**n *np.exp(-delta*EC)*(delta/np.math.factorial(n));

def make_EC_square(EC, EC_not, d_not):
    if(not isinstance(EC, np.ndarray)): raise TypeError;
    assert(EC_not - d_not > EC[0] and EC_not + d_not < EC[-1]);

    ret = np.zeros_like(EC);
    ret[abs(EC-EC_not)<d_not] = np.ones_like(ret[abs(EC-EC_not)<d_not]);
    ret = ret/(2*d_not);
    plt.plot(EC, ret);
    plt.show();
    return ret

def dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC): 
    '''
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;

    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    ret = np.zeros_like(Vb);
    for ECval in make_EC_list(EC):
        ret += dI_of_Vb_zero(Vb-V0, mymu0, Gamma, ECval, 0.0, ns);
    return tau0*ret; # overall factor of tau0

def dIdV_lorentz_integrand(Vb, V0, tau0, Gamma, EC):
    '''
    '''
    if(not isinstance(EC, np.ndarray)): raise TypeError;

    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    Vb = Vb - V0; # shifted

    # chem potentials
    muL, muR = mymu0+Vb, mymu0;

    # conductance
    conductance = np.zeros_like(EC);
    for n in ns:
        Enval = En(n,EC,Vb);
        conductance += 1/(1+(muL-Enval)*(muL-Enval)/(Gamma*Gamma) );
        conductance += 1/(1+(muR-Enval)*(muR-Enval)/(Gamma*Gamma) );
    conductance = e2overh*(1/2)*conductance; # return val of landauer, dI_of_Vb_zero
    return tau0*conductance; # overall factor of tau0


####################################################################
#### run

if(__name__ == "__main__"):

    Vbvals = np.linspace(-0.1,0.1,int(200));

    # fitting params
    V0_not, tau0_not, Gamma_not, EC_not =0.0, 0.011/4, 0.002, 0.005; 
    single_vals = dIdV_lorentz_zero(Vbvals, V0_not, tau0_not, Gamma_not, EC_not);

    # EC distribution
    EC_mesh = np.linspace(0.0,10*EC_not,int(1e3));
    square = True;
    if(square): EC_dist = make_EC_square(EC_mesh, EC_not, 0.01*EC_not);
    else: EC_dist = make_EC_dist(EC_mesh, EC_not);
    plt.plot(EC_mesh, EC_dist, label=scipy_integ(EC_dist, EC_mesh));
    plt.legend();
    plt.show();
    
    # integrate over EC dist
    dist_vals = np.empty_like(Vbvals);
    for Vbi in range(len(Vbvals)):
        integrand = dIdV_lorentz_integrand(Vbvals[Vbi], V0_not, tau0_not, Gamma_not, EC_mesh);
        dist_vals[Vbi] = scipy_integ(integrand*EC_dist, EC_mesh);
        
    # plot
    fig, ax = plt.subplots();
    ax.plot(Vbvals, single_vals);
    ax.plot(Vbvals, dist_vals);
    plt.show();

    

