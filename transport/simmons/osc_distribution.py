'''
'''

from utils import plot_fit, load_dIdV, fit_wrapper
from landauer import En

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

    # visualize
    fig, ax = plt.subplots()
    ax.plot(EC, ret, label=scipy_integ(ret, EC));
    ax.set_xlabel("$E_C$"); ax.set_ylabel("PDF"); plt.legend();
    plt.show();
    return ret;

# antisymmetry in coupling
def Sfunc(Sdummy, Edummy, Eintercept = 0.0005):
    if(not isinstance(Edummy, np.ndarray)): raise TypeError;
    slope = Sdummy/Eintercept;
    ret = slope*Edummy - Sdummy;
    ret[Edummy<Eintercept] = np.zeros_like(ret[Edummy<Eintercept]);
    return ret;

    slope = -Sdummy/Eintercept;
    ret = slope*Edummy + Sdummy;
    ret[Edummy>Eintercept] = np.zeros_like(ret[Edummy>Eintercept]);
    return ret;

def dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC, Sparam=0): 
    '''
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;

    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    Vb = Vb - V0; # shifted

    # chem potentials
    muL, muR = mymu0+Vb, mymu0;

    # conductance
    conductance = np.zeros_like(Vb);
    for n in ns:
        Enval = En(n,EC,Vb) + Sfunc(Sparam, np.array([EC]))[0]*Vb;
        term1 = 1/(1+(muL-Enval)*(muL-Enval)/(Gamma*Gamma) );
        term2 = 1/(1+(muR-Enval)*(muR-Enval)/(Gamma*Gamma) );
        conductance += (1/2)*(term1+term2) - Sparam*(term1-term2);
    conductance = e2overh*conductance; # return val of landauer, dI_of_Vb_zero
    return tau0*conductance; # overall factor of tau0

def dIdV_lorentz_integrand(Vb, V0, tau0, Gamma, EC, Sparam=0):
    '''
    '''
    if(not isinstance(EC, np.ndarray)): raise TypeError;

    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    Vb = Vb - V0; # shifted

    # chem potentials
    muL, muR = mymu0+Vb, mymu0;
    if False:
        fig, ax = plt.subplots();
        y = Sfunc(Sparam, EC);
        ax.plot(EC, y)
        plt.show()
        assert False

    # conductance
    conductance = np.zeros_like(EC);
    for n in ns:
        Enval = En(n,EC,Vb) + Sfunc(Sparam, EC)*Vb;
        term1 = 1/(1+(muL-Enval)*(muL-Enval)/(Gamma*Gamma) );
        term2 = 1/(1+(muR-Enval)*(muR-Enval)/(Gamma*Gamma) );
        conductance += (1/2)*(term1+term2) - Sparam*(term1-term2);
    conductance = e2overh*conductance; # return val of landauer, dI_of_Vb_zero
    return tau0*conductance; # overall factor of tau0

####################################################################
#### run

if(__name__ == "__main__"):

    Vbvals = np.linspace(-0.1,0.1,int(400));

    # fitting params
    V0_not, tau0_not, Gamma_not, EC_not, Snot = 0.0, 0.011/4, 0.001, 0.005, 0.15; 
    single_vals = dIdV_lorentz_zero(Vbvals, V0_not, tau0_not, Gamma_not, EC_not, 0.0);

    # EC distribution
    EC_mesh = np.linspace(0.0,10*EC_not,int(1e3));
    square = True;
    if(square): EC_dist = make_EC_square(EC_mesh, EC_not, 0.1*EC_not);
    else: EC_dist = make_EC_dist(EC_mesh, EC_not);
    
    # integrate over EC dist
    dist_vals = np.empty_like(Vbvals);
    for Vbi in range(len(Vbvals)):
        integrand = dIdV_lorentz_integrand(Vbvals[Vbi], V0_not, tau0_not, Gamma_not, EC_mesh, Snot);
        dist_vals[Vbi] = scipy_integ(integrand*EC_dist, EC_mesh);
        
    # plot
    fig, ax = plt.subplots();
    ax.plot(Vbvals, single_vals);
    ax.plot(Vbvals, dist_vals);
    ax.set_xlabel("$V_b$");
    ax.set_ylabel("$dI/dV_b$");
    ax.set_title("$\Gamma = {:.3f}, E_C = {:.3f}, S = {:.3f}$".format(Gamma_not, EC_not, Snot)); 
    plt.show();

    

