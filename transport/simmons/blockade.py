'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''

import numpy as np
import matplotlib.pyplot as plt

def nFD(epsilon, kBT):
    '''
    '''
    
    return 1/(np.exp(epsilon/kBT)+1);

def Lorentzian(x, Gamma):
    numerator = Gamma/(2*np.pi);
    denominator = x*x + (Gamma/2)*(Gamma/2);
    return numerator/denominator;

def Pn_recursive(n, Vb, EC, mutilde, kBT):
    '''
    '''
    
    if(n==0):
        return np.ones_like(Vb);
    else:
        numerator   = nFD( (2*n-1)*EC+mutilde-Vb/2, kBT) + nFD( (2*n-1)*EC+mutilde+Vb/2, kBT);
        denominator = nFD(-(2*n-1)*EC-mutilde+Vb/2, kBT) + nFD(-(2*n-1)*EC-mutilde-Vb/2, kBT);
        retval = (numerator/denominator)*Pn_recursive(n-1, Vb, EC, mutilde, kBT);
        return retval;

def I_of_Vb(Vb, EC, mutilde, kBT, nmax):
    '''
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(EC, float)): raise TypeError;
    if(not isinstance(mutilde, float)): raise TypeError;

    # get current
    current = np.zeros_like(Vb);
    for n in range(nmax+1):
        Pn = Pn_recursive(n, Vb, EC, mutilde, kBT);
        current +=  nFD( EC*( 2*n+1) + (mutilde-Vb/2),kBT)*Pn;
        current += -nFD( EC*(-2*n+1) - (mutilde-Vb/2),kBT)*Pn;
    return (94/4)*current; #### fudge factor!

if(__name__ == "__main__"):

    # physical params, in eV
    # when mutilde+2*EC < 0 there are instabilities
    my_mutilde = 0.001;
    my_EC = 0.005; # should be of order of observed conductance osc period (10s of meV)
    kelvin2eV =  8.617e-5;
    my_temp = 5.0*kelvin2eV;
    nmax = 10;

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    I_Vb_max = 94; # nano amps

    if False: # test Pn
        my_Vb = np.array([-Vb_max,0.0,Vb_max]);
        ns = np.array(range(nmax));
        Pns = np.empty((len(ns), len(my_Vb)), dtype=float);
        for ni in range(len(ns)):
            print("\tn = ",ns[ni])
            Pns[ni,:] = Pn_recursive(ns[ni], my_Vb, my_EC, my_mutilde, my_temp);

        # plot Pn vs n
        fig, ax = plt.subplots();
        for Vbi in range(len(my_Vb)):
            ax.plot(ns[1:], Pns[1:,Vbi]);
        plt.show();

    if False:
        from utils import load_IVb
        V_exp, I_exp = load_IVb("KExp.txt","Mn/",5);
        fig, ax = plt.subplots();
        ax.plot(V_exp, I_exp);
        plt.show();
        assert False

    if True: # get I(Vb)
        Vbs = np.linspace(-Vb_max,Vb_max,100);
        Is = I_of_Vb(Vbs, my_EC, my_mutilde, my_temp, nmax);

        # plot differential conductance
        fig, ax = plt.subplots();
        ax.plot(Vbs, Is);
        ax.plot(Vbs,np.gradient(Is));

        # fit to Lorentzian
        x0 = 0.025/Vb_max;
        height = 3/(my_temp/kelvin2eV);
        GammaT = 2/(np.pi*height);
        #ax.plot(Vbs, Lorentzian( Vbs/Vb_max-x0, GammaT), color="black");
        #ax.plot(Vbs, Lorentzian( Vbs/Vb_max+x0, GammaT), color="black");

        # format
        plt.tight_layout();
        plt.show();

    
    
