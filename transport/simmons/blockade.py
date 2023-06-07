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
    # Need to properly normalize
    # norm is a function of (Vb)
    
    if(n==0):
        return np.ones_like(Vb);
    else:
        numerator   = nFD( (2*n-1)*EC+mutilde-Vb/2, kBT) + nFD( (2*n-1)*EC+mutilde+Vb/2, kBT);
        denominator = nFD(-(2*n-1)*EC-mutilde+Vb/2, kBT) + nFD(-(2*n-1)*EC-mutilde-Vb/2, kBT);
        retval = (numerator/denominator)*Pn_recursive(n-1, Vb, EC, mutilde, kBT);
        return retval;

def I_of_Vb(Vb, EC, mutilde, kBT, nmax, return_Pn = False):
    '''
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(EC, float)): raise TypeError;
    if(not isinstance(mutilde, float)): raise TypeError;

    # get Pns
    Pns = np.empty((nmax+1,len(Vb)),dtype=float);
    for n in range(nmax+1):
        Pns[n,:] = Pn_recursive(n, Vb, EC, mutilde, kBT);
    # normalize for each Vb
    for Vbi in range(len(Vb)): 
        #Pns[:,Vbi] = Pns[:,Vbi]/np.sum(Pns[:,Vbi]);
        pass

    if(return_Pn):
        assert(len(Vb)==1);
        return Pns[:,0];

    # get current
    current = np.zeros_like(Vb);
    for n in range(nmax+1):
        current +=  nFD( EC*( 2*n+1) + (mutilde-Vb/2),kBT)*Pns[n];
        current += -nFD( EC*(-2*n+1) - (mutilde-Vb/2),kBT)*Pns[n];
    return current; 

if(__name__ == "__main__"):

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    kelvin2eV =  8.617e-5;

    if False: # plot differential conductance vs EC
        fig, ax = plt.subplots();
        ECvals = np.array([5,10,20])/1000;
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        nmax = 10;
        # when mutilde+2*EC < 0 there are instabilities
        my_mutilde = 0.0;
        my_temp = 50.0*kelvin2eV; # peak separation should be strictly T independent

        for ECvali in range(len(ECvals)):
            Is = I_of_Vb(Vbs, ECvals[ECvali], my_mutilde, my_temp, nmax);
            ax.plot(Vbs,1e-3*ECvali+np.gradient(Is),label = "{:.3f} eV".format(ECvals[ECvali]));
            ax.axvline(-2*ECvals[ECvali],color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $\mu = $ {:.3f} eV".format(my_temp/kelvin2eV,my_mutilde));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot differential conductance vs temp
                # TODO:  get rid of linear T peak shift
        fig, ax = plt.subplots();
        Tvals = np.array([5,10,20])*kelvin2eV;
        Vbs = np.linspace(-Vb_max,2*Vb_max,int(1e5));

        # physical params, in eV
        nmax = 10;
        # when mutilde+2*EC < 0 there are instabilities
        my_mutilde = 0.0;
        my_EC = 0.010; # should be of order of observed conductance osc period (10s of meV)

        for Tval in Tvals:
            Is = I_of_Vb(Vbs, my_EC, my_mutilde, Tval, nmax);
            ax.plot(Vbs, np.gradient(Is), label = "{:.1f} K".format(Tval/kelvin2eV));

        # format
        ax.set_title( "$E_C = $ {:.3f} eV, $\mu = $ {:.3f} eV".format(my_EC, my_mutilde));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot differential conductance vs mu tilde
        fig, ax = plt.subplots();
        muvals = np.array([0,1,5])/1000;
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        nmax = 10;
        # when mutilde+2*EC < 0 there are instabilities
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.010; # should be of order of observed conductance osc period (10s of meV)

        for muval in muvals:
            Is = I_of_Vb(Vbs, my_EC, muval, my_temp, nmax);
            ax.plot(Vbs,np.gradient(Is), label = "{:.3f} eV".format(muval));
        ax.axvline(-2*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$E_C = $ {:.3f} eV, $T = $ {:.1f} K".format(my_EC,my_temp/kelvin2eV));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if True: # plot current vs n max
        fig, ax = plt.subplots();
        muvals = np.array([0,1,5])/1000;
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        nmax = 1;
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.005; # should be of order of observed conductance osc period (10s of meV)
        my_mutilde = 0.00;

        nvals = np.array([0,1,2]);
        for nval in nvals:
            Is = I_of_Vb(Vbs, my_EC, my_mutilde, my_temp, nval);
            ax.plot(Vbs,Is, label = str(nval));
        for integer in [-3,-1,1,3]:
            ax.axvline(integer*2*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $eV_g+\mu_0 = $ {:.3f} eV".format(my_temp/kelvin2eV, my_EC, my_mutilde));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$I(V_b)$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # Pn vs T

        # physical params, in eV
        nmax = 10;
        my_mutilde = 0.0; # when mutilde+2*EC < 0 there are instabilities
        my_EC = 0.010; # should be of order of observed conductance osc period (10s of meV)
        my_Vb = 0.0;
        
        Tvals = np.array([5,25,50])*kelvin2eV;
        ns = np.array(range(nmax+1));
        Pns = np.empty((len(Tvals), len(ns)), dtype=float);
        for Tvali in range(len(Tvals)):
            Pns[Tvali] = I_of_Vb(np.array([my_Vb]), my_EC, my_mutilde, Tvals[Tvali], nmax, return_Pn=True);

        # plot Pn vs n
        fig, ax = plt.subplots();
        for Tvali in range(len(Tvals)):
            ax.plot(ns, Pns[Tvali], label="$T = ${:.1f}".format(Tvals[Tvali]/kelvin2eV));
            print(">>>",np.sum(Pns[Tvali]));
        ax.set_title( "$E_C = $ {:.3f} eV, $eV_g+\mu_0 = $ {:.3f} eV, $V_b = $ {:.3f} eV".format(my_EC,my_mutilde, my_Vb));
        ax.set_xlabel("$n$");
        ax.set_ylabel("$P_n$");
        plt.legend();
        plt.show();


    if False: # Pn vs Vb

        # physical params, in eV
        nmax = 10;
        my_temp = 25.0*kelvin2eV;
        my_EC = 0.010; # should be of order of observed conductance osc period (10s of meV)
        my_mutilde = 0.0; # when mutilde+2*EC < 0 there are instabilities
        
        Vbvals = np.array([-Vb_max,0.0,0.5*Vb_max,Vb_max]);
        ns = np.array(range(nmax+1));
        Pns = np.empty((len(Vbvals), len(ns)), dtype=float);
        for Vbvali in range(len(Vbvals)):
            Pns[Vbvali] = I_of_Vb(Vbvals[Vbvali:Vbvali+1], my_EC, my_mutilde, my_temp, nmax, return_Pn=True);

        # plot Pn vs n
        fig, ax = plt.subplots();
        for Vbvali in range(len(Vbvals)):
            ax.plot(ns, Pns[Vbvali], label="$eV_b = ${:.3f}".format(Vbvals[Vbvali]));
            print(">>>",np.sum(Pns[Vbvali]));
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $eV_g+\mu_0 = $ {:.3f} eV".format(my_temp/kelvin2eV, my_EC, my_mutilde));
        ax.set_xlabel("$n$");
        ax.set_ylabel("$P_n$");
        plt.legend();
        plt.show();

    if False: # Pn vs EC

        # physical params, in eV
        nmax = 10;
        my_mutilde = -0.0; # when mutilde+2*EC < 0 there are instabilities
        my_temp = 50.0*kelvin2eV;
        my_Vb = 0.0;
        
        ECvals = np.array([5,10,20])/1000;
        ns = np.array(range(nmax));
        Pns = np.empty((len(ECvals), len(ns)), dtype=float);
        for ni in range(len(ns)):
            print("\tn = ",ns[ni]);
            for ECvali in range(len(ECvals)):
                Pns[ECvali,ni] = Pn_recursive(ns[ni], my_Vb, ECvals[ECvali], my_mutilde, my_temp);

        # plot Pn vs n
        fig, ax = plt.subplots();
        for ECvali in range(len(ECvals)):
            ax.plot(ns, Pns[ECvali], label="$E_C = ${:.3f}".format(ECvals[ECvali]));
        ax.set_title( "$T = $ {:.1f} K, $eV_g+\mu_0 = $ {:.3f} eV, $V_b = $ {:.3f} eV".format(my_temp/kelvin2eV,my_mutilde, my_Vb)); 
        ax.set_xlabel("$n$");
        ax.set_ylabel("$P_n/P_0$");
        plt.legend();
        plt.show();

    if False: # Pn vs mu

        # physical params, in eV
        nmax = 10;
        my_EC = 0.010;  
        my_temp = 50.0*kelvin2eV;
        my_Vb = 0.0;
        
        muvals = np.array([-20,-5,5,20])/1000; # when mutilde+2*EC < 0 there are instabilities
        ns = np.array(range(nmax));
        Pns = np.empty((len(muvals), len(ns)), dtype=float);
        for ni in range(len(ns)):
            print("\tn = ",ns[ni]);
            for muvali in range(len(muvals)):
                Pns[muvali,ni] = Pn_recursive(ns[ni], my_Vb, my_EC, muvals[muvali], my_temp);

        # plot Pn vs n
        fig, ax = plt.subplots();
        for muvali in range(len(muvals)):
            ax.plot(ns, Pns[muvali], label="$\mu = ${:.3f}".format(muvals[muvali]));
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $V_b = $ {:.3f} eV".format(my_temp/kelvin2eV,my_EC, my_Vb)); 
        ax.set_xlabel("$n$");
        ax.set_ylabel("$P_n/P_0$");
        plt.legend();
        plt.show();

    
    
