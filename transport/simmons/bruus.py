'''
Coulomb blockade effects in an n-electron molecular island
See Bruus & Flensberg, Sec. 10.2

Questions about blockade
How can we incorporate n < 0 ?
How do we symmetrize L and R current ?
We know under muL=mu0+Vb/2, muR=u0-Vb/2, the peak separation is 4*EC.
Using muL=mu0, muR=mu0-Vb, does this change?
Yes, it changes to 2*EC but peaks only appear at Vb < EC
The fact that the results depend strongly on this choice is a bad sign
'''

import numpy as np
import matplotlib.pyplot as plt

def En(n, EC, Vg):
    '''
    '''
    
    return EC*n*n-Vg*n;

def f_func(epsilon, kBT):
    '''
    '''
    #return 1/(np.exp(epsilon/kBT)+1);
    return epsilon/(np.exp(epsilon/kBT)-1);

def Gamma_n_pm(pm, n, EC, Vg, mualpha, kBT):
    '''
    '''
    
    if(pm==0): # n+1
        return f_func(En(n+1,EC,Vg)-En(n,EC,Vg)-mualpha,kBT);
    elif(pm==1): #n-1
        return f_func(En(n-1,EC,Vg)-En(n,EC,Vg)+mualpha,kBT);

def Pn_recursive(n, Vb, EC, Vg, mu0, kBT):
    '''
    '''
    
    if(n==0):
        return np.ones_like(Vb);
    else:
        # sum each over alpha
        numerator, denominator = 0.0, 0.0;
        mualphas = [mu0+Vb/2, mu0-Vb/2];
        mualphas = [mu0+Vb, mu0];
        for mualpha in mualphas:
            numerator   += Gamma_n_pm(0, n-1,EC, Vg, mualpha, kBT);  
            denominator += Gamma_n_pm(1, n,  EC, Vg, mualpha, kBT);
        retval = (numerator/denominator)*Pn_recursive(n-1, Vb, EC, Vg, mu0, kBT);
        return retval;

def I_of_Vb(Vb, EC, Vg, mu0, kBT, nmax, return_Pn = False):
    '''
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(EC, float)): raise TypeError;

    # get Pns
    Pns = np.empty((nmax+1,len(Vb)),dtype=float);
    for n in range(nmax+1):
        Pns[n,:] = Pn_recursive(n, Vb, EC, Vg, mu0, kBT);
    # normalize for each Vb
    for Vbi in range(len(Vb)): 
        Pns[:,Vbi] = Pns[:,Vbi]/np.sum(Pns[:,Vbi]);
        pass

    if(return_Pn):
        assert(len(Vb)==1);
        return Pns[:,0];

    # chemical potential
    muL, muR = mu0+Vb/2, mu0-Vb/2;
    muL, muR = mu0+Vb, mu0;

    # get current
    current = np.zeros_like(Vb);
    for n in range(nmax+1):
        # left part
        current += Gamma_n_pm(0, n, EC, Vg, muL, kBT)*Pns[n]/2; #n+1
        current +=-Gamma_n_pm(1, n, EC, Vg, muL, kBT)*Pns[n]/2; #n-1
        # right part
        current += Gamma_n_pm(1, n, EC, Vg, muR, kBT)*Pns[n]/2; #n-1
        current +=-Gamma_n_pm(0, n, EC, Vg, muR, kBT)*Pns[n]/2; #n+1
    return current; 

if(__name__ == "__main__"):

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    kelvin2eV =  8.617e-5;

    if False: # plot differential conductance at various EC
        fig, ax = plt.subplots();
        ECvals = np.array([5,10,20])/1000;
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        nmax = 10;
        # when mutilde+2*EC < 0 there are instabilities
        my_mutilde = 0.0;
        my_temp = 5.0*kelvin2eV; # peak separation should be strictly T independent

        for ECvali in range(len(ECvals)):
            Is = I_of_Vb(Vbs, ECvals[ECvali], my_mutilde, my_temp, nmax);
            ax.plot(Vbs,np.gradient(Is),label = "{:.3f} eV".format(ECvals[ECvali]));
            ax.axvline(-2*ECvals[ECvali],color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $eV_g +\mu_0 = $ {:.3f} eV".format(my_temp/kelvin2eV,my_mutilde));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot differential conductance at various temp
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
            Is = I_of_Vb(Vbs, my_EC, my_mutilde/2, my_mutilde/2, Tval, nmax);
            ax.plot(Vbs, np.gradient(Is), label = "{:.1f} K".format(Tval/kelvin2eV));

        # format
        ax.set_title( "$E_C = $ {:.3f} eV, $\mu = $ {:.3f} eV".format(my_EC, my_mutilde));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if True: # plot differential conductance at various Vg
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        nmax = 10;
        # when mutilde+2*EC < 0 there are instabilities
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.01; # should be of order of observed conductance osc period (10s of meV)
        my_mu0 = 0.00;
        Vgvals = np.array([0,0.5,0.8])*my_EC;
        
        for Vgval in Vgvals:
            Is = I_of_Vb(Vbs, my_EC, Vgval, my_mu0, my_temp, nmax);
            ax.plot(Vbs,np.gradient(Is), label = "{:.3f} eV".format(Vgval));
        ax.axvline(-2*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$E_C = $ {:.3f} eV, $T = $ {:.1f} K, $\mu_0 = $ {:.3f}".format(my_EC,my_temp/kelvin2eV, my_mu0));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot differential conductance at various n max
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.005; # should be of order of observed conductance osc period (10s of meV)
        my_mutilde = 0.00;

        nvals = np.array([0,1,2,10]);
        for nval in nvals:
            Is = I_of_Vb(Vbs, my_EC, my_mutilde/2, my_mutilde/2, my_temp, nval);
            ax.plot(Vbs, np.gradient(Is), label = str(nval));
        for integer in [-3,-1,1,3]:
            ax.axvline(integer*2*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $eV_g+\mu_0 = $ {:.3f} eV".format(my_temp/kelvin2eV, my_EC, my_mutilde));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_ylabel("$dI/V_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot differential conductance vs Vg
        fig, ax = plt.subplots();
        my_Vb, Vbstep = 0.00, 1e-6;
        Vbs = my_Vb + Vbstep*np.array([-2,-1,0,1,2]);

        # physical params, in eV
        nmax = 10;
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.010; # should be of order of observed conductance osc period (10s of meV)
        my_mu0 = 0.015; # should be >> kBT and EC
        Vgvals = np.linspace(-7*my_EC, 7*my_EC, int(1e3));
        dIvals = np.empty_like(Vgvals);

        for Vgvali in range(len(Vgvals)):
            I_of_Vg = I_of_Vb(Vbs, my_EC, Vgvals[Vgvali], my_mu0, my_temp, nmax);
            dIvals[Vgvali] = np.gradient(I_of_Vg)[len(I_of_Vg)//2];
            
        ax.plot(Vgvals, dIvals, label = str(nmax));
        for integer in [-3,-1,1,3]:
            ax.axvline(integer*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $\mu_0 = $ {:.3f} eV, $eV_b = $ {:.3f} eV".format(my_temp/kelvin2eV, my_EC, my_mu0, my_Vb));
        ax.set_xlabel("$V_g$ (V)");
        ax.set_ylabel("$dI/dV_b$ (au)");
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # Pn vs n at various T

        # physical params, in eV
        nmax = 10;
        my_mutilde = 0.0; # when mutilde+2*EC < 0 there are instabilities
        my_EC = 0.001; # should be of order of observed conductance osc period (10s of meV)
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
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.010; # should be of order of observed conductance osc period (10s of meV)
        my_mutilde = 0.0; # when mutilde+2*EC < 0 there are instabilities
        
        Vbvals = np.array([-0.999*Vb_max,0.0,0.5*Vb_max,0.999*Vb_max,2*Vb_max]);
        ns = np.array(range(nmax+1));
        Pns = np.empty((len(Vbvals), len(ns)), dtype=float);
        for Vbvali in range(len(Vbvals)):
            Pns[Vbvali] = I_of_Vb(Vbvals[Vbvali:Vbvali+1], my_EC, my_mutilde/2, my_mutilde/2, my_temp, nmax, return_Pn=True);

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

    if False: # Pn vs n at various EC

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

    if False: # Pn vs n at various Vg

        # physical params, in eV
        nmax = 10;
        my_EC = 0.010;  
        my_temp = 50.0*kelvin2eV;
        my_Vb = 0.0;
        my_mu0 = 0.00;

        # iter over Vg
        Vgvals = np.array([-20,-5,0,5,20])/1000;
        ns = np.array(range(nmax+1));
        Pns = np.empty((len(Vgvals), len(ns)), dtype=float);
        for Vgvali in range(len(Vgvals)):
            Pns[Vgvali] = I_of_Vb(np.array([my_Vb]), my_EC, Vgvals[Vgvali], my_mu0, my_temp, nmax, return_Pn=True);

        # plot Pn vs n
        fig, ax = plt.subplots();
        for Vgvali in range(len(Vgvals)):
            ax.plot(ns, Pns[Vgvali], label="$V_g = ${:.3f}".format(Vgvals[Vgvali]));
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $V_b = $ {:.3f} eV".format(my_temp/kelvin2eV,my_EC, my_Vb)); 
        ax.set_xlabel("$n$");
        ax.set_ylabel("$P_n/P_0$");
        plt.legend();
        plt.show();

    
    
