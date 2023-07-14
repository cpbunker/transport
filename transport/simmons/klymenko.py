'''
Coulomb blockade effects in a K-fold degenerate molecular orbital
See https://doi.org/10.1063/1.1491179
'''

import numpy as np
import matplotlib.pyplot as plt

def f_func(epsilon, kBT):
    '''
    '''
    
    return 1/(np.exp(epsilon/kBT)+1);

def W_n_pm(pm, n, EC, Vg, mualpha, kBT):
    '''
    '''

    if(pm==0):  # n-1 -> n transition
        return f_func(EC*(2*n-1)-(mualpha+Vg), kBT);
    elif(pm==1):# n -> n-1 transition
        return f_func(-EC*(2*n-1)+(mualpha+Vg), kBT);

def Gamma_n_pm(pm, n, EC, Vg, mualpha, kBT):
    '''
    '''
    raise NotImplementedError;
    
    if(pm==0):  # n -> n+1 transition
        return f_func(En(n+1,EC,Vg)-En(n,EC,Vg)-mualpha,kBT);
    elif(pm==1):# n -> n-1 transition
        return f_func(En(n-1,EC,Vg)-En(n,EC,Vg)+mualpha,kBT);

def Pn_recursive(n, Vb, EC, Vg, mu0, eta, kBT):
    '''
    '''
    
    if(n==0):
        return np.ones_like(Vb);
    else:
        # sum each over alpha
        numerator, denominator = 0.0, 0.0;
        mualphas = [mu0+eta*Vb, mu0-(1-eta)*Vb];
        for mualpha in mualphas:
            numerator   += W_n_pm(0, n, EC, Vg, mualpha, kBT);  
            denominator += W_n_pm(1, n, EC, Vg, mualpha, kBT);
        retval = (numerator/denominator)*Pn_recursive(n-1, Vb, EC, Vg, mu0, eta, kBT);
        return retval;

def I_of_Vb(Vb, EC, Vg, mu0, eta, kBT, nmax, return_Pn = False):
    '''
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(EC, float)): raise TypeError;

    # get Pns
    Pns = np.empty((nmax+1,len(Vb)),dtype=float);
    for n in range(nmax+1):
        Pns[n,:] = Pn_recursive(n, Vb, EC, Vg, mu0, eta, kBT);
        
    # normalize for each Vb
    for Vbi in range(len(Vb)):
        if True:
            norm = 0.0;
            for n in range(nmax+1):
                choose_coef = np.math.factorial(nmax)/(np.math.factorial(n)*np.math.factorial(nmax-n)); 
                norm += Pns[n,Vbi]*choose_coef; 
            Pns[:,Vbi] = Pns[:,Vbi]/norm;

    if(return_Pn):
        assert(len(Vb)==1);
        return Pns[:,0];

    # chemical potential
    muL, muR = mu0+eta*Vb, mu0-(1-eta)*Vb

    # get current
    current = np.zeros_like(Vb);
    for n in range(nmax+1):
        # left part
        choose_coef = np.math.factorial(nmax)/(np.math.factorial(n)*np.math.factorial(nmax-n));
        current += W_n_pm(0, n+1, EC, Vg, muL, kBT)*(nmax-n)*choose_coef*Pns[n];
        current +=-W_n_pm(1, n  , EC, Vg, muL, kBT)*n*choose_coef*Pns[n];        
    return current; 

if(__name__ == "__main__"):

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    kelvin2eV =  8.617e-5;
    conductance_quantum = 7.748e-5/2; # units amp/volt

    if False: # plot at various n max
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.01;
        my_Vg = -1.00;
        my_mu0 = 1.10; # scaling of separation with mu0+Vg makes fits ambiguous
        my_Vg, my_mu0 = 0.0, 0.0;
        my_eta = 1.0; # Neo only changes upper electrode voltage
        conductance=True;

        nvals = np.array([16,8,4]);
        for nval in nvals:
            Is = I_of_Vb(Vbs, my_EC, my_Vg, my_mu0, my_eta, my_temp, nval);
            if(conductance): Is = np.gradient(Is);
            ax.plot(Vbs, Is, label = str(nval));
        for integer in [0,1,2]:
            ax.axvline((2*integer+1)*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $\mu_0-E_0= $ {:.3f} eV, $\eta = $ {:.1f}".format(my_temp/kelvin2eV, my_EC, my_mu0+my_Vg, my_eta));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (au)");
        else: ax.set_ylabel("$I(V_b)$ (au)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot at various T
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        my_EC = 0.01;
        my_Vg = -1.0;
        my_mu0 = 1.10;
        #my_Vg, my_mu0 = 0.0, 0.0;
        my_eta = 1.0; # Neo only changes upper electrode voltage
        nmax = 16;
        conductance=True;

        Tvals = np.array([5,10,20])*kelvin2eV;
        for Tval in Tvals:
            Is = I_of_Vb(Vbs, my_EC, my_Vg, my_mu0, my_eta, Tval, nmax);
            if(conductance): Is = np.gradient(Is);
            ax.plot(Vbs, Is, label = "{:.1f}".format(Tval/kelvin2eV));
        for integer in [0,1,2]:
            ax.axvline((2*integer+1)*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$E_C = $ {:.3f} eV, $\mu_0-E_0 = $ {:.3f} eV, $\eta = $ {:.1f}".format(my_EC, my_mu0+my_Vg, my_eta));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (au)");
        else: ax.set_ylabel("$I(V_b)$ (au)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

    if True: # plot at various Vg -> linear (1 for eta=1)
                # leftward shift for positive Vg, mu0
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e5));

        # physical params, in eV
        my_temp = 5.0*kelvin2eV;
        my_EC = 0.01;
        my_mu0 = 0.02;
        my_eta = 1.0; # Neo only changes upper electrode voltage
        nmax = 16;
        conductance=True;

        Vgvals = np.array([0.001, 0.005,0.05]);
        for Vgval in Vgvals:
            Is = I_of_Vb(Vbs, my_EC, Vgval, my_mu0, my_eta, my_temp, nmax);
            if(conductance): Is = np.gradient(Is);
            ax.plot(Vbs, Is, label = "{:.3f}".format(Vgval));
        for integer in [0,1,2,3,4]:
            ax.axvline((2*integer+1)*my_EC,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $E_C = $ {:.3f} eV, $\mu_0= $ {:.3f} eV, $\eta = $ {:.1f}".format(my_temp, my_EC, my_mu0, my_eta));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (au)");
        else: ax.set_ylabel("$I(V_b)$ (au)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();




    
    
