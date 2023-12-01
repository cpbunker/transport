'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios

Main problems:
Is it valid to use the charging energy form for single-particle En ?
Oscillations are not big enough (order of 10s of nA rather than 100s of nA)
Peak height falls off to slowly with T (5 K -> 50 K only a 1/3 dropoff)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as scipy_integ

import time

# units
kelvin2eV =  8.617e-5; # units eV/K
e2overh= 7.748e-5 *1e9/2; # units nA/volt


def En(n, EC, Vb):
    '''
    This gives the energy of a single-fermion level rather than a
    many-body charge state. 
    '''

    return EC*(2*n+1)+Vb/2 # chem potential of island is zero of energy scale

def nFD(epsilon, kBT):
    '''
    '''
    
    return 1/(np.exp(epsilon/kBT)+1);

def I_of_Vb_zero(Vb, mu0, Gamma, EC, kBT, ns):
    '''
    Compute the zero temperature current according to
    the analytical integration result
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(mu0, float)): raise TypeError;
    if(not isinstance(ns, np.ndarray)): raise TypeError;
    if(ns.dtype != int): raise TypeError;
    assert (kBT == 0.0); # just to have same call signature

    # chem potentials
    muL, muR = mu0+Vb, mu0;

    # current
    current = np.zeros_like(Vb);
    for n in ns:
        Enval = En(n,EC,Vb);
        current += np.arctan((muL-Enval)/Gamma);
        current +=-np.arctan((muR-Enval)/Gamma);
    return e2overh*Gamma*current;

def dI_of_Vb_zero(Vb, mu0, Gamma, EC, kBT, ns):
    '''
    Compute the zero temperature conductance according to
    the analytical integration result
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(mu0, float)): raise TypeError;
    if(not isinstance(ns, np.ndarray)): raise TypeError;
    if(ns.dtype != int): raise TypeError;
    assert (kBT == 0.0); # just to have same call signature

    # chem potentials
    muL, muR = mu0+Vb, mu0;

    # conductance
    conductance = np.zeros_like(Vb);
    for n in ns:
        Enval = En(n,EC,Vb);
        conductance += 1/(1+(muL-Enval)*(muL-Enval)/(Gamma*Gamma) );
        conductance += 1/(1+(muR-Enval)*(muR-Enval)/(Gamma*Gamma) );
    return e2overh*(1/2)*conductance;

def I_of_Vb(Vb, mu0, Gamma, EC, kBT, ns, Mn=10, xvals=1e4):
    '''
    Compute the finite temperature current by numerical integration
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(mu0, float)): raise TypeError;
    if(not isinstance(ns, np.ndarray)): raise TypeError;
    if(ns.dtype != int): raise TypeError;

    # variable of integration
    mulimits = np.array([mu0,mu0-np.min(Vb),mu0+np.min(Vb),mu0-np.max(Vb),mu0+np.max(Vb)]);
    Elimits = (-Mn*kBT+np.min(mulimits),+Mn*kBT+np.max(mulimits));
    Evals = np.linspace(*Elimits,int(xvals));
    print("Integration limits = ",Elimits);

    # integrate to get current contribution
    integration_result = np.zeros_like(Vb);
    for Vbi, Vbval in enumerate(Vb):
        for n in ns:
            muL, muR = mu0+Vbval, mu0;
            Enval = En(n,EC,Vbval);
            tau_vals = 1/(1+ np.power( (Evals-Enval)/Gamma,2));
            integration_result[Vbi] += np.trapz(tau_vals*(nFD(Evals-muL,kBT)-nFD(Evals-muR,kBT)), Evals);

    return e2overh*integration_result;

def dI_of_Vb(Vb, mu0, Gamma, EC, kBT, ns, Mn = 10, xvals=1e4):
    '''
    Compute the finite temperature conductance by numerical integration
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(mu0, float)): raise TypeError;
    if(not isinstance(ns, np.ndarray)): raise TypeError;
    if(ns.dtype != int): raise TypeError;

    # variable of integration
    mulimits = np.array([mu0,mu0-np.min(Vb),mu0+np.min(Vb),mu0-np.max(Vb),mu0+np.max(Vb)]);
    Elimits = (-Mn*kBT+np.min(mulimits),+Mn*kBT+np.max(mulimits));
    Emesh = np.linspace(*Elimits, int(xvals));
    Beta = 1/kBT;

    # store exp(E-) as array outside loop
    # evaluate all constants outside the loop

    # function to integrate
    def integrand(Eint, nints, Vbint):
        retval = np.zeros_like(Eint);
        exp_muR = np.exp((Eint-mu0)*Beta);
        exp_muL = exp_muR*np.exp(-Vbint*Beta);
        #oneover_expmuL_p1 = 1/( np.exp((Eint-mu0)*Beta)*np.exp(-Vbint*Beta) +1);
        #oneover_expmuLinv_p1 = 1/( (1/
        for n in nints:
            ratio_n = (Eint-(2*n+1)*EC-Vbint/2)/Gamma;
            lorentzian_n = 1/(1+ (ratio_n)*(ratio_n));
            ret_n = Beta/((exp_muL+1)*(1/exp_muL+1)); # <- fixed overflow
            ret_n += ratio_n/Gamma *lorentzian_n*(1/(exp_muL+1) - 1/(exp_muR+1));
            retval += ret_n*lorentzian_n;
        return retval;

    # plot integrand
    if False:
        print("Integration limits = ",Elimits);
        fig, ax = plt.subplots();
        print(np.min(Vb), np.max(Vb));
        print(np.min(mulimits), np.max(mulimits));
        for Vval in [Vb[0], 8*EC, 10*EC, Vb[-1]]:
            yvals = integrand(Emesh, ns, Vval);
            ax.plot(Emesh, yvals, label = "Vb={:.3f} eV".format(Vval));
        plt.legend()
        plt.show();
        assert False;
    
    # integrate to get current contribution
    integration_result = np.empty_like(Vb);
    for Vbi, Vbval in enumerate(Vb):
        Vb_integrand = integrand(Emesh, ns, Vbval);
        integration_result[Vbi] = scipy_integ(Vb_integrand, Emesh);

    return e2overh*integration_result;


if(__name__ == "__main__"):

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    conductance = True;

    if False: # plot with various methods

        # physical params, in eV
        my_mu0 = 0.00; # *relative* to the island chem potential
        my_EC = 0.005;
        my_temp = 30.0*kelvin2eV;
        nmax = 4;
        narr = np.arange(-nmax,nmax+1);

        for my_Gamma in [2*my_EC/np.sqrt(48), 2*my_EC/np.sqrt(48)/100]:

            # plotting
            fig, axes = plt.subplots(2, sharex=True);
            Vbs = np.linspace(-Vb_max,Vb_max,int(200));

        
            # zero temp analytic
            Is = dI_of_Vb_zero(Vbs, my_mu0, my_Gamma, my_EC, 0.0, narr);
            axes[0].plot(Vbs, Is, label = "dI_of_Vb_zero");

            # finite temp analytical
            start = time.time();
            Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_temp, narr);
            stop = time.time();
            print("dI_of_Vb time = ",stop-start);
            axes[0].plot(Vbs, Is, label = "dI_of_Vb");

            if False:
                # zero temp gradient
                Is = np.gradient(I_of_Vb_zero(Vbs, my_mu0, my_Gamma, my_EC, 0.0, narr));
                axes[1].plot(Vbs, Is, label = "gradient of I_of_Vb_zero");

                # finite temp gradient
                start = time.time();
                Is = np.gradient(I_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_temp, narr));
                stop = time.time();
                print("I_of_Vb time = ",stop-start);
                axes[1].plot(Vbs, Is, label = "gradient of I_of_Vb");

            # peaks
            for integer in [0]:
                print("2EC(2n+1) - 2mu0 = {:.3f}".format(2*En(integer, my_EC, 0.0)-2*my_mu0));
                for ax in axes:
                    ax.axvline(2*En(integer, my_EC, 0.0)-2*my_mu0,color="black",linestyle="dashed");

            # format
            axes[0].set_title( "$T = $ {:.3f} eV, $\mu_0 = $ {:.3f} eV, $\Gamma = $ {:.3f} eV, $E_C = $ {:.3f} eV".format(my_temp, my_mu0, my_Gamma, my_EC));
            axes[-1].set_xlabel("$V_b$ (V)");
            for ax in axes:
                ax.set_ylabel("$dI/dV_b$ (nA/V)");
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
                ax.legend();
            plt.tight_layout();
            plt.show();

    if False: # plot at various Gamma
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e4));

        # physical params, in eV
        my_mu0 = 0.0; # *relative* to the island chem potential
        my_EC = 0.010;
        my_temp = 0.0*kelvin2eV;
        nmax = 10;
        narr = np.arange(-nmax,nmax+1);

        # iter over Gamma
        Gammavals = np.array([1e-4,my_EC/np.sqrt(3)]); 
        to_save = np.empty((len(Gammavals),len(Vbs)),dtype=float); to_savei=0;
        for Gammaval in Gammavals:
            if(conductance): Is = dI_of_Vb_zero(Vbs, my_mu0, Gammaval, my_EC, my_temp, narr);
            else: Is = I_of_Vb_zero(Vbs, my_mu0, Gammaval, my_EC, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, Is, label = "$\Gamma = $ {:.5f}".format(Gammaval));
        for integer in [0]:
            print("2EC(2n+1) - 2mu0 = {:.3f}".format(2*En(integer, my_EC, 0.0)-2*my_mu0));
            ax.axvline(2*En(integer, my_EC, 0.0)-2*my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title("$T = $ {:.1f} K, $\mu_0 = $ {:.3f} eV, $E_C = $ {:.3f} eV".format(my_temp/kelvin2eV, my_mu0, my_EC));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

    if True: # plot at various EC
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e4));

        # physical params, in eV
        my_mu0 = 0.0; # *relative* to the island chem potential
        my_Gamma = 0.0002; 
        my_temp = 0.0*kelvin2eV;
        nmax = 10;
        narr = np.arange(-nmax,nmax+1);
        narr = np.array([-1,0,1,2]);

        # iter over EC
        ECvals = np.array([0.001,0.004]);
        to_save = np.empty((len(ECvals),len(Vbs)),dtype=float); to_savei=0;
        for ECval in ECvals:
            if(conductance): Is = dI_of_Vb_zero(Vbs, my_mu0, my_Gamma, ECval, my_temp, narr);
            else: Is = I_of_Vb_zero(Vbs, my_mu0, my_Gamma, ECval, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, Is, label = "$E_C = $ {:.4f} eV".format(ECval));
            for integer in [0]:
                print("2EC(2n+1) - 2mu0 = {:.3f}".format(2*En(integer, ECval, 0.0)-2*my_mu0));
                ax.axvline(2*En(integer, ECval, 0.0)-2*my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $\mu_0 = $ {:.4f} eV, $\Gamma = $ {:.4f} eV".format(my_temp/kelvin2eV, my_mu0, my_Gamma));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot at various temperatures
        fig, ax = plt.subplots();
        Vbs = np.linspace(0,Vb_max,int(1e4));

        # physical params, in eV
        my_mu0 = 0.0; # *relative* to the island chem potential
        my_Gamma = 0.0001;
        my_EC = 0.010;
        nmax = 10;
        narr = np.arange(-nmax,nmax+1);

        # iter over T
        Tvals = np.array([2.5,5,10,20])*kelvin2eV;
        to_save = np.empty((len(Tvals),len(Vbs)),dtype=float); to_savei=0;
        for Tval in Tvals:
            start = time.time();
            if(conductance): Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, Tval, narr);
            else: Is = I_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, Tval, narr);
            stop = time.time();
            print("Total integration time = ",stop-start);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, Is, label = "$k_B T = $ {:.4f} eV".format(Tval));
        for integer in [0]:
            print("2EC(2n+1) - 2mu0 = {:.4f}".format(2*En(integer, my_EC, 0.0)-2*my_mu0));
            ax.axvline(2*En(integer, my_EC, 0.0)-2*my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$\mu_0 = $ {:.4f} eV, $\Gamma = $ {:.4f} eV, $E_C = $ {:.4f} eV".format(my_mu0, my_Gamma, my_EC));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_xlim(np.min(Vbs), np.max(Vbs));
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

    if False: # plot at various mu0
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e3));

        # physical params, in eV
        my_EC = 0.01; 
        my_Gamma = 1e-5; 
        my_temp = 0.0*kelvin2eV;
        nmax = 5;
        narr = np.arange(-nmax,nmax+1);

        # iter over EC
        muvals = np.array([0.001,0.05]);
        to_save = np.empty((len(muvals),len(Vbs)),dtype=float); to_savei=0;
        for muval in muvals:
            if(conductance): Is = dI_of_Vb_zero(Vbs, muval, my_Gamma, my_EC, my_temp, narr);
            else: Is = I_of_Vb_zero(Vbs, muval, my_Gamma, my_EC, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, Is, label = "$\mu_0 = $ {:.3f}".format(muval));
            for integer in [0]:
                print("2EC(2n+1) - 2mu0 = {:.3f}".format(2*En(integer, my_EC, 0.0)-2*muval));
                ax.axvline(2*En(integer, my_EC, 0.0)-2*muval,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $\Gamma = $ {:.3f} eV, $E_C = $ {:.3f} eV".format(my_temp/kelvin2eV, my_Gamma, my_EC));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();
    
