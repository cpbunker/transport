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
from scipy.integrate import quad

import time

def En(n, EC, Vb):
    '''
    This gives the energy of a single-fermion level rather than a
    many-body charge state. 
    '''
    if False:
        assert(EC == 0.0);
        return EC*n*n-Vg*n;
    else:
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
        current += np.arctan((muL-Enval)/(2*Gamma));
        current +=-np.arctan((muR-Enval)/(2*Gamma));
    return Gamma*Gamma/(Gamma+Gamma) *current;

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
        conductance += 1/(1+np.power((muL-Enval)/(2*Gamma),2));
        conductance += 1/(1+np.power((muR-Enval)/(2*Gamma),2));
    return Gamma*Gamma/(2*np.power(2*Gamma,2))*conductance;

def I_of_Vb(Vb, mu0, Gamma, EC, kBT, ns, xvals=1e5):
    '''
    Compute the finite temperature current by numerical integration
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(mu0, float)): raise TypeError;
    if(not isinstance(ns, np.ndarray)): raise TypeError;
    if(ns.dtype != int): raise TypeError;

    # variable of integration
    mulimits = np.array([mu0,mu0-np.min(Vb),mu0+np.min(Vb),mu0-np.max(Vb),mu0+np.max(Vb)]);
    Elimits = (-5*kBT+np.min(mulimits),+5*kBT+np.max(mulimits));
    Evals = np.linspace(*Elimits,int(xvals));
    print("Integration limits = ",Elimits);

    # integrate to get current contribution
    integration_result = np.zeros_like(Vb);
    for Vbi, Vbval in enumerate(Vb):
        for n in ns:
            muL, muR = mu0+Vbval, mu0;
            Enval = En(n,EC,Vbval);
            tau_vals = 1/(1+ np.power( (Evals-Enval)/(2*Gamma),2));
            integration_result[Vbi] += np.trapz(tau_vals*(nFD(Evals-muL,kBT)-nFD(Evals-muR,kBT)), Evals);

    return Gamma*Gamma/(4*Gamma*Gamma)*integration_result

def dI_of_Vb(Vb, mu0, Gamma, EC, kBT, ns, Mn = 10):
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

    # function to integrate
    def integrand(Eint, Cint, Vbint):
        retval = np.zeros_like(Eint);
        prefactor = 1/(1+np.power( (Eint-Cint-Vbint/2)/(2*Gamma),2));
        retval += (1/kBT)*np.exp((Eint-mu0-Vbint)/kBT)*np.power(np.exp((Eint-mu0-Vbint)/kBT)+1,-2);
        retval += (Eint-Cint-Vbint/2)/(4*Gamma*Gamma) *prefactor*(1/(np.exp((Eint-mu0-Vbint)/kBT)+1) - 1/(np.exp((Eint-mu0)/kBT)+1));
        return prefactor*retval;

    # plot integrand
    if False:
        print("Integration limits = ",Elimits);
        fig, ax = plt.subplots();
        for Vval in [Vb[0], Vb[len(Vb)//3], Vb[2*len(Vb)//3], Vb[-1]]:
            xvals = np.linspace(*Elimits, 1000);
            yvals = integrand(xvals, (2*ns[-1]+11)*EC, Vval);
            ax.plot(xvals, yvals, label = Vval);
        plt.show();
        assert False;
    
    # conductance
    conductance = np.zeros_like(Vb);
    for n in ns:
        # integrate to get current contribution
        Cnval = (2*n+1)*EC;
        integration_result = np.empty_like(Vb);
        for Vbi, Vbval in enumerate(Vb):
            integration_result[Vbi] = quad(integrand, *Elimits, args = (Cnval, Vbval))[0];
        conductance += integration_result;

    return Gamma*Gamma/np.power(2*Gamma,2)*conductance;


if(__name__ == "__main__"):

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    kelvin2eV =  8.617e-5; # units eV/K
    conductance_quantum = 7.748e-5; # units amp/volt
    conductance = True;

    if True: # plot with various methods

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
            axes[0].plot(Vbs, conductance_quantum*1e9*Is, label = "dI_of_Vb_zero");

            # finite temp analytical
            start = time.time();
            Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_temp, narr);
            stop = time.time();
            print("dI_of_Vb time = ",stop-start);
            axes[0].plot(Vbs, conductance_quantum*1e9*Is, label = "dI_of_Vb");

            if False:
                # zero temp gradient
                Is = np.gradient(I_of_Vb_zero(Vbs, my_mu0, my_Gamma, my_EC, 0.0, narr));
                axes[1].plot(Vbs, conductance_quantum*1e9*Is, label = "gradient of I_of_Vb_zero");

                # finite temp gradient
                start = time.time();
                Is = np.gradient(I_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_temp, narr));
                stop = time.time();
                print("I_of_Vb time = ",stop-start);
                axes[1].plot(Vbs, conductance_quantum*1e9*Is, label = "gradient of I_of_Vb");

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
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e6));

        # physical params, in eV
        my_mu0 = 0.0; # *relative* to the island chem potential
        my_EC = 0.005;
        my_temp = 0.0*kelvin2eV;
        nmax = 5;
        narr = np.arange(-nmax,nmax+1);

        # iter over Gamma
        Gammavals = np.array([1e-4,2*my_EC/np.sqrt(48),0.003]);
        to_save = np.empty((len(Gammavals),len(Vbs)),dtype=float); to_savei=0;
        for Gammaval in Gammavals:
            if(conductance): Is = dI_of_Vb_zero(Vbs, my_mu0, Gammaval, my_EC, my_temp, narr);
            else: Is = I_of_Vb_zero(Vbs, my_mu0, Gammaval, my_EC, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = "$\Gamma_0 = $ {:.5f}".format(Gammaval));
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

        # save
        fname = "land_data/vsGamma"
        save_params = np.array([my_mu0, np.nan, my_EC, my_temp, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);

    if False: # plot at various temperatures
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e3));

        # physical params, in eV
        my_mu0 = 0.0; # *relative* to the island chem potential
        my_Gamma = 0.0001;
        my_EC = 0.005;
        nmax = 10;
        narr = np.arange(-nmax,nmax+1);

        # iter over T
        Tvals = np.array([5,10,20])*kelvin2eV;
        to_save = np.empty((len(Tvals),len(Vbs)),dtype=float); to_savei=0;
        for Tval in Tvals:
            start = time.time();
            if(conductance): Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, Tval, narr);
            else: Is = I_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, Tval, narr);
            stop = time.time();
            print("Total integration time = ",stop-start);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = "$k_B T = $ {:.5f} eV".format(Tval));
        for integer in [0]:
            print("2EC(2n+1) - 2mu0 = {:.3f}".format(2*En(integer, my_EC, 0.0)-2*my_mu0));
            ax.axvline(2*En(integer, my_EC, 0.0)-2*my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$\mu_0 = $ {:.3f} eV, $\Gamma = $ {:.4f} eV, $E_C = $ {:.3f} eV".format(my_mu0, my_Gamma, my_EC));
        ax.set_xlabel("$V_b$ (V)");
        ax.set_xlim(np.min(Vbs), np.max(Vbs));
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

        # save
        fname = "land_data/vsT"
        save_params = np.array([my_mu0, my_Gamma, my_EC, np.nan, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);

    if False: # plot at various EC
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e3));

        # physical params, in eV
        my_mu0 = 0.0; # *relative* to the island chem potential
        my_Gamma = 0.001; 
        my_temp = 10.0*kelvin2eV;
        nmax = 5;
        narr = np.arange(-nmax,nmax+1);

        # iter over EC
        ECvals = np.array([0.01,0.04]);
        to_save = np.empty((len(ECvals),len(Vbs)),dtype=float); to_savei=0;
        for ECval in ECvals:
            if(conductance): Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, ECval, my_temp, narr);
            else: Is = I_of_Vb(Vbs, my_mu0, my_Gamma, ECval, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = "$E_C = $ {:.3f}".format(ECval));
            for integer in [0]:
                print("2EC(2n+1) - 2mu0 = {:.3f}".format(2*En(integer, ECval, 0.0)-2*my_mu0));
                ax.axvline(2*En(integer, ECval, 0.0)-2*my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $\mu_0 = $ {:.3f} eV, $\Gamma = $ {:.3f} eV".format(my_temp/kelvin2eV, my_mu0, my_Gamma));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

        # save
        fname = "land_data/vsEC"
        save_params = np.array([my_mu0, my_Gamma, np.nan, my_temp, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);

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
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = "$\mu_0 = $ {:.3f}".format(muval));
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

        # save
        fname = "land_data/vsEC"
        save_params = np.array([np.nan, my_Gamma, my_EC, my_temp, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);
    
