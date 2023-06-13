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

def En(n, EC, Vg):
    '''
    This gives the energy of a single-fermion level rather than a
    many-body charge state. 
    '''
    if False:
        assert(EC == 0.0);
        return EC*n*n-Vg*n;
    else:
        return EC*(2*n+1)-Vg;

def nFD(epsilon, kBT):
    '''
    '''
    
    return 1/(np.exp(epsilon/kBT)+1);

def I_of_Vb_zero(Vb, mu0, Gamma, EC, Vg, kBT, ns):
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
        Enval = En(n,EC,Vg);
        current += np.arctan((muL-Enval)/(2*Gamma));
        current +=-np.arctan((muR-Enval)/(2*Gamma));
    return Gamma*Gamma/(Gamma+Gamma) *current;

def dI_of_Vb_zero(Vb, mu0, Gamma, EC, Vg, kBT, ns):
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

    def deriv_arctan(mualpha, E_n, Gam):
        return 1/(1+np.power( (mualpha-E_n)/(2*Gam),2) );

    # conductance
    conductance = np.zeros_like(Vb);
    for n in ns:
        Enval = En(n,EC,Vg);
        conductance += deriv_arctan(muL, Enval, Gamma);
    return 1/4*conductance;

def I_of_Vb(Vb, mu0, Gamma, EC, Vg, kBT, ns, xvals=1e5):
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

    # current
    current = np.zeros_like(Vb);
    for n in ns:
        Enval = En(n,EC,Vg);
        tau_vals = Gamma*Gamma/( np.power(Evals-Enval,2)+np.power(Gamma+Gamma,2));
        # integrate to get current contribution
        integration_result = np.empty_like(Vb);
        for Vbi in range(len(Vb)):
            # chem potentials
            muL, muR = mu0+Vb[Vbi], mu0;
            integration_result[Vbi] = np.trapz(tau_vals*(nFD(Evals-muL,kBT)-nFD(Evals-muR,kBT)), Evals);
        #print(integration_result);
        current += integration_result;

    return current;

def dI_of_Vb(Vb, mu0, Gamma, EC, Vg, kBT, ns, xvals=1e5):
    '''
    Compute the finite temperature conductance by numerical integration
    '''
    if(not isinstance(Vb, np.ndarray)): raise TypeError;
    if(not isinstance(mu0, float)): raise TypeError;
    if(not isinstance(ns, np.ndarray)): raise TypeError;
    if(ns.dtype != int): raise TypeError;

    # chem potentials
    muL, muR = mu0+Vb, mu0;

    # variable of integration
    mulimits = np.array([mu0,mu0-np.min(Vb),mu0+np.min(Vb),mu0-np.max(Vb),mu0+np.max(Vb)]);
    Elimits = (-5*kBT+np.min(mulimits),+5*kBT+np.max(mulimits));
    Evals = np.linspace(*Elimits,int(xvals));
    print("Integration limits = ",Elimits);

    # conductance
    conductance = np.zeros_like(Vb);
    for n in ns:
        Enval = En(n,EC,Vg);
        tau_part = 1/(1+ np.power((Evals-Enval)/(2*Gamma),2))
        # integrate to get current contribution
        integration_result = np.empty_like(Vb);
        for Vbi in range(len(Vb)):
            nFD_part = np.exp((Evals-mu0-Vb[Vbi])/kBT)*np.power(np.exp((Evals-mu0-Vb[Vbi])/kBT)+1,-2);
            integration_result[Vbi] = np.trapz(tau_part*nFD_part, Evals);
        conductance += integration_result;

    return conductance/(4*kBT);


if(__name__ == "__main__"):

    # experimental params (Mn/5KExp.txt)
    Vb_max = 0.1;
    kelvin2eV =  8.617e-5; # units eV/K
    conductance_quantum = 7.748e-5; # units amp/volt
    conductance = True;

    if True: # plot with various methods

        # physical params, in eV
        my_mu0 = 0.0;
        my_EC = 0.025;
        my_Vg = my_EC;
        my_temp = 10.0*kelvin2eV;
        nmax = 3;
        narr = np.arange(-nmax,nmax+1);

        for my_Gamma in [0.00015, 0.015]:

            # plotting
            fig, axes = plt.subplots(2, sharex=True);
            Vbs = np.linspace(-Vb_max,Vb_max,int(1e3));
        
            # zero temp analytic
            Is = dI_of_Vb_zero(Vbs, my_mu0, my_Gamma, my_EC, my_Vg, 0.0, narr);
            axes[0].plot(Vbs, conductance_quantum*1e9*Is, label = "T=0 analytical");

            # zero temp numerical
            Is = np.gradient(I_of_Vb_zero(Vbs, my_mu0, my_Gamma, my_EC, my_Vg, 0.0, narr));
            axes[1].plot(Vbs, conductance_quantum*1e9*Is, label = "T=0 numerical");

            # finite temp analytical
            Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_Vg, my_temp, narr);
            axes[0].plot(Vbs, conductance_quantum*1e9*Is, label = "T!=0 analytical");

            # finite temp numerical
            Is = np.gradient(I_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_Vg, my_temp, narr));
            axes[1].plot(Vbs, conductance_quantum*1e9*Is, label = "T!=0 numerical");

            # peaks
            for integer in narr:
                print("En - mu0 = {:.3f}".format(En(integer, my_EC, my_Vg)-my_mu0));
                for ax in axes:
                    ax.axvline(En(integer, my_EC, my_Vg)-my_mu0,color="black",linestyle="dashed");

            # format
            axes[0].set_title( "$T = $ {:.1f} K, $\mu_0 = $ {:.3f} eV, $\Gamma = $ {:.3f} eV,\n\
                $E_C = $ {:.3f} eV, $eV_g = $ {:.3f} eV".format(my_temp/kelvin2eV, my_mu0, my_Gamma, my_EC, my_Vg));
            axes[-1].set_xlabel("$V_b$ (V)");
            for ax in axes:
                ax.set_ylabel("$dI/dV_b$ (nA/V)");
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
                ax.legend();
            plt.tight_layout();
            plt.show();

    if False: # plot at various temperatures
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e3));

        # physical params, in eV
        my_mu0 = 0.0;
        my_Gamma = 1e-4;
        my_EC = 0.025;
        my_Vg = my_EC;
        nmax = 3;
        narr = np.arange(-nmax,nmax+1);

        # iter over T
        Tvals = np.array([5,10,20])*kelvin2eV;
        to_save = np.empty((len(Tvals),len(Vbs)),dtype=float); to_savei=0;
        for Tval in Tvals:
            if(conductance): Is = dI_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_Vg, Tval, narr);
            else: Is = I_of_Vb(Vbs, my_mu0, my_Gamma, my_EC, my_Vg, Tval, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = "$T = $ {:.1f} K".format(Tval/kelvin2eV));
        for integer in narr:
            print("En - mu0 = {:.3f}".format(En(integer, my_EC, my_Vg)-my_mu0));
            ax.axvline(En(integer, my_EC, my_Vg)-my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$\mu_0 = $ {:.3f} eV, $\Gamma = $ {:.3f} eV,\n\
            $E_C = $ {:.3f} eV, $eV_g = $ {:.3f} eV".format( my_mu0, my_Gamma, my_EC, my_Vg));
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
        save_params = np.array([my_mu0, my_Gamma, my_EC, my_Vg, np.nan, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);

    if False: # plot at various EC
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e3));

        # physical params, in eV
        my_mu0 = -0.0;
        my_Gamma = 1e-4; 
        my_Vg = 0.0;
        my_temp = 0.0*kelvin2eV;
        nmax = 5;
        narr = np.arange(-nmax,nmax+1);

        # iter over EC
        ECvals = np.array([0.01,0.05]);
        to_save = np.empty((len(ECvals),len(Vbs)),dtype=float); to_savei=0;
        for ECval in ECvals:
            if(conductance): Is = dI_of_Vb_zero(Vbs, my_mu0, my_Gamma, ECval, my_Vg, my_temp, narr);
            else: Is = I_of_Vb_zero(Vbs, my_mu0, my_Gamma, ECval, my_Vg, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = "{:.3f}".format(ECval));
            for integer in [0]:
                print("En - mu0 = {:.3f}".format(En(integer, ECval, my_Vg)-my_mu0));
                ax.axvline(En(integer, ECval, my_Vg)-my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $\mu_0 = $ {:.3f} eV, $\Gamma = $ {:.3f} eV,\n\
            $eV_g = $ {:.3f} eV".format(my_temp/kelvin2eV, my_mu0, my_Gamma, my_Vg));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

        # save
        fname = "land_data/vsEC"
        save_params = np.array([my_mu0, my_Gamma, np.nan, my_Vg, my_temp, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);

    if False: # plot at various Gamma
        fig, ax = plt.subplots();
        Vbs = np.linspace(-Vb_max,Vb_max,int(1e6));

        # physical params, in eV
        my_mu0 = 0.0
        my_EC = 0.01;
        my_Vg = 0.0;
        my_temp = 0.0*kelvin2eV;
        nmax = 5;
        narr = np.arange(-nmax,nmax+1);

        # iter over EC
        Gammavals = np.array([1e-2,1e-3,1e-4,1e-5]);
        to_save = np.empty((len(Gammavals),len(Vbs)),dtype=float); to_savei=0;
        for Gammaval in Gammavals:
            if(conductance): Is = dI_of_Vb_zero(Vbs, my_mu0, Gammaval, my_EC, my_Vg, my_temp, narr);
            else: Is = I_of_Vb_zero(Vbs, my_mu0, Gammaval, my_EC, my_Vg, my_temp, narr);
            to_save[to_savei]=Is; to_savei += 1;
            ax.plot(Vbs, conductance_quantum*1e9*Is, label = str(Gammaval));
        for integer in [0]:
            print("En - mu0 = {:.3f}".format(En(integer, my_EC, my_Vg)-my_mu0));
            ax.axvline(En(integer, my_EC, my_Vg)-my_mu0,color="black",linestyle="dashed");

        # format
        ax.set_title( "$T = $ {:.1f} K, $\mu_0 = $ {:.3f} eV, \n\
            $E_C = $ {:.3f} eV, $eV_g = $ {:.3f} eV".format(my_temp/kelvin2eV, my_mu0, my_EC, my_Vg));
        ax.set_xlabel("$V_b$ (V)");
        if(conductance): ax.set_ylabel("$dI/dV_b$ (nA/V)");
        else: ax.set_ylabel("$I(V_b)$ (nA)");
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        plt.legend();
        plt.tight_layout();
        plt.show();

        # save
        fname = "land_data/vsGamma"
        save_params = np.array([my_mu0, np.nan, my_EC, my_Vg, my_temp, nmax]);
        print("Saving to "+fname);
        np.savetxt(fname+".txt",save_params);
        np.save(fname, to_save);
    
    
