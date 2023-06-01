'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''

from utils import plot_fit, load_dIdV

import numpy as np
import matplotlib.pyplot as plt

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

###############################################################
#### fitting dI/dV with background and oscillations

def dIdV_quad(Vb, V0, dI0, alpha2):
    '''
    quadratic background to sense V0 and dI0
    '''

    # even powers
    rets = np.zeros_like(Vb);
    rets += dI0;
    rets += alpha2*np.power(Vb-V0,2);

    return rets;

def dIdV_back(Vb, E0, G3, G2):
    '''
    '''

    # the F function from XGZ's magnon paper, Eq (17)
    def Ffunc(E, kBT):

        numerator = np.log(1+ E0/(E+kBT));
        denominator = 1 - kBT/(E0+0.4*E) + 12*kBT*kBT/np.power(E0+2.4*E,2);
        return numerator/denominator;

    kelvin2eV =  8.617e-5;
    return G2 - G3*Ffunc(abs(Vb-V0_kwarg), kelvin2eV*temp_kwarg);

def dIdV_sin(Vb, alpha, amplitude, period):
    '''
    Designed to be passed to scipy.optimize.curve_fit
    '''

    ang_freq = 2*np.pi/period
    return amplitude*np.sin(ang_freq*(Vb-V0_kwarg)-alpha);

def dIdV_lorentz(Vb, EC, mutilde):
    from blockade import I_of_Vb
    kelvin2eV =  8.617e-5;
    nmax = 10;
    return np.gradient( I_of_Vb(Vb, EC, mutilde, kelvin2eV*temp_kwarg, nmax) );


####################################################################
#### main

from utils import fit_wrapper

def fit_dIdV(metal, temp, area, phi_not, amp_not, period_not,
             phi_percent, amp_percent, period_percent, lorentzian=False,nsigma=6,verbose=0):
    '''
    '''

    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp);

    #### symmetrize and trim outliers
    if(verbose>4): # show before processing
        outfig, outax = plt.subplots();
        outax.scatter(V_exp, dI_exp, color="cornflowerblue",marker="o");

    # symmetrize
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_exp = dI_exp[abs(V_exp)<=Vlim];
    V_exp = V_exp[abs(V_exp)<=Vlim];

    # trim outliers
    dI_sigma= np.std(dI_exp);
    dI_mu = np.mean(dI_exp);
    V_exp = V_exp[ abs(dI_exp-dI_mu) < nsigma*dI_sigma];
    dI_exp = dI_exp[ abs(dI_exp-dI_mu) < nsigma*dI_sigma];
    
    if(verbose>4): # show after processing
        outax.scatter(V_exp, dI_exp, color="black",marker="+");
        outax.set_title("Processing data");
        outax.set_xlabel("$V_b$ (V)");
        outax.set_ylabel("$dI/dV_b$ (nA/V)");
        plt.tight_layout();
        plt.show();

    #### fit V0
    params_quad_guess = [0.0,dI_mu, dI_sigma];
    (V0, _, _), _ = fit_wrapper(dIdV_quad, V_exp, dI_exp,
                                  params_quad_guess, None, ["V0", "dI0","alpha2"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");

    global V0_kwarg; V0_kwarg = V0; # very bad practice
    global temp_kwarg; temp_kwarg = temp;

    #### fit to background
    params_back_guess = np.array([2*Vlim, 2*dI_sigma, dI_mu]);
    bounds_back = np.array([[params_back_guess[0]*(1-1),params_back_guess[1]*(1-1), params_back_guess[2]*(1-1)],
                            [params_back_guess[0]*(1+1),params_back_guess[1]*(1+1), params_back_guess[2]*(1+1)]]);
    (E0, G3, G2), _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_back_guess, bounds_back, ["E0", "G3","G2"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");

    #### fit oscillations

    # subtract background
    background = dIdV_back(V_exp, E0, G3, G2);
    dI_exp = dI_exp - background;

    # fit to either sines or lorentzians
    if lorentzian: # lorentzians
        params_guess = [0.005,0.001];
        bounds = np.array([[params_guess[0]*(1-1),params_guess[1]*(1-1)],
                            [params_guess[0]*(1+1),params_guess[1]*(1+1)]]);
        (EC, mutilde), rmse = fit_wrapper(dIdV_lorentz, V_exp, dI_exp,
                             params_guess, None, ["EC","mutilde"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");
        results = (EC, mutilde, rmse);
        
    else: # sines
        params_guess = np.array([phi_not,amp_not, period_not]);
        bounds = np.array([[phi_not*(1-phi_percent), amp_not*(1-amp_percent), period_not*(1-period_percent)],
                  [phi_not*(1+phi_percent),amp_not*(1+amp_percent), period_not*(1+period_percent)]]);
 
        (alpha_ang, amp, period), rmse = fit_wrapper(dIdV_sin, V_exp, dI_exp,
                            params_guess, bounds, ["alpha","amp","per"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");
        results = (alpha_ang, amp, period, rmse);

    return (results, bounds)

####################################################################
#### wrappers

def fit_Mn_data():
    metal="Mn/"; # points to data folder

    # experimental params
    Ts = np.array([5,10,15,20,25,30]);
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # guesses
    phi_guess = (1*np.pi/1)*np.ones_like(Ts);
    amp_guess = 100*np.ones_like(Ts);
    period_guess = 0.02*np.ones_like(Ts);

    # bounds
    phi_percent, amp_percent, period_percent = 1.0,0.99, 0.1;

    # how to do oscillation fit
    lorentzian = False;

    # fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        if(lorentzian):
            rlabels = ["$E_C$ (eV)", "$\\tilde{\mu}$ (eV)", "RMSE"];
        else:
            rlabels = ["$\\alpha$ (rad)","$A$ (nA/V)", "$\Delta V$ (V)", "RMSE"];

        # get fit results
        temp_results, temp_bounds = fit_dIdV(metal,Ts[datai], area,
            phi_guess[datai], amp_guess[datai], period_guess[datai],
                    phi_percent, amp_percent, period_percent,
                                verbose=1, lorentzian=lorentzian);
        results.append(temp_results); 
        temp_bounds = np.append(temp_bounds, [[0],[0.1]], axis=1); # fake rmse bounds
        boundsT.append(temp_bounds);

    # plot fitting results vs T
    results, boundsT = np.array(results), np.array(boundsT);
    nresults = len(results[0]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    for resulti in range(nresults):
        axes[resulti].plot(Ts, results[:,resulti], color=mycolors[0],marker=mymarkers[0]);
        axes[resulti].set_ylabel(rlabels[resulti]);
        axes[resulti].plot(Ts,boundsT[:,0,resulti], color=accentcolors[0],linestyle='dashed');
        axes[resulti].plot(Ts,boundsT[:,1,resulti], color=accentcolors[0],linestyle='dashed');

    # plot temp dependence of period
    Tenergies = 8.617e-5*np.array(Ts-5); # in eV
    C0energy = results[0,-2]; # e * Delta Vb = energy scale = e^2/C0
    print(Tenergies)
    print(C0energy);
    #axes[-2].plot(Ts, C0energy-Tenergies,color=accentcolors[1]);

    # format
    axes[-1].set_xlabel("$T$ (K)");
    plt.show();

if(__name__ == "__main__"):
    fit_Mn_data();
