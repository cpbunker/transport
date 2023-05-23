'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''

from utils import plot_fit, load_dIdV

import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit
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

def dIdV_background(Vb, V0, alpha0, alpha2):
    '''
    '''

    # iter over even powers
    rets = np.zeros_like(Vb);
    rets += alpha0;
    rets += alpha2*np.power(Vb-V0,2);
    #rets += alpha4*np.power(Vb-V0,4);

    return rets;

def dIdV_osc(Vb, phi, amplitude, period):
    '''
    '''

    ang_freq = 2*np.pi/period
    return amplitude*np.sin(ang_freq*Vb-phi);

def dIdV(Vb, phi, amplitude, period):
    '''
    Designed to be passed to scipy.optimize.curve_fit
    '''

    rets = dIdV_osc(Vb,phi,amplitude, period);
    rets += dIdV_background(Vb,*alphas_kwarg);
    return rets;

####################################################################
#### main

def fit_dIdV(metal, temp, area, phi_not, amp_not, period_not,
             phi_percent, amp_percent, period_percent, verbose=0):
    '''
    '''

    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp);

    #### fit to background
    from simmons_formula import I_of_Vb_linear, I_of_Vb_cubic

    #### fit experimental data to non-ohmic part
    params_back_guess = [0.0,np.mean(dI_exp),np.max(dI_exp)-np.mean(dI_exp)];
    params_back, pcov_back = scipy_curve_fit(dIdV_background, V_exp, dI_exp, p0 = params_back_guess);
    dI_fit_back = dIdV_background(V_exp, *params_back);
    # visualize background fit
    if(verbose):
        print_str = "dIdV_background fitting results:\n";
        print_str += "        V0 = "+str(params_back[0])+" (V)\n";
        print_str += "     alphas = "+str(params_back[1:].round(1));
        print(print_str);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit_back, mytitle = print_str[:print_str.find("(")]);
    # save results of background fit
    global alphas_kwarg; alphas_kwarg = tuple(params_back);
    del params_back, pcov_back;

    #### fit oscillations
    params_guess = [phi_not,amp_not, period_not];
    bounds = [[phi_not*(1-phi_percent), amp_not*(1-amp_percent), period_not*(1-period_percent)],
                  [phi_not*(1+phi_percent),amp_not*(1+amp_percent), period_not*(1+period_percent)]];
    params, pcov = scipy_curve_fit(dIdV, V_exp, dI_exp,
                                           p0 = params_guess, bounds = bounds);
    dI_fit = dIdV(V_exp, *params);
    rmse_final =  np.sqrt( np.mean( (dI_exp-dI_fit)*(dI_exp-dI_fit) ))/abs(np.mean(dI_exp));
    if(verbose):
        print("dIdV fitting results:");
        print_str = "        phi = {:6.4f} "+str((bounds[0][0],bounds[1][0]))+" V\n\
        amp = {:6.4f} "+str((bounds[0][1],bounds[1][1]))+" nA/V\n\
        per = {:6.4f} "+str((bounds[0][2],bounds[1][2]))+" nA/V\n\
        err = {:6.4f}";
        print(print_str.format(*params, rmse_final));
    if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit);

    results = list(params);
    results.append(rmse_final);
    return (tuple(results), bounds)

####################################################################
#### wrappers

def fit_Mn_data():
    metal="Mn/"; # points to data folder

    # experimental params
    Ts = [5,10,15,20,25,30];
    #Ts = Ts[:1];
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # guesses
    phi_guess = np.pi*np.ones_like(Ts);
    amp_guess = 100*np.ones_like(Ts);
    period_guess = 0.02*np.ones_like(Ts);

    # bounds
    phi_percent, amp_percent, period_percent = 1.0,0.99, 0.1;

    # fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        rlabels = ["$\\alpha$ (rad)","$A$ (nA/V)", "$\Delta V$ (V)", "RMSE"];
        temp_results, temp_bounds = fit_dIdV(metal,Ts[datai], area,
            phi_guess[datai], amp_guess[datai], period_guess[datai],
                    phi_percent, amp_percent, period_percent,
                                verbose=1);
        results.append(temp_results); 
        temp_bounds = np.append(temp_bounds, [[0],[0.2]], axis=1); # fake rmse bounds
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
    Tenergies = 8.617e-5*np.array(Ts); # in eV
    C0energy = results[0,-2]; # e * Delta Vb = energy scale = e^2/C0
    print(Tenergies)
    print(C0energy);
    #axes[-2].plot(Ts, C0energy-Tenergies,color=accentcolors[1]);

    # format
    axes[-1].set_xlabel("$T$ (K)");
    plt.show();

if(__name__ == "__main__"):
    fit_Mn_data();
