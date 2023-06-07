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
    assert False

    # even powers
    rets = np.zeros_like(Vb);
    rets += dI0;
    rets += alpha2*np.power(Vb-V0,2);

    return rets;

def dIdV_mag(Vb, V0, E0, G3, G2):
    '''
    '''
    kelvin2eV =  8.617e-5;
    kBT = kelvin2eV*temp_kwarg

    Gmag = -2*kBT*np.log(1-np.exp(-E0/kBT));
    Gmag += (abs(Vb-V0)+E0)/(np.exp( (abs(Vb-V0)+E0)/kBT)-1);
    Gmag += (abs(Vb-V0)-E0)/(1-np.exp( -(abs(Vb-V0)-E0)/kBT ) );
    return G2-G3*Gmag;

def dIdV_back(Vb, V0, E0, G3, G2):
    '''
    '''
    kelvin2eV =  8.617e-5;

    # the F function from XGZ's magnon paper, Eq (17)
    def Ffunc(E, kBT):

        numerator = np.log(1+ E0/(E+kBT));
        denominator = 1 - kBT/(E0+0.4*E) + 12*kBT*kBT/np.power(E0+2.4*E,2);
        return numerator/denominator;

    return G2 - G3*Ffunc(abs(Vb-V0), kelvin2eV*temp_kwarg);

def dIdV_sin(Vb, alpha, amplitude, period):
    '''
    Designed to be passed to scipy.optimize.curve_fit
    '''

    ang_freq = 2*np.pi/period
    return amplitude*np.sin(ang_freq*Vb-alpha);

def dIdV_lorentz(Vb, V0, EC, mutilde, amp): # add a constant offset ?
    '''
    '''
    from blockade import I_of_Vb
    
    kelvin2eV =  8.617e-5;
    nmax = 10;
    return amp*np.gradient( I_of_Vb(Vb-V0, EC, mutilde, kelvin2eV*temp_kwarg, nmax) );


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

    
    # don't do this
    
    dI_sigma= np.std(dI_exp);
    dI_mu = np.mean(dI_exp);
    V_exp = V_exp[ abs(dI_exp-dI_mu) < nsigma*dI_sigma];
    dI_exp = dI_exp[ abs(dI_exp-dI_mu) < nsigma*dI_sigma];
    
    if(verbose>4): # show after processing
        outax.scatter(V_exp, dI_exp, color="black",marker="+");
        outax.set_title("Cleaning data");
        outax.set_xlabel("$V_b$ (V)");
        outax.set_ylabel("$dI/dV_b$ (nA/V)");
        plt.tight_layout();
        plt.show();

    #### fit to background
    global temp_kwarg; temp_kwarg = temp; # very bad practice
    params_back_guess = np.array([-0.01,2*Vlim, 2*dI_sigma, dI_mu]);
    bounds_back = [ [-Vlim/5],[Vlim/5] ];
    for pguess in params_back_guess[1:]:
        bounds_back[0].append(pguess*(1-1));
        bounds_back[1].append(pguess*(1+1));
    bounds_back = np.array(bounds_back);
    params_back, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_back_guess, bounds_back, ["V0", "E0", "G3","G2"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");

    #### fit oscillations

    # subtract background
    background = dIdV_back(V_exp, *params_back);
    dI_exp = dI_exp - background;

    # fit to either sines or lorentzians
    if lorentzian: # lorentzians
        dI_exp = dI_exp - np.min(dI_exp);
        params_guess = [Vlim/10,0.005,1e-6,1e4];
        bounds = [ [-Vlim/5],[Vlim/5] ];
        for pguess in params_guess[1:]:
            bounds[0].append(pguess*(1-1));
            bounds[1].append(pguess*(1+1));
        bounds = np.array(bounds);
        (V0, EC, mutilde, amp), rmse = fit_wrapper(dIdV_lorentz, V_exp, dI_exp,
                             params_guess, bounds, ["V0","EC","mutilde","Amp"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");
        results = (EC, mutilde, amp, rmse);
        
    else: # sines
        params_guess = np.array([phi_not,amp_not, period_not]);
        bounds = np.array([[phi_not*(1-phi_percent), amp_not*(1-amp_percent), period_not*(1-period_percent)],
                  [phi_not*(1+phi_percent),amp_not*(1+amp_percent), period_not*(1+period_percent)]]);
 
        (alpha_ang, amp, period), rmse = fit_wrapper(dIdV_sin, V_exp, dI_exp,
                            params_guess, bounds, ["alpha","amp","per"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");
        results = (alpha_ang, amp, period, rmse);

    if(verbose==10): assert False
    return (results, bounds)

####################################################################
#### wrappers

def fit_Mn_data():
    metal="Mn/"; # points to data folder

    # experimental params
    Ts = np.array([5,10,15,20,25,30]);
    Ts = Ts[:5];
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # guesses
    phi_guess = (1*np.pi/1)*np.ones_like(Ts);
    amp_guess = 100*np.ones_like(Ts);
    period_guess = 0.02*np.ones_like(Ts);

    # bounds
    phi_percent, amp_percent, period_percent = 1.0,0.99, 0.1;

    # how to do oscillation fit
    lorentzian = True;

    # fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        if(lorentzian):
            rlabels = ["$E_C$ (eV)", "$eV_g + \mu_0$ (eV)", "Amp", "RMSE"];
        else:
            rlabels = ["$\\alpha$ (rad)","$A$ (nA/V)", "$\Delta V$ (V)", "RMSE"];

        # get fit results
        temp_results, temp_bounds = fit_dIdV(metal,Ts[datai], area,
            phi_guess[datai], amp_guess[datai], period_guess[datai],
                    phi_percent, amp_percent, period_percent,
                                verbose=10, lorentzian=lorentzian);
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

    # format
    axes[-1].set_xlabel("$T$ (K)");
    plt.show();

if(__name__ == "__main__"):
    fit_Mn_data();
