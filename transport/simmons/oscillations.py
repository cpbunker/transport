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

kelvin2eV =  8.617e-5;
conductance_quantum = 7.748e-5; # units amp/volt

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
    kBT = kelvin2eV*temp_kwarg

    Gmag = np.zeros_like(Vb);
    Gmag += -2*kBT*np.log(1-np.exp(-E0/kBT));
    Gmag += (abs(Vb-V0)+E0)/(np.exp( (abs(Vb-V0)+E0)/kBT)-1);
    Gmag += (abs(Vb-V0)-E0)/(1-np.exp( -(abs(Vb-V0)-E0)/kBT ) );
    return G2+G3*Gmag;

def dIdV_back(Vb, V0, E0, G3, G2):
    '''
    '''

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

def dIdV_lorentz(Vb, V0, dI0, Gamma, EC, zero=True): 
    '''
    '''
    from landauer import dI_of_Vb_zero, dI_of_Vb

    nmax = 20;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    if(zero):
        return -dI0+1e9*conductance_quantum*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);
    else:
        return -dI0+1e9*conductance_quantum*dI_of_Vb(Vb-V0, mymu0, Gamma, EC, kelvin2eV*temp_kwarg, ns);

####################################################################
#### main

from utils import fit_wrapper

def fit_dIdV(metal, temp, area, V0_not, dI0_not, Gamma_not, EC_not,
             dI0_percent, Gamma_percent, EC_percent, rescale = 1, verbose=0):
    '''
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);   
    dI_sigma= np.std(dI_exp);
    dI_mu = np.mean(dI_exp);

    #### fit to background
    global temp_kwarg; temp_kwarg = temp; # very bad practice
    params_back_guess = np.array([0.0,Vlim/10, 5*dI_sigma, dI_mu]);
    bounds_back = [ [-1e-6],[1e-6] ];
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

    # <--- RESCALE <---
    dI_exp = rescale*dI_exp;
    dI_sigma= np.std(dI_exp);
    dI_mu = np.mean(dI_exp);

    # fit to sines
    params_sin_guess = np.array([np.pi/2, dI_sigma, Vlim/5]);
    bounds_sin = [[],[]];
    for pguess in params_sin_guess:
        bounds_sin[0].append(pguess*(1-1));
        bounds_sin[1].append(pguess*(1+1));
    bounds_sin = np.array(bounds_sin);
    if True:
        _ = fit_wrapper(dIdV_sin, V_exp, dI_exp,
                        params_sin_guess, bounds_sin, ["alpha","amp","per"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");

    # fit to lorentzians
    params_guess = [V0_not, dI0_not, Gamma_not, EC_not];
    bounds = np.array([ [-Vlim/5, dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
               [ Vlim/5, dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);

    if True:
        (V0, dI0, Gamma, EC), rmse = fit_wrapper(dIdV_lorentz, V_exp, dI_exp,
                             params_guess, bounds, ["V0","dI0","Gamma", "EC"], verbose=verbose, myylabel="$dI/dV_b$ (nA/V)");
        results = (dI0, Gamma, EC, rmse);
    else:
        dI_fit = dIdV_lorentz(V_exp, *params_guess, zero=False);
        if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit);

    if(verbose==10): assert False;
    return (results, bounds[:,1:]);

####################################################################
#### wrappers

def fit_Mn_data():
    metal="Mn/"; # points to data folder

    # experimental params
    kelvin2eV =  8.617e-5;
    Ts = np.array([5.0,10.0,15.0,20.0,25.0,30.0]);
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # <--- RESCALE <---
    rescale = 1;

    # guesses
    V0_guess = -0.0044*np.ones_like(Ts);
    dI0_guess = 57800*np.ones_like(Ts);
    Gamma_guess = 0.0048*np.ones_like(Ts); # gamma dominates T in the smearing
    EC_guess = (0.0196/4)*np.ones_like(Ts);
    dI0_percent = 0.2;
    Gamma_percent = 1.0;
    EC_percent = 0.1;

    # modify if rescaling
    if(rescale > 1):
        dI0_guess = 31000*np.ones_like(Ts);
        Gamma_guess = 0.003*np.ones_like(Ts);

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.1f} K, {:.4f} eV".format(Ts[datai], Ts[datai]*kelvin2eV));
        rlabels = ["$dI0$ (nA/V)", "$\Gamma$ (eV)", "$E_C$ (eV)", "RMSE"];

        # get fit results
        temp_results, temp_bounds = fit_dIdV(metal,Ts[datai], area,
            V0_guess[datai], dI0_guess[datai], Gamma_guess[datai], EC_guess[datai],
            dI0_percent, Gamma_percent, EC_percent, rescale=rescale, verbose=10);
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
