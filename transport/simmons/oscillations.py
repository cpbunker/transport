'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''

from simmons_formula import plot_fit

import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit
import matplotlib.pyplot as plt

def load_dIdV(folder,temp):
    '''
    Get dIdV vs V data at a certain temp

    returns:
    V in volts, I in nano amps
    '''

    fname = "{:.0f}".format(temp) + "KdIdV.txt"
    print("Loading data from "+folder+fname);
    IV = np.loadtxt(folder+fname);
    Vs = IV[:, 0];
    dIs = IV[:, 1];
    if( len(np.shape(Vs)) != 1 or np.shape(Vs) != np.shape(dIs) ): raise TypeError;

    return Vs, 1e9*dIs;

###############################################################
#### fitting dI/dV with background and oscillations

def dIdV_background(Vb, alpha0, alpha2):
    '''
    '''

    # iter over even powers
    rets = np.zeros_like(Vb);
    rets += alpha0;
    rets += alpha2*np.power(Vb,2);
    #rets += alpha4*np.power(Vb,4);

    return rets;

def dIdV_osc(Vb, amplitude, ang_freq):
    '''
    '''

    return amplitude*np.sin(ang_freq*Vb);

def dIdV(Vb,amplitude, ang_freq, alpha0, alpha2):
    '''
    Designed to be passed to scipy.optimize.curve_fit
    '''

    rets = dIdV_osc(Vb,amplitude,ang_freq);
    rets += dIdV_background(Vb,alpha0,alpha2);
    return rets;

####################################################################
#### main

def fit_dIdV(metal, temp, area, amp_not, ang_not, amp_percent, ang_percent, verbose=0):
    '''
    '''

    V_exp, dI_exp = load_dIdV(metal, temp);

    #### fit experimental data to background
    params_back, pcov_back = scipy_curve_fit(dIdV_background, V_exp, dI_exp);
    dI_fit_back = dIdV_background(V_exp, *params_back);
    # visualize background fit
    if(verbose):
        print("dIdV_background fitting results:");
        print_str = " alphas = "+str(params_back);
        print(print_str);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit_back, mytitle = print_str);
    del params_back, pcov_back;
    assert False;

####################################################################
#### wrappers

def fit_Mn_data():
    metal="Mn/"; # points to data folder

    # experimental params
    Ts = [5,10,15,20,25,30];
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # guesses
    amp_guess = 0.1*np.ones_like(Ts);
    period_guess = 0.1*np.ones_like(Ts);
    ang_guess = 2*np.pi/period_guess;

    # bounds
    amp_percent, ang_percent = 0.5, 0.5;

    # fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        rlabels = ["Amplitude (nA/V", "Period (V)"];
        (temp_amp, temp_ang), (amp_bounds, ang_bounds) = fit_dIdV(metal,Ts[datai], area,
            amp_guess[datai], ang_guess[datai], amp_percent, ang_percent,
                                verbose=10);
        results.append(temp_results); 
        temp_bounds = np.append(temp_bounds, [[0],[0.1]], axis=1); # fake rmse bounds
        boundsT.append(temp_bounds);

if(__name__ == "__main__"):
    fit_Mn_data();
