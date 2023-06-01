'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''

from utils import load_IVb, load_dIdV_tocurrent, plot_fit

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
#### current density functions

def J_of_Vb_lowbias(Vb, d, phibar, m_r):
    '''
    Get the current density J as function of applied bias Vb
    from Simmons Eq 24, ie the low bias (ohmic) limit

    Vb, applied bias voltage, units volts
    d, barrier width, units nm
    phibar, avg barrier height, units eV
    m_r, ratio of eff electron mass to me, unitless
    
    Returns:
    J, current density, units amps/(nm^2)
    '''
    if( len(np.shape(Vb)) != 1): raise TypeError;

    # decay length
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    d_d = d_d_prefactor/np.sqrt(m_r*phibar); # decay length, units nm

    # current density
    J_prefactor = 9.685*1e-6 # =e^2/(8*\pi*\hbar), units amp/volt
    return J_prefactor * Vb/(d*d_d) * np.exp(-d/d_d); # units amps/(nm^2)

def J_of_Vb(Vb, d, phibar, m_r):
    '''
    Get the current density J as function of applied bias Vb
    from Simmons Eq 20

    Vb, applied bias voltage, units volts
    d, barrier width, units nm
    phibar, avg barrier height, units eV
    m_r, ratio of eff electron mass to me, unitless
    
    Returns:
    J, current density, units amps/(nm^2)
    '''
    if( len(np.shape(Vb)) != 1): raise TypeError;
    if(phibar < max(abs(Vb))): raise ValueError;
    raise NotImplementedError;

    # beta
    beta = 1.0; # unitless, depends on the specifics of \phi(x), see Simmons Eq A6

    # decay length
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    d_d = beta*d_d_prefactor/np.sqrt(m_r*phibar); # decay length, units nm
    d_d_withbias = beta*d_d_prefactor/np.sqrt(m_r*(phibar+Vb)); # decay length
                # when phibar is shifted by eVb, NB Vb has units eV

    # current density
    J_prefactor = 6.166*1e-6 # =e^2/(4*\pi^2*\hbar), units amp/volt
    J_0 = J_prefactor/(beta*d)**2 ; # units amp/(volt*nm^2)
    J_right = J_0*phibar*np.exp(-d/d_d); # NB phibar here has units volts
    J_left = J_0*(phibar+Vb)*np.exp(-d/d_d_withbias); # same
    return J_right - J_left; # units amp/nm^2

def J_of_Vb_asym(Vb, d, phi1, phi2, m_r):
    '''
    Get the current density J as function of applied bias Vb
    from Li Eq 1
    
    Returns:
    J, current density, units amps/(nm^2)
    '''
    if( len(np.shape(Vb)) != 1): raise TypeError;

    # current density
    J_prefactor = 6.166*1e-6 # =e^2/(4*\pi^2*\hbar), units amp/volt
    J_0 = J_prefactor/(d)**2 ; # units amp/(volt*nm^2)

    # asymmetry
    phibar = (phi1+phi2)/2; # average barrier height in eV
    c0 = 1+(phi2-Vb-phi1)**2 /(48*phibar*phibar); # asymmetry parameter, unitless

    # asymmetric decay lengths, units nm
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    d_d_minus = d_d_prefactor*np.sqrt(c0)/np.sqrt(m_r*(phibar-Vb/2)); # decay length for lowered side
    d_d_plus = d_d_prefactor*np.sqrt(c0)/np.sqrt(m_r*(phibar+Vb/2)); # decay length for lowered side
    
    # current density
    J_right = J_0*c0*(phibar-Vb/2)*(1+3*d_d_minus/d + 3*(d_d_minus/d)**2);
    J_right *= np.exp(-d/d_d_minus);
    J_left =  J_0*c0*(phibar+Vb/2)*(1+3*d_d_plus/d + 3*(d_d_plus/d)**2);
    J_left *= np.exp(-d/d_d_plus);

    #### checking ####
    if False:
        print("MY CODE");
        start = len(Vb)//2;
        thru = 1;
        print("Vb = ",Vb[start:start+thru]);
        print("c_0 = ", c0[start:start+thru], ", unitless");
        print("c_minus = ", (1+3*d_d_minus/d + 3*(d_d_minus/d)**2)[start:start+thru], ", unitless");
        print("c_plus = ", (1+3*d_d_plus/d + 3*(d_d_plus/d)**2)[start:start+thru], ", unitless");
        assert False
        
    return J_right - J_left;
    

###############################################################
#### current functions

def I_of_Vb_linear(Vb, I0, slope):
    '''
    Get the current density J as function of applied bias Vb
    for a simple linear model

    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    I0, y-intercept
    slope, slope

    Returns:
    I, current, nano amps
    '''

    return slope*Vb + I0;

def I_of_Vb_cubic(Vb, V0, slope):
    '''
    Get the current density J as function of applied bias Vb
    for a simple cubic model

    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    V0, x intercept

    Returns:
    I, current, nano amps
    '''

    return slope*np.power(Vb-V0,3);

def I_of_Vb_lowbias(Vb, d, phibar, m_r):
    '''
    Wraps J_of_Vb_lowbias so that
    - instead of phi1, phi2 separate fittable params, there is just phibar
    - Vbs are shifted by V0_kwarg
    - Is are put in nano amps and shifted by I0_kwarg
    
    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    d, barrier width, units nm
    phibar, avg barrier height, units eV
    m_r, ratio of eff electron mass to me, unitless

    Returns:
    I, current, nano amps
    '''

    Js = J_of_Vb_lowbias(Vb-V0_kwarg, d, phibar, m_r);
    Is = Js*convert_J2I_kwarg*1e9; # in nano amps
    return Is + I0_kwarg;

def I_of_Vb_asym(Vb, d, phi1, phi2, m_r):
    '''
    Wraps J_of_Vb_asym so that
    - Vbs are shifted by V0_kwarg
    - Is are put in nano amps and shifted by I0_kwarg
    
    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    d, barrier width, units nm
    phi1, effective barrier height near left lead, units eV
    phi2, effective barrier height near right lead, units eV
    m_r, ratio of eff electron mass to me, unitless

    Returns:
    I, current, nanao amps
    '''

    Js = J_of_Vb_asym(Vb-V0_kwarg, d, phi1, phi2, m_r);
    Is = Js*convert_J2I_kwarg*1e9; # in nano amps
    return Is + I0_kwarg;

def J2I(temp, area, thermal_exp = 0.0):
    '''
    Gets the conversion factor from current density J(Vb) in amps/nm^2
    to current in amps at a given temperature
    '''

    return area*( 1+thermal_exp*temp)**2;

###########################################################################
#### main fitting function

from utils import fit_wrapper

def fit_I(metal, temp, area, d_not, phi1_not, phi2_not, m_r_not,
            d_percent, phibar_percent, m_r_percent,
            lowbias_fit=False, Vcut=None, verbose=0):
    '''
    Given data for a specific temp, taken for a sample of certain area,
    1) fit a linear model of J(Vb) to get an x-intercept V0,
        a y-intercept J0, and a slope
    2) fit the Simmons model of J(Vb) to get a sample thickness d (in nm), 
        a sample avg barrier height phibar (in eV), and an electron effective
        mass ratio m_r (unitless)

    Args:
    Experimental params:
    metal, str, Co or Mn, tells which type of metalPc and what folder data is in
    temp, temperature in Kelvin
    area, area in nm^2
    Fitting params:
    d, barrier width, units nm
    phibar, avg barrier height, units eV.
            NB \phi_1 + \phi_2 =2\\bar{phi} eventually have to be fit independently
    m_r, ratio of eff electron mass to me, unitless
    
    the _not args are the initial guesses for the above fitting parameters
    the _percent args determine the fitting bounds for the above fitting parameters
    '''
    if( not isinstance(metal, str)): raise TypeError;

    # read in the experimental data
    if(metal == "Co/"): 
        base = "KExp.txt";
        V_exp, I_exp = load_IVb(base,metal,temp); # in volts, nano amps
    elif(metal == "Mn/"): 
        base = "KdIdV.txt";
        V_exp, I_exp = load_dIdV_tocurrent(base,metal,temp,Vcut);
    else: raise NotImplementedError;

    #### fit expermental data

    # linear fit for I0
    (I0, _), _ = fit_wrapper(I_of_Vb_linear, V_exp, I_exp, None, None, ["I0","slope"], verbose=verbose);

    # cubic fit for V0
    (V0, _), _ = fit_wrapper(I_of_Vb_cubic, V_exp, I_exp-I0, None, None, ["V0","slope"], verbose=verbose);

    # results of linear and cubic fits
    global V0_kwarg; V0_kwarg = V0; # very bad practice
    global I0_kwarg; I0_kwarg = I0;
    global convert_J2I_kwarg; convert_J2I_kwarg = J2I(temp, area);

    # low bias simmons fit at fixed d to narrow down phibar * m_r
    if lowbias_fit:

        phibar_not = (phi1_not+phi2_not)/2;
        init_params_low = [d_not, phibar_not, m_r_not];
        bounds_low = np.array([[d_not*(1-d_percent),phibar_not*(1-phibar_percent),m_r_not*(1-m_r_percent)],
                  [d_not*(1+d_percent),phibar_not*(1+phibar_percent),m_r_not*(1+m_r_percent)]]);

        (dlow, phibarlow, m_rlow), rmse_low = fit_wrapper(I_of_Vb_lowbias, V_exp, I_exp,
                                        init_params_low, bounds_low, ["d","phibar","m_r"], verbose=verbose);

        return ((dlow, phibarlow, m_rlow, rmse_low), bounds_low);

        # update guesses and narrow bounds
        '''
        phi1_not, m_r_not = tuple(params_lowbias[1:]);
        phi2_not = phi1_not;
        phibar_percent, m_r_percent = phibar_percent/10, m_r_percent/10;
        del init_params_lowbias, bounds_lowbias, params_lowbias, pcov_lowbias;
        '''
    
    # full simmons fit in narrowed bounds
    else:
        
        init_params = np.array([d_not, phi1_not, phi2_not, m_r_not]); 
        bounds = np.array([[d_not*(1-d_percent),phi1_not*(1-phibar_percent),phi2_not*(1-phibar_percent),m_r_not*(1-m_r_percent)],
                  [d_not*(1+d_percent),phi1_not*(1+phibar_percent),phi2_not*(1+phibar_percent),m_r_not*(1+m_r_percent)]]);

        # generalized Simmons fit
        (d, phi1, phi2, m_r), rmse_gen = fit_wrapper(I_of_Vb_asym, V_exp, I_exp,
                                            init_params, bounds, ["d","phi1","phi2","m_r"], verbose=verbose);

        return ((d, phi1, phi2, m_r, rmse_gen), bounds);

###############################################################
#### wrappers

def SLL_plot():

    # Shuanglong data
    SLL_I0 = 1e-10;
    SLL_temp = 40;
    SLL_area = (200*1e3)**2 *np.pi;
    convert_J2I = J2I(SLL_temp, SLL_area);
    SLL_J0 = SLL_I0/convert_J2I;
    plot_guess(SLL_temp, SLL_area,0.0, SLL_J0, 5.01, 1.90, 0.319);

def fit_Co_data(lowbias_fit=False):
    metal="Co/"; # points to data folder

    # experimental params
    Ts = np.array([40, 80, 120, 160, 200]);
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;
       
    #### define the search space
    if(not lowbias_fit):
        rlabels = ["$d$ (nm)", "$e\phi_1$ (eV)", "$e\phi_2$ (eV)",  "$m_r$", "RMSE"]

        # d
        d_guess = 5.0*np.ones_like(Ts);
        d_guess_abs = 0.3; # measurement good to a few angstrom
        d_guess_percent = abs(d_guess_abs/d_guess[0]);

        # m
        m_guess = 0.30*np.ones_like(Ts);
        m_guess_percent = 1e-6;

        # phi
        phibar_guess = 1.0*np.array([1.9,1.8,1.7,1.7,1.6]);
        phi_split = 0.4;
        phi1_guess = np.copy(phibar_guess);
        phi2_guess = np.copy(phibar_guess);
        phi1_guess += phi_split;
        phi2_guess -= phi_split;
        phi_guess_percent = 0.5;
        
    else:
        rlabels = ["$d$ (nm)", "$e\\bar{\phi}$ (eV)", "$m_r$", "RMSE"]
        
        # d
        d_guess = 5.0*np.ones_like(Ts);
        d_guess_percent = 1e-6;

        # m
        m_guess = 0.30*np.ones_like(Ts);
        m_guess_percent = 1e-6;

        # phi
        phibar_guess = 1.8*np.ones_like(Ts);
        phi_split = 0.4;
        phi1_guess = np.copy(phibar_guess);
        phi2_guess = np.copy(phibar_guess);
        phi1_guess += phi_split;
        phi2_guess -= phi_split;
        phi_guess_percent = 0.5;

    # fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        temp_results, temp_bounds = fit_I(metal,Ts[datai], area,
            d_guess[datai], phi1_guess[datai], phi2_guess[datai], m_guess[datai],
                d_guess_percent, phi_guess_percent, m_guess_percent,
                                lowbias_fit=lowbias_fit, verbose=1);
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
    axes[-1].set_xlabel("$T$ (K)");
    #axes[-1].set_ylim(0.0,0.1);
    plt.suptitle(metal[:-1]+" fitting");
    plt.show();

def fit_Mn_data(lowbias_fit):
    metal="Mn/"; # points to data folder

    # experimental params
    Ts = np.array([5,10,15,20,25,30]);
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;
        
    if(not lowbias_fit):
        pass;

    else:
        
        d_guess = 2.1*np.ones_like(Ts);
        d_guess_percent = 0.5;
                
        m_guess = 0.30*np.ones_like(Ts);
        m_guess_percent = 1e-6;

        phibar_guess = 5.8;
        phi_split = 0.0;
        phi1_guess = phibar_guess*np.ones_like(Ts);
        phi2_guess = phibar_guess*np.ones_like(Ts);
        phi1_guess += phi_split;
        phi2_guess -= phi_split;
        phi_guess_percent = 0.5;

    # fitting results
    raise Exception("Needs to be 3")
    Ts = Ts[:];
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        # normal way
        rlabels = ["$d$ (nm)", "$\phi_1$ (eV)", "$\phi_2$ (eV)",  "$m_r$", "RMSE"];
        temp_results, temp_bounds = fit_I(metal,Ts[datai], area,
            d_guess[datai], phi1_guess[datai], phi2_guess[datai], m_guess[datai],
                d_guess_percent, phi_guess_percent, m_guess_percent,
                                lowbias_fit=lowbias_fit, Vcut=0.1, verbose=1);
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
        #axes[resulti].plot(Ts,boundsT[:,0,resulti], color=accentcolors[0],linestyle='dashed');
        #axes[resulti].plot(Ts,boundsT[:,1,resulti], color=accentcolors[0],linestyle='dashed');
    axes[-1].set_xlabel("$T$ (K)");
    #axes[-1].set_ylim(0.0,0.1);
    plt.suptitle(metal[:-1]+" fitting");
    plt.show();

###############################################################
#### exec
if __name__ == "__main__":

    # args are lowbias_fit, refined
    fit_Co_data();

