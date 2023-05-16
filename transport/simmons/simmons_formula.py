'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''
import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit
import matplotlib.pyplot as plt

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
        #assert False
        
    return J_right - J_left;

def J_of_Vb_combined(Vb, d_rt_m, d_pre, phi1, phi2):
    '''
    J of Vb as in https://arxiv.org/abs/2204.13152
    '''

    # current density 
    J_prefactor = 6.166*1e-6 # =e^2/(4*\pi^2*\hbar), units amp/volt
    J_0 = J_prefactor/(d_pre)**2 ; # units amp/(volt*nm^2)
    # NB d does not enter in

    # asymmetry
    phibar = (phi1+phi2)/2; # average barrier height in eV
    c0 = 1+(phi2-Vb-phi1)**2 /(48*phibar*phibar); # asymmetry parameter, unitless

    # asymmetric decay lengths, units nm
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    d_d_minus = d_d_prefactor*np.sqrt(c0)/np.sqrt((phibar-Vb/2)); # decay length for lowered side
    d_d_plus = d_d_prefactor*np.sqrt(c0)/np.sqrt((phibar+Vb/2)); # decay length for lowered side
    
    # current density
    J_right = J_0*(c0)*(phibar-Vb/2)*(1+3*d_d_minus/d_rt_m + 3*(d_d_minus/d_rt_m)**2);
    J_right *= np.exp(-d_rt_m/d_d_minus);
    J_left =  J_0*(c0)*(phibar+Vb/2)*(1+3*d_d_plus/d_rt_m + 3*(d_d_plus/d_rt_m)**2);
    J_left *= np.exp(-d_rt_m/d_d_plus);

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

def I_of_Vb_combined(Vb, d_rt_m, deltaphi, phi1=1.8, d_pre=5.0):
    '''
    '''

    Js = J_of_Vb_combined(Vb-V0_kwarg, d_rt_m, d_pre, phi1, phi1+deltaphi);
    Is = Js*convert_J2I_kwarg*1e9; # in nano amps
    return Is + I0_kwarg;

def J2I(temp, area, thermal_exp = 0.5*1e-4):
    '''
    Gets the conversion factor from current density J(Vb) in amps/nm^2
    to current in amps at a given temperature
    '''

    return area*( 1+thermal_exp*temp)**2;

###############################################################
#### plotting functions

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

def plot_fit(V_exp, I_exp, I_fit, mytitle = ''):
    '''
    '''
    if( np.shape(I_exp) != np.shape(I_fit) ): raise TypeError;
    fig, ax = plt.subplots();
   
    # plot
    slope = (I_fit[-1]-I_fit[0])/(V_exp[-1]-V_exp[0]);
    ax.scatter(V_exp, I_exp, color=mycolors[0], label = "Exp.", linewidth = mylinewidth);
    ax.plot(V_exp, I_fit, color=accentcolors[0], label = "Fit", linewidth = mylinewidth);

    # error
    error = np.sqrt( np.mean( (I_fit - I_exp)*(I_fit - I_exp) ));
    norm_error = error/np.mean(I_exp);
    ax.plot( [0.0], [error], color='white', label = "Error = {:1.2f} ".format(norm_error));
 
    # format
    ax.set_xlabel("V (V)");
    ax.set_ylabel("I (nA)");
    plt.legend();
    plt.title(mytitle, fontsize = myfontsize);
    plt.show();

def plot_guess(temp, area, V0_not, J0_not, d_not, phibar_not, m_r_not):

    # experimental params
    Vmax = 1.0;
    Vbs = np.linspace(-Vmax,Vmax,myxvals);

    # current density at this guess
    Js = J_of_Vb_asym(Vbs-V0_not, d_not, phibar_not, phibar_not, m_r_not)+J0_not;
    #print(Js); assert False; # should be of order 1e-22

    # current
    Is = Js*J2I(temp, area);
    I0_not = J0_not*J2I(temp, area);

    # plot
    nano = 1e9;
    fig, ax = plt.subplots();
    ax.plot(Vbs, nano*Is, color=accentcolors[0], linewidth = mylinewidth);

    # format
    ax.set_xlabel("V (V)");
    ax.set_ylabel("I (nA)");
    plt.title("T = {:.0f} K, V0 = {:1.2f} V, I0 = {:1.2f} nA".format(temp, V0_not, nano*I0_not), fontsize = myfontsize);
    plt.show();

###############################################################
#### fitting functions

def load_IVb(temp,folder):
    '''
    Get I vs V data at a certain temp

    returns:
    V in volts, I in nano amps
    '''

    fname = "{:.0f}".format(temp) + "KExp.txt"
    IV = np.loadtxt(folder+fname);
    Vs = IV[:, 0];
    Is = IV[:, 1];
    if( len(np.shape(Vs)) != 1 or np.shape(Vs) != np.shape(Is) ): raise TypeError;

    return Vs, 1e9*Is;

def fit_I_combined(temp, area, dtilde_not, deltaphi_not,
            dtilde_percent, deltaphi_percent, verbose=0):
    '''
    Given data for a specific temp, taken for a sample of certain area,
    1) fit a linear model of J(Vb) to get an x-intercept V0,
        a y-intercept J0, and a slope
    2) fit the Simmons model of J(Vb) to get a sample thickness d (in nm), 
        a sample avg barrier height phibar (in eV), and an electron effective
        mass ratio m_r (unitless)

    Args:
    temp, temperature in Kelvin
    area, area in nm^2
    the _not args are the initial guesses for the above fitting parameters
    '''

    # read in the experimental data
    V_exp, I_exp = load_IVb(temp); # in volts nano amps

    #### fit expermental data

    # interpolate V0, I0, and slope guesses from exp data
    V0_not = V_exp[len(V_exp)//2];
    I0_not = I_exp[len(I_exp)//2];
    slope_not = (I_exp[-1] - I_exp[0])/(V_exp[-1] - V_exp[0]);

    # set up linear fit
    init_params_linear = [I0_not, slope_not];
    bounds_linear = [[I0_not*0.01, slope_not*0.01],
                     [I0_not*100, slope_not*100]];

    # linear fit gives I0
    params_linear, pcov_linear = scipy_curve_fit(I_of_Vb_linear, V_exp, I_exp,
                                p0=init_params_linear, bounds=bounds_linear);
    I0, slopeI = tuple(params_linear);
    I_fit_linear = I_of_Vb_linear(V_exp, I0, slopeI);
    if(verbose):
        print("I_of_Vb_linear fitting results:");
        print_str = "           I0 = {:6.4f} "+str((bounds_linear[0][0],bounds_linear[1][0]))+"nA\n\
        slope = {:6.4f} "+str((bounds_linear[0][1],bounds_linear[1][1]))+" nA/V"
        print(print_str.format(I0, slopeI));
    if(verbose > 4): plot_fit(V_exp, I_exp, I_fit_linear, mytitle = print_str[:print_str.find("(")].format(I0));
    del init_params_linear, bounds_linear, params_linear, pcov_linear;

    # set up cubic fit
    init_params_cubic = [V0_not, slope_not];
    bounds_cubic = [[V_exp[0], slope_not*0.01],
                 [V_exp[-1], slope_not*100]];
    
    # cubic fit gives V0
    params_cubic, pcov_cubic = scipy_curve_fit(I_of_Vb_cubic, V_exp, I_exp-I0,
                                p0=init_params_cubic, bounds=bounds_cubic);
    V0, slopeI = tuple(params_cubic);
    I_fit_cubic = I_of_Vb_cubic(V_exp, V0, slopeI);
    if(verbose):
        print("I_of_Vb_cubic fitting results:");
        print_str = "           V0 = {:6.4f} "+str((bounds_cubic[0][0],bounds_cubic[1][0]))+" V\n\
        slope = {:6.4f} "+str((bounds_cubic[0][1],bounds_cubic[1][1]))+" nA/V"
        print(print_str.format(V0, slopeI));
    if(verbose > 4): plot_fit(V_exp, I_exp, I_fit_cubic+I0, mytitle = print_str[:print_str.find("(")].format(V0));
    del init_params_cubic, bounds_cubic, params_cubic, pcov_cubic;

    # results of linear and cubic fits
    #global V0_kwarg; V0_kwarg = V0; # very bad practice
    #global I0_kwarg; I0_kwarg = I0;
    #global convert_J2I_kwarg; convert_J2I_kwarg = J2I(temp, area);

    # set up simmons fit
    init_params = [dtilde_not, deltaphi_not];
    bounds = np.array([[dtilde_not*(1-dtilde_percent),deltaphi_not*(1-deltaphi_percent)],
              [dtilde_not*(1+dtilde_percent),deltaphi_not*(1+deltaphi_percent)]]);

    # full simmons fit in narrowed bounds
    params, pcov = scipy_curve_fit(I_of_Vb_combined, V_exp, I_exp,
                            p0 = init_params, bounds = bounds, max_nfev = 1e6);
    dtilde, deltaphi = tuple(params);
    I_fit = I_of_Vb_combined(V_exp, dtilde, deltaphi);
    rmse_final =  np.sqrt( np.mean( (I_exp-I_fit)*(I_exp-I_fit) ))/np.mean(I_exp);
    if(verbose):
        print("I_of_Vb_asym_wrapped fitting results:");
        print_str = "            dtilde = {:6.4f} "+str((bounds[0][0],bounds[1][0]))+" nm\n\
          delta phi = {:6.4f} "+str((bounds[0][1],bounds[1][1]))+" eV\n\
          error = {:6.4f}";
        print(print_str.format(dtilde, deltaphi, rmse_final));
    if(verbose > 4): plot_fit(V_exp, I_exp, I_fit, mytitle = print_str[:print_str.find("(")].format(dtilde)+" nm");

    return (dtilde, deltaphi, rmse_final);

def my_curve_fit(fx, xvals, fxvals, init_params, bounds, focus_i, focus_meshpts=10, verbose=0):
    '''
    On top of scipy_curve_fit, build extra resolution to the
    dependence of the error on params[focus_i]
    '''
    if(not isinstance(bounds, np.ndarray)): raise TypeError;
    if(focus_i >= len(init_params)): raise ValueError;

    # mesh search over focus param vals
    focus_lims = np.linspace(bounds[0][focus_i],bounds[1][focus_i],focus_meshpts);
    focus_opts = np.empty((len(focus_lims)-1,));
    focus_errors = np.empty((len(focus_lims)-1,));    
    for fvali in range(len(focus_lims)-1):

        # update guess
        init_fparams = np.copy(init_params);
        init_fparams[focus_i] = (focus_lims[fvali] + focus_lims[fvali+1])/2;

        # truncate focus bounds
        fbounds = np.copy(bounds);
        fbounds[0][focus_i], fbounds[1][focus_i] = focus_lims[fvali], focus_lims[fvali+1];

        # fit within this narrow range
        nano, convert_J2I = 1e9, 1;
        #plot_guess(myT, myA, 0.0, 1e-10/convert_J2I, init_params[0], ):
        fparams, pcov = scipy_curve_fit(fx, xvals, fxvals,
                                 p0 = init_fparams, bounds = fbounds, max_nfev = 1e6);
        fxvals_fit = fx(xvals, *fparams);
        ferror = np.sqrt( np.mean( (fxvals-fxvals_fit)*(fxvals-fxvals_fit) ))/np.mean(fxvals);

        # update error and optimum
        focus_opts[fvali] = fparams[focus_i];
        focus_errors[fvali] = ferror;

        # visualize
        if(verbose > 2):
            print("my_curve_fit fitting results, focus_i = ",focus_i);
            print_str = "            d = {:6.4f} "+str((fbounds[0][0],fbounds[1][0]))+" nm\n\
            phi = {:6.4f} "+str((fbounds[0][1],fbounds[1][1]))+" eV\n\
            m_r = {:6.4f} "+str((fbounds[0][2],fbounds[1][2]))+"\n\
            err = {:6.4f}";
            print(print_str.format(*fparams, ferror));
        if(verbose > 4): plot_fit(xvals, fxvals*convert_J2I*nano, fxvals_fit*convert_J2I*nano,
                                  mytitle = "d = {:1.2f} nm, phi = {:1.2f} eV, m_r = {:1.2f} ".format(*fparams));

    # show the results of the mesh search
    if verbose:
        fig, ax = plt.subplots();
        ax.plot(focus_opts, focus_errors, color=accentcolors[0]);

        # format
        ax.set_xlabel("Params["+str(focus_i)+"]");
        ax.set_ylabel("Norm. RMS Error");
        #ax.set_yscale('log');
        plt.show();
    return focus_opts, focus_errors;

###########################################################################
#### main

def fit_I(temp, area, d_not, phibar_not, m_r_not,
            d_percent, phibar_percent, m_r_percent,
            lowbias_fit=True, verbose=0):
    '''
    Given data for a specific temp, taken for a sample of certain area,
    1) fit a linear model of J(Vb) to get an x-intercept V0,
        a y-intercept J0, and a slope
    2) fit the Simmons model of J(Vb) to get a sample thickness d (in nm), 
        a sample avg barrier height phibar (in eV), and an electron effective
        mass ratio m_r (unitless)

    Args:
    Experimental params:
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

    # read in the experimental data
    V_exp, I_exp = load_IVb(temp); # in volts nano amps

    #### fit expermental data

    # interpolate V0, I0, and slope guesses from exp data
    V0_not = V_exp[len(V_exp)//2];
    I0_not = I_exp[len(I_exp)//2];
    slope_not = (I_exp[-1] - I_exp[0])/(V_exp[-1] - V_exp[0]);
    
    # linear fit for I0
    init_params_linear = [I0_not, slope_not];
    bounds_linear = [[I0_not*0.01, slope_not*0.01],
                     [I0_not*100, slope_not*100]];
    params_linear, pcov_linear = scipy_curve_fit(I_of_Vb_linear, V_exp, I_exp,
                                p0=init_params_linear, bounds=bounds_linear);
    I0, slopeI = tuple(params_linear);
    I_fit_linear = I_of_Vb_linear(V_exp, I0, slopeI);

    # visualize linear fit
    if(verbose):
        print("I_of_Vb_linear fitting results:");
        print_str = "           I0 = {:6.4f} "+str((bounds_linear[0][0],bounds_linear[1][0]))+"nA\n\
        slope = {:6.4f} "+str((bounds_linear[0][1],bounds_linear[1][1]))+" nA/V"
        print(print_str.format(I0, slopeI));
    if(verbose > 4): plot_fit(V_exp, I_exp, I_fit_linear, mytitle = print_str[:print_str.find("(")].format(I0));
    del init_params_linear, bounds_linear, params_linear, pcov_linear;

    # cubic fit for V0
    init_params_cubic = [V0_not, slope_not];
    bounds_cubic = [[V_exp[0], slope_not*0.01],
                 [V_exp[-1], slope_not*100]];
    params_cubic, pcov_cubic = scipy_curve_fit(I_of_Vb_cubic, V_exp, I_exp-I0,
                                p0=init_params_cubic, bounds=bounds_cubic);
    V0, slopeI = tuple(params_cubic);
    I_fit_cubic = I_of_Vb_cubic(V_exp, V0, slopeI);

    # visualize cubic fit
    if(verbose):
        print("I_of_Vb_cubic fitting results:");
        print_str = "           V0 = {:6.4f} "+str((bounds_cubic[0][0],bounds_cubic[1][0]))+" V\n\
        slope = {:6.4f} "+str((bounds_cubic[0][1],bounds_cubic[1][1]))+" nA/V"
        print(print_str.format(V0, slopeI));
    if(verbose > 4): plot_fit(V_exp, I_exp, I_fit_cubic+I0, mytitle = print_str[:print_str.find("(")].format(V0));
    del init_params_cubic, bounds_cubic, params_cubic, pcov_cubic;

    # results of linear and cubic fits
    global V0_kwarg; V0_kwarg = V0; # very bad practice
    global I0_kwarg; I0_kwarg = I0;
    global convert_J2I_kwarg; convert_J2I_kwarg = J2I(temp, area);

    # low bias simmons fit at fixed d to narrow down phibar and m_r
    if lowbias_fit:
        init_params_lowbias = [d_not, phibar_not, m_r_not];
        bounds_lowbias = np.array([[d_not*0.999,phibar_not*(1-phibar_percent),m_r_not*(1-m_r_percent)],
                  [d_not*1.001,phibar_not*(1+phibar_percent),m_r_not*(1+m_r_percent)]]);
        params_lowbias, pcov_lowbias = scipy_curve_fit(I_of_Vb_lowbias, V_exp, I_exp,
                                p0 = init_params_lowbias, bounds = bounds_lowbias, max_nfev = 1e6);
        I_fit_lowbias = I_of_Vb_lowbias(V_exp, *params_lowbias);

        # visualize low bias fit
        if(verbose):
            print("I_of_Vb_lowbias fitting results:");
            print_str = "            d = {:6.4f} "+str((bounds_lowbias[0][0],bounds_lowbias[1][0]))+" nm\n\
            phi = {:6.4f} "+str((bounds_lowbias[0][1],bounds_lowbias[1][1]))+" eV\n\
            m_r = {:6.4f} "+str((bounds_lowbias[0][2],bounds_lowbias[1][2]));
            print(print_str.format(*params_lowbias));
        if(verbose > 4): plot_fit(V_exp, I_exp, I_fit_lowbias, mytitle = print_str[:print_str.find("(")].format(params_lowbias[0])+" nm");

        # update guesses and narrow bounds
        phibar_not, m_r_not = tuple(params_lowbias[1:]); 
        phibar_percent, m_r_percent = phibar_percent/10, m_r_percent/10;
        del init_params_lowbias, bounds_lowbias, params_lowbias, pcov_lowbias;
    
    # full simmons fit in narrowed bounds
    init_params = [d_not, phibar_not, phibar_not, m_r_not]; # phi1, phi2 now independent!
    bounds = np.array([[d_not*(1-d_percent),phibar_not*(1-phibar_percent),phibar_not*(1-phibar_percent),m_r_not*(1-m_r_percent)],
              [d_not*(1+d_percent),phibar_not*(1+phibar_percent),phibar_not*(1+phibar_percent),m_r_not*(1+m_r_percent)]]);
    params, pcov = scipy_curve_fit(I_of_Vb_asym, V_exp, I_exp,
                            p0 = init_params, bounds = bounds, max_nfev = 1e6);
    d, phi1, phi2, m_r = tuple(params);
    I_fit = I_of_Vb_asym(V_exp, d, phi1, phi2, m_r);
    rmse_final =  np.sqrt( np.mean( (I_exp-I_fit)*(I_exp-I_fit) ))/np.mean(I_exp);
    if(verbose):
        print("I_of_Vb_asym fitting results:");
        print_str = "            d = {:6.4f} "+str((bounds[0][0],bounds[1][0]))+" nm\n\
          ph1 = {:6.4f} "+str((bounds[0][1],bounds[1][1]))+" eV\n\
          ph2 = {:6.4f} "+str((bounds[0][2],bounds[1][2]))+" eV\n\
          m_r = {:6.4f} "+str((bounds[0][3],bounds[1][3]))+"\n\
          err = {:6.4f}";
        print(print_str.format(d, phi1, phi2, m_r, rmse_final));
    if(verbose > 4): plot_fit(V_exp, I_exp, I_fit, mytitle = print_str[:print_str.find("(")].format(d)+" nm");

    return (d, phi1, phi2, m_r, rmse_final);

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

def fit_data():

    # experimental params
    Ts = [40, 80, 120, 160, 200];
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # fitting param guesses
    phi_guess = np.array([1.85, 1.65, 1.60, 1.35, 1.35]);
    m_guess = np.array([0.32, 0.36, 0.36, 0.42, 0.41]);
    d_guess = np.array([5.01, 5.02, 5.03, 5.04, 5.05]);

    d_guess = 5.0*np.ones_like(d_guess)
    phi_guess = 1.7*np.ones_like(phi_guess)
    m_guess = 0.35*np.ones_like(m_guess)

    # fitting param bounds
    d_guess_percent = 0.5;
    phi_guess_percent = 0.5;
    m_guess_percent = 0.5;
    bounds = np.array([[d_guess[0]*(1-d_guess_percent),phi_guess[0]*(1-phi_guess_percent),phi_guess[0]*(1-phi_guess_percent),m_guess[0]*(1-m_guess_percent),0.0],
              [d_guess[0]*(1+d_guess_percent),phi_guess[0]*(1+phi_guess_percent),phi_guess[0]*(1+phi_guess_percent),m_guess[0]*(1+m_guess_percent),0.2]]);
    
    # fitting results
    combined = False;
    lowbias_fit = True;
    Ts = Ts[:];
    results = [];
    for datai in range(len(Ts)):
        print("\nT = {:.0f} K".format(Ts[datai]));
        if not combined: # normal way
            rlabels = ["$d$ (nm)", "$\phi_1$ (eV)", "$\phi_2$ (eV)",  "$m_r$", "RMSE"];
            results.append(fit_I(Ts[datai], area, d_guess[datai], phi_guess[datai], m_guess[datai],
                    d_guess_percent, phi_guess_percent, m_guess_percent,
                                 lowbias_fit=lowbias_fit, verbose=4));
            
        else: # d and sqrt(m_r) combined as in arxiv paper
            rlabels = ["$d\sqrt{m_r}$ (nm)", "$\phi_2-\phi_1$ (eV)", "RMSE"];
            results.append(fit_I_combined(Ts[datai], area, d_guess[datai]*np.sqrt(m_guess[datai]), 1.0,
                    d_guess_percent, phi_guess_percent, verbose=4));

    # plot fitting results vs T
    results = np.array(results);
    nresults = len(results[0]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    for resulti in range(nresults):
        axes[resulti].plot(Ts, results[:,resulti], color=mycolors[0],marker=mymarkers[0]);
        axes[resulti].set_ylabel(rlabels[resulti]);
        if not lowbias_fit: axes[resulti].axhline(bounds[0][resulti], color=accentcolors[0],linestyle='dashed');
        if not lowbias_fit: axes[resulti].axhline(bounds[1][resulti], color=accentcolors[0],linestyle='dashed');
    axes[-1].set_xlabel("$T$ (K)");
    plt.show();

###############################################################
#### exec
if __name__ == "__main__":

    fit_data();

