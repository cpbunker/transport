'''
My edits to Shuanglong's Simmons formula code
'''
import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit
import matplotlib.pyplot as plt

###############################################################
#### current and current density functions

def J_of_Vb_linear(Vb, J0, slope):
    '''
    Get the current density J as function of applied bias Vb
    for a simple linear model

    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    J0, y-intercept
    slope, slope

    Returns:
    J, current density, units amps/(nm^2)
    '''

    return slope*Vb + J0;

def J_of_Vb_cubic(Vb, V0, slope):
    '''
    Get the current density J as function of applied bias Vb
    for a simple cubic model

    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    V0, x intercept

    Returns:
    J, current density, units amps/(nm^2)
    '''

    return slope*np.power(Vb-V0,3)

def J_of_Vb_lowbias(Vb, d, phibar, m_r):
    '''
    Get the current density J as function of applied bias Vb
    from Simmons Eq 24, ie the low bias (ohmic) limit

    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
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

    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    d, barrier width, units nm
    phibar, avg barrier height, units eV
    m_r, ratio of eff electron mass to me, unitless
    
    Returns:
    J, current density, units amps/(nm^2)
    '''
    if( len(np.shape(Vb)) != 1): raise TypeError;
    if(phibar < max(abs(Vb))): raise ValueError;

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
    D_asym = (phi2-Vb-phi1)**2 /(48*phibar*phibar); # asymmetry parameter, unitless

    # S factors
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    S_sym = np.sqrt(m_r)/d_d_prefactor; # units 1/(nm*eV^1/2)
    S_asym = S_sym/np.sqrt(1+D_asym); # units 1/(nm*eV^1/2)
    # NB for m_r = 1, S_sym = 10.25
    # m_r = 1; S_sym = np.sqrt(m_r)/d_d_prefactor; print(S_sym); assert False;

    # asymmetric decay lengths, units nm
    d_d_minus = d_d_prefactor*np.sqrt(1+D_asym)/np.sqrt(m_r*(phibar-Vb/2)); # decay length for lowered side
    d_d_plus = d_d_prefactor*np.sqrt(1+D_asym)/np.sqrt(m_r*(phibar+Vb/2)); # decay length for lowered side
    
    # current density
    J_right = J_0*(1+D_asym)*(phibar-Vb/2)*(1+3*d_d_minus/d + 3*(d_d_minus/d)**2);
    J_right *= np.exp(-d/d_d_minus);
    J_left =  J_0*(1+D_asym)*(phibar+Vb/2)*(1+3*d_d_plus/d + 3*(d_d_plus/d)**2);
    J_left *= np.exp(-d/d_d_plus);

    #### checking ####
    if False:
        print("MY CODE");
        start = len(Vb)//2;
        thru = 1;
        print("Vb = ",Vb[start:start+thru]);
        my_radius = 200*1e3; my_tec = 0.5*1e-4;
        my_area = (my_radius*(1+my_tec*40))**2 * np.pi;
        print(" I = ",my_area*(J_right-J_left)[start:start+thru]);
        print("c_1 = ", (1+D_asym)[start:start+thru], ", unitless");
        print("c_minus = ", (1+3*d_d_minus/d + 3*(d_d_minus/d)**2)[start:start+thru], ", unitless");
        print("c_plus = ", (1+3*d_d_plus/d + 3*(d_d_plus/d)**2)[start:start+thru], ", unitless");
        #assert False
        
    return J_right - J_left;

def J_of_Vb_asym_wrapped(Vb, d, phibar, m_r):
    '''
    Wraps J_of_Vb_asym so that
    - instead of phi1, phi2 separate fittable params, there is just phibar
    - Vbs are scaled to
    - Js are scaled to
    
    NB this function is designed to be passed to scipy.optimize.curve_fit
    Independent variable:
    Vb, applied bias voltage, units volts
    Fitting params:
    d, barrier width, units nm
    phibar, avg barrier height, units eV
    m_r, ratio of eff electron mass to me, unitless
    '''

    return J0_kwarg+J_of_Vb_asym(Vb-V0_kwarg, d, phibar, phibar, m_r) + 1e-10/((200e3)**2 *np.pi);

def J2I(temp, area, thermal_exp = 0.5*1e-4):
    '''
    Gets the conversion factor from current density J(Vb) in amps/nm^2
    to current in amps at a given temperature
    '''

    return area*( 1+thermal_exp*temp)**2;

###############################################################
#### fitting functions

def load_IVb(temp):
    '''
    Get I vs V data at a certain temp
    '''

    IV = np.loadtxt("{:.0f}".format(temp) + "KExp.txt");
    Vs = IV[:, 0];
    Is = IV[:, 1];
    if( len(np.shape(Vs)) != 1 or np.shape(Vs) != np.shape(Is) ): raise TypeError;

    return Vs, Is;

def fit_JVb(temp, area, d_not, phibar_not, m_r_not,verbose=0):
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
    V_exp, I_exp = load_IVb(temp);

    # convert from Is to Js
    convert_J2I = J2I(temp, area);
    J_exp = I_exp/convert_J2I;
    nano = 1e9;
    #print(J_exp); assert False; # should be of order 1e-22

    #### fit expermental data

    # interpolate V0, I0, and slope guesses from exp data
    V0_not = V_exp[len(V_exp)//2];
    I0_not = I_exp[len(V_exp)//2];
    J0_not = I0_not/convert_J2I;
    slope_not = (J_exp[1+len(V_exp)//2] - J_exp[len(V_exp)//2])/(V_exp[1+len(V_exp)//2] - V_exp[len(V_exp)//2]);
    init_params_linear = [J0_not, slope_not];
    bounds_linear = [[J0_not*0.5, slope_not*0.5],
                     [J0_not*1.5, slope_not*1.5]];
    init_params_cubic = [V0_not, slope_not];
    bounds_cubic = [[V0_not*0.5, slope_not*0.5],
                 [V0_not*1.5, slope_not*1.5]];

    # linear fit gives I0
    params_linear, pcov_linear = scipy_curve_fit(J_of_Vb_linear, V_exp, J_exp,
                                sigma = 0.001*np.copy(J_exp), p0=init_params_linear, bounds=bounds_linear);
    J0, slopeJ = tuple(params_linear);
    J_fit_linear = J_of_Vb_linear(V_exp, J0, slopeJ);
    if(verbose):
        print("J_of_Vb_linear fitting results:");
        print("           I0 = {:6.4f} nA\n\
        slope = {:6.4f} nA/V".format(J0*convert_J2I*nano, slopeJ*convert_J2I*nano));
    if(verbose > 4): plot_fit(V_exp, I_exp*nano, J_fit_linear*convert_J2I*nano,
                              mytitle = "I0 = {:1.4f} nA, Slope = {:1.4f} nA/V".format(J0*convert_J2I*nano, slopeJ*convert_J2I*nano ));

    # cubic fit gives V0
    params_cubic, pcov_cubic = scipy_curve_fit(J_of_Vb_cubic, V_exp, J_exp-J0,
                                sigma = 0.001*np.copy(J_exp-J0), p0=init_params_cubic, bounds=bounds_cubic);
    V0, slopeJ = tuple(params_cubic);
    J_fit_cubic = J_of_Vb_cubic(V_exp, V0, slopeJ);
    if(verbose):
        print("J_of_Vb_cubic fitting results:");
        print("           V0 = {:6.4f} V\n\
        slope = {:6.4f} nA/V".format(V0, slopeJ*convert_J2I*nano));
    if(verbose > 4): plot_fit(V_exp, I_exp*nano, (J_fit_cubic+J0)*convert_J2I*nano,
                              mytitle = "V0 = {:1.4f} V, I0 = {:1.4f} nA, Slope = {:1.4f} nA/V".format(V0, J0*convert_J2I*nano, slopeJ*convert_J2I*nano ));
    assert False

    # simmons fit
    init_params = [d_not, phibar_not, m_r_not];
    d_percent = 0.5;
    phibar_percent = 0.2;
    m_r_percent = 0.5;
    bounds = np.array([[d_not*(1-d_percent),phibar_not*(1-phibar_percent),m_r_not*(1-m_r_percent)],
              [d_not*(1+d_percent),phibar_not*(1+phibar_percent),m_r_not*(1+m_r_percent)]]);
    params, pcov = scipy_curve_fit(J_of_Vb_asym_wrapped, V_exp, J_exp,
                            sigma = 0.001*np.copy(J_exp), p0 = init_params, bounds = bounds);
    d, phibar, m_r = tuple(params);
    J_fit = J_of_Vb_asym_wrapped(V_exp, d, phibar, m_r, V0_kwarg = V0, J0_kwarg = J0);
    if(verbose):
        print("J_of_Vb_asym_wrapped fitting results:");
        print_str = "            d = {:6.4f} "+str((bounds[0][0],bounds[1][0]))+" nm\n\
          phi = {:6.4f} "+str((bounds[0][1],bounds[1][1]))+" eV\n\
          m_r = {:6.4f} "+str((bounds[0][2],bounds[1][2]));
        print(print_str.format(d, phibar, m_r));
    if(verbose > 4): plot_fit(V_exp, I_exp*nano, J_fit*convert_J2I*nano,
                              mytitle = "d = {:1.2f} nm, phi = {:1.2f} eV, m_r = {:1.2f} ".format(d, phibar, m_r));

    # fine meshed simmons fit
    #my_curve_fit(J_of_Vb_asym_wrapped, V_exp, J_exp, init_params, bounds, 1, verbose=verbose);

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
        fparams, pcov = scipy_curve_fit(fx, xvals, fxvals, sigma = 0.001*np.copy(fxvals), method = 'dogbox',
                                 p0 = init_fparams, bounds = fbounds);
        fxvals_fit = fx(xvals, *fparams);
        ferror = np.sqrt( np.mean( (fxvals-fxvals_fit)*(fxvals-fxvals_fit) ))/np.mean(fxvals);

        # update error and optimum
        focus_opts[fvali] = fparams[focus_i];
        focus_errors[fvali] = ferror;

        # visualize
        if(verbose):
            print("my_curve_fit fitting results, focus_i = ",focus_i);
            print_str = "            d = {:6.4f} "+str((fbounds[0][0],fbounds[1][0]))+" nm\n\
            phi = {:6.4f} "+str((fbounds[0][1],fbounds[1][1]))+" eV\n\
            m_r = {:6.4f} "+str((fbounds[0][2],fbounds[1][2]))+"\n\
            err = {:6.4f}";
            print(print_str.format(*fparams, ferror));
        if(verbose > 4 and True): plot_fit(xvals, fxvals*convert_J2I*nano, fxvals_fit*convert_J2I*nano,
                                  mytitle = "d = {:1.2f} nm, phi = {:1.2f} eV, m_r = {:1.2f} ".format(*fparams));

    # show the results of the mesh search
    fig, ax = plt.subplots();
    ax.plot(focus_opts, focus_errors, color=accentcolors[0]);

    # format
    ax.set_xlabel("Params["+str(focus_i)+"]");
    ax.set_ylabel("Norm. RMS Error");
    ax.set_yscale('log');
    plt.show();

###############################################################
#### plots

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

    for datai in range(1):
        print("\nT = {:.0f} K".format(Ts[datai]));
        fit_JVb(Ts[datai], area, d_guess[datai], phi_guess[datai], m_guess[datai], verbose=10);

###############################################################
#### exec
if __name__ == "__main__":

    #SLL_plot();
    fit_data();

