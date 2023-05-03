'''
My edits to Shuanglong's Simmons formula code
'''
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

###############################################################
#### current and current density functions

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
    return J_right - J_left;

def I_of_T(Js, temp, area, thermal_exp = 0.5*1e-4):
    '''
    Given a Simmons formula for current density J(Vb) in amps/nm^2
    gets the current in amps at a given temperature
    '''
    if( len(np.shape(Js)) != 1): raise TypeError;

    # temperature affects on area
    area_temp = area*( 1+thermal_exp*temp)**2;
    return Js*area_temp;

###############################################################
#### fitting functions

def fit_IV(phi0, m0, d0):

    ### Read in the data

    IV = np.loadtxt("{:.0f}".format(T) + "KExp.txt")
    V = IV[:, 0]
    I = zoom * IV[:, 1]

    ### Fitting

    eps_phi = 0.1
    eps_m = 0.001
    eps_d = 0.001
    popt, pcov = curve_fit(IV_per_T, V, I, p0=[phi0, m0, d0], bounds=((phi0-eps_phi, m0-eps_m, d0-eps_d), (phi0+eps_phi, m0+eps_m, d0+eps_d)))

    ### Output fitted parameters

    phi = popt[0]
    m   = popt[1]
    d   = popt[2]
    I_fitted = IV_per_T(V, phi, m, d)
    rmse = np.sqrt( np.mean( (I_fitted - I)**2 ) )
    print("      d = {:6.2f} nm\n\
    phi = {:6.2f} eV\n\
      m = {:6.2f} me\n\
   rmse = {:6.3f} * 10**-10 A".format(d, phi, m, rmse))

    ### Ohmic regime

    V_dense = np.linspace(-0.2, 0.1, 31, endpoint=True)
    I_fitted = IV_per_T(V_dense, phi, m, d)

    V0 = 0
    I0 = 1
    alpha0 = 1
    eps_V0 = 0.2
    eps_I0 = 0.5
    eps_alpha = 5
    popt, pcov = curve_fit(IV_per_T_linear, V_dense, I_fitted, p0=[V0, I0, alpha0], bounds=((V0-eps_V0, I0-eps_I0, alpha0-eps_alpha), (V0+eps_V0, I0+eps_I0, alpha0+eps_alpha)))
    print("  slope = {:6.3f} * 10**-10 A/V\n".format(popt[2]))

    ### Save fitted I(V)

    V_dense = np.linspace(-1.2, 1.2, 240, endpoint=True)
    I_fitted = IV_per_T(V_dense, phi, m, d)
    fname = "I_fitted_" + "{:.0f}".format(T) + ".dat"
    with open(fname, "w") as f:
        nV = len(V)
        for i in range(nV):
            f.write("{:8.4f}  {:12.6f}\n".format(V[i], I[i]))
        f.write("\n\n")
        nV = len(V_dense)
        for i in range(nV):
            f.write("{:8.4f}  {:12.6f}\n".format(V_dense[i], I_fitted[i]))
    
    ### Plot the fitted curves
    
    fig = plt.figure()
    
    fig.set_size_inches(5.5, 4.5)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    
    plt.plot(V, I, 'bo', label="Exp")
    plt.plot(V_dense, I_fitted, 'r-', label="Fitted values")
    plt.ylim([0.6, 1.4])
    #plt.ylim([-1, 4])
    plt.legend()

    plt.xlabel("V (V)")
    plt.ylabel("I (10**-10 A)")
    
    fname = "I_fitted_" + "{:.0f}".format(T) + ".pdf"
    plt.savefig(fname)

###############################################################
#### wrappers

def plot_data():

    # experimental params
    Vmax = 1.0;
    Vbs = np.linspace(-Vmax,Vmax,11);
    temp = 0.0;
    radius = 200*1e3; # 200 micro meter
    area = np.pi*radius*radius;

    # fitting params
    d0 = 4.91; # nm
    phi0 = 1.95; # eV
    m0 = 0.32; # m_*/m_e

    # current density
    Js = J_of_Vb_asym(Vbs, d0, phi0, phi0, m0);
    #print(Js); # should be of order 1e-22
    #assert False

    # current
    Is = I_of_T(Js, temp, area);

    # plot
    fig, ax = plt.subplots();
    ax.plot(Vbs, Is);
    plt.show();
    

def fit_data():

    # experimental params
    Ts = [40, 80, 120, 160, 200];

    # fitting param guesses
    phis = [1.85, 1.65, 1.60, 1.35, 1.35];
    ms = [0.32, 0.36, 0.36, 0.42, 0.41];
    ds = [5.01, 5.02, 5.03, 5.04, 5.05];

    for datai in range(1):
        print("T = {:.0f} K".format(Ts[datai]));
        fit_IV(phi0=phis[datai], m0=ms[datai], d0=ds[datai]-0.1);

###############################################################
#### exec
if __name__ == "__main__":

    plot_data();

