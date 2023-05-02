'''
My edits to Shuanglong's Simmons formula code
'''
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

'''
e = 1
me = 1
hbar = 1
h = 2*np.pi*hbar

au2si_energy = 4.35975*10**(-18)
au2si_length = 5.29177*10**(-11)
au2si_charge = 1.602188*10**(-19)
au2si_time = 2.41888*10**(-17)
au2si_current = au2si_charge/au2si_time
au2si_current_density = au2si_current/au2si_length**2
au2si_voltage = au2si_energy/au2si_charge

har2eV=27.211396641308
nm2bohr = 18.897161646321
mm2bohr = 10**6 * nm2bohr
microm2bohr = 10**3 * nm2bohr

tec = 0.5 * 10**(-4)   # Thermal expansion coefficient
zoom = 10**10          # factor for magnifying the current
'''

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

    # decay length
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    d_d = d_d_prefactor/np.sqrt(m_r*phibar); # decay length, units nm

    # current
    J_prefactor = 9.685*1e-6 # =e^2/(8*\pi*\hbar), units amp/volt
    return J_prefactor * Vb/(d*d_d) * np.exp(-d/d_d);

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

    # beta
    beta = 1.0; # unitless, depends on the specifics of \phi(x), see Simmons Eq A6

    # decay length
    d_d_prefactor = 0.09766 # =\hbar/sqrt(8*me), units nm*eV^1/2
    d_d = d_d_prefactor/np.sqrt(m_r*phibar); # decay length, units nm

    # current
    J_prefactor = 9.685*1e-6 # =e^2/(8*\pi*\hbar), units amp/volt
    return J_prefactor * Vb/(d*d_d) * np.exp(-d/d_d);
    

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

if __name__ == "__main__":

    # data at different temps
    Ts = [40, 80, 120, 160, 200];

    # fitting param guesses
    phis = [1.85, 1.65, 1.60, 1.35, 1.35];
    ms = [0.32, 0.36, 0.36, 0.42, 0.41];
    ds = [5.01, 5.02, 5.03, 5.04, 5.05];

    for datai in range(1):
        print("T = {:.0f} K".format(Ts[datai]));
        fit_IV(phi0=phis[datai], m0=ms[datai], d0=ds[datai]-0.1)

