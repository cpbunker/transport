import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

ms = [0.32, 0.36, 0.36, 0.42, 0.41]
ds = [5.01, 5.02, 5.03, 5.04, 5.05]

def IV_per_T_linear(V, V0, I0, alpha):

    I = alpha * (V - V0) + I0

    return I

def IV_per_T(V, phi, m, d):

    if T == 40:
        V0 = -0.0308
    elif T == 80:
        V0 = -0.0161
    elif T == 120:
        V0 = -0.0443
    elif T == 160:
        V0 = -0.0351
    elif T == 200:
        V0 = -0.0661
    else:
        pass

    r0 = 200               # micro meter

    d = d * nm2bohr

    r0 = r0 * microm2bohr
    r = (1 + tec * T) * r0
    A = np.pi * r**2
    #print(A)

    V = V - V0

    c1 = 1 + (e*V/phi)**2/48 # assumes phi_1 = phi_2 = phi. Also NB phi 1 and phi_2 are separate fitting parameters

    V = V/au2si_voltage

    phi = phi / har2eV
    phi_p = phi + e*V/2
    phi_m = phi - e*V/2

    S = 4*np.pi*np.sqrt(2*m)/h
    S_prime = S / np.sqrt(c1)
    c_p = phi_p + 3*np.sqrt(phi_p)/(S_prime * d) + 3/(S_prime * d)**2
    c_m = phi_m + 3*np.sqrt(phi_m)/(S_prime * d) + 3/(S_prime * d)**2

    J = e/(2*np.pi*h*d**2) * c1 * \
        ( c_m * np.exp(-S_prime * d * np.sqrt(phi_m)) - \
          c_p * np.exp(-S_prime * d * np.sqrt(phi_p)) )

    I = zoom * ( au2si_current * A * J + 10**(-10) )

    return I

def fit_IV(phi0=1, m0=1, d0=5.0):

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
    plt.show();
    #plt.savefig(fname)

if __name__ == "__main__":

    T=40;  print("T = {:.0f} K".format(T)); fit_IV(phi0=1.85, m0=ms[0], d0=ds[0]-0.1)
    #T=80;  print("T = {:.0f} K".format(T)); fit_IV(phi0=1.65, m0=ms[1], d0=ds[1]-0.12)
    #T=120; print("T = {:.0f} K".format(T)); fit_IV(phi0=1.60, m0=ms[2], d0=ds[2]-0.1)
    #T=160; print("T = {:.0f} K".format(T)); fit_IV(phi0=1.35, m0=ms[3], d0=ds[3]+0.1)
    #T=200; print("T = {:.0f} K".format(T)); fit_IV(phi0=1.35, m0=ms[4], d0=ds[4]-0.13)

