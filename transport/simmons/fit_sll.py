'''
Shuanglong's Simmons formula code, lightly edited
'''
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# conversion factors

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

def IV_per_T_linear(V, V0, I0, alpha):

    I = alpha * (V - V0) + I0

    return I

def IV_per_T(V, phi, m, d, check=False):


    V0 = T_dict[T][1]; # NB global

    r0 = 200               # micro meter

    d = d * nm2bohr

    r0 = r0 * microm2bohr
    r = (1 + tec * T) * r0
    A = np.pi * r**2

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

    I = zoom * ( au2si_current * A * J + T_dict[T][0] ) # NB global 

    #### checking ####
    if check:
        print("SLL CODE");
        start = len(V)//2;
        thru = 1;
        print("Vb = ",(au2si_voltage*V)[start:start+thru]);
        print(" I = ",(I/zoom)[start:start+thru]);
        print("c_1 = ", c1[start:start+thru], ", unitless");
        print("c_minus = ", (c_m/phi_m)[start:start+thru], ", unitless");
        print("c_plus = ", (c_p/phi_p)[start:start+thru], ", unitless");
        #assert False

    return I

def fit_IV(phi0=1, m0=1, d0=5.0):

    # Read in the data

    IV = np.loadtxt("{:.0f}".format(T) + "KExp.txt")
    V = IV[:, 0]
    I = zoom * IV[:, 1]

    # Fitting

    eps_phi = 0.1
    eps_m = 0.001
    eps_d = 1.0
    popt, pcov = curve_fit(IV_per_T, V, I, p0=[phi0, m0, d0], bounds=((phi0-eps_phi, m0-eps_m, d0-eps_d), (phi0+eps_phi, m0+eps_m, d0+eps_d)))

    # Output fitted parameters

    phi = popt[0]
    m   = popt[1]
    d   = popt[2]
    I_fitted = IV_per_T(V, phi, m, d, check=True)
    rmse = np.sqrt( np.mean( (I_fitted - I)**2 ) )/np.mean(I);
    print("IV_per_T fitting results:");
    print_str = "        d = {:6.4f} "+str((d0-eps_d,d0+eps_d))+" nm\n\
      phi = {:6.4f} "+str((phi0-eps_phi,phi0+eps_phi))+" eV\n\
        m = {:6.4f} "+str((m0-eps_m, m0+eps_m))+" me\n\
     rmse = {:6.4f} "
    print(print_str.format(d, phi, m, rmse))

    # Ohmic regime

    V_dense = np.linspace(-0.2, 0.1, 31, endpoint=True)
    I_fitted_dense = IV_per_T(V_dense, phi, m, d)

    V0_guess = 0
    I0_guess = 1
    alpha_guess = 1
    eps_V0 = 0.2
    eps_I0 = 0.5
    eps_alpha = 5
    popt, pcov = curve_fit(IV_per_T_linear, V_dense, I_fitted_dense, # NB fits the fitted data!
                        p0=[V0_guess, I0_guess, alpha_guess],
                           bounds=((V0_guess-eps_V0, I0_guess-eps_I0, alpha_guess-eps_alpha), (V0_guess+eps_V0, I0_guess+eps_I0, alpha_guess+eps_alpha)))
    print("IV_per_T_linear fitting results:");
    print("       V0 = {:6.4f} V\n\
       I0 = {:6.4f} A\n\
    slope = {:6.4f} * 10**-10 A/V".format(*tuple(popt)) )

    #### check my J function with fitted params
    ####
    from simmons_formula import I_of_Vb_asym as my_I_func
    from simmons_formula import J2I

    # experimental params
    my_area = (200*1e3)**2 * np.pi;
    # experimental shifts of I, V
    my_Ishift, my_Vshift = T_dict[T]; # if we do this my curve should agree perfectly

    # convert
    my_Is = my_I_func(V-my_Vshift, d, phi, m)+my_Jshift;
    my_Is = my_Js * J2I(T, my_area, thermal_exp = tec);
    ####
    #### end my addition

    # Save fitted I(V)

    V_dense = np.linspace(-1.2, 1.2, 240, endpoint=True)
    I_fitted_dense = IV_per_T(V_dense, phi, m, d)
    fname = "I_fitted_" + "{:.0f}".format(T) + ".dat"
    if False:
        with open(fname, "w") as f:
            nV = len(V)
            for i in range(nV):
                f.write("{:8.4f}  {:12.6f}\n".format(V[i], I[i]))
            f.write("\n\n")
            nV = len(V_dense)
            for i in range(nV):
                f.write("{:8.4f}  {:12.6f}\n".format(V_dense[i], I_fitted_dense[i]))
    
    # Plot the fitted curves
    
    fig = plt.figure()
    
    fig.set_size_inches(5.5, 4.5)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    
    plt.plot(V, I, 'bo', label="Exp")
    plt.plot(V, I_fitted, 'r-', label="Fitted values")
    plt.ylim([0.6, 1.4])
    #plt.ylim([-1, 4])

    # plot my results
    plt.plot(V, my_Is*zoom, 'g-', label = "My fit");

    plt.title("T = {:.0f} K, I0 = {:6.4f} A*1e-10, V0 = {:6.4f} V".format(T, T_dict[T][0]*zoom, T_dict[T][1]));
    plt.xlabel("V (V)")
    plt.ylabel("I (10**-10 A)")
    plt.legend()
    
    fname = "I_fitted_" + "{:.0f}".format(T) + ".pdf"
    plt.show();
    #plt.savefig(fname)

if __name__ == "__main__":

    # V0s and I0s at a given T
    T_dict = {40: (1.00/zoom, -0.0308), 80: (1.00/zoom, -0.0161), 120: (1.00/zoom, -0.0443),
             160: (1.00/zoom, -0.0351), 200: (1.00/zoom, -0.0661)}

    # guesses
    phi0s = np.array([1.85, 1.65, 1.60, 1.35, 1.35]);
    m0s = np.array([0.32, 0.36, 0.36, 0.42, 0.41]);
    d0s = np.array([5.01, 5.02, 5.03, 5.04, 5.05]);
    d0_shifts = np.array([-0.1,-0.12,-0.1,+0.1,-0.13]);
    d0s = d0s+d0_shifts;

    T=40;  print("\nT = {:.0f} K".format(T)); fit_IV(phi0=phi0s[0], m0=m0s[0], d0=d0s[0])
    #T=80;  print("\nT = {:.0f} K".format(T)); fit_IV(phi0=phi0s[0], m0=m0s[1], d0=d0s[1])
    #T=120; print("\nT = {:.0f} K".format(T)); fit_IV(phi0=phi0s[0], m0=m0s[2], d0=d0s[2])
    #T=160; print("\nT = {:.0f} K".format(T)); fit_IV(phi0=phi0s[0], m0=m0s[3], d0=d0s[3])
    #T=200; print("\nT = {:.0f} K".format(T)); fit_IV(phi0=phi0s[0], m0=m0s[4], d0=d0s[4])

