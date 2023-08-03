'''
Outline of procedure for fitting MnPc data with oscillations.py.
There are three classes of fitting parameters:
i) Physical background parameters: eps0, epsc, G1, G2, G3
ii) Experimental background parameters: V0, Tsurf
iii) Oscillation parameters: tau0, Gamma, EC, Tjunc
There are three levels of fitting, from most to least rigorous:
a) lorentz: the oscillation terms are computed from the numerical integral with
    full temperature dependence included (therefore slow). The code then auto-
    matically updates the parameters to minimize the error w/r/t single dataset
b) lorentz_zero: the oscillation terms are approximated by the analytical form
    of the integral at T=0 (therefore faster then lorentz). Parameters
    are again updated automatically to minimize the error w/r/t single dataset
c) by-hand freezing, the physical background parameters are not allowed to
    update, but are instead frozen to ensure physical reasonableness / good
    fitting behavior across datasets

The data is organized so that each data folder was taken on about the same
calendar day, and at the same B field. There are in general different data
sets within each data folder which differ by nominal temperature.

The entire fitting procedure consists of:
1) Applying (b) to (i)-(iii) for the LOWEST temperature dataset in the data
    folder, to find the best results for (i). Subsequently, (i) is
    treated at (c) for ALL data sets in the data folder.
2) Applying (b) to (ii)-(iii), dropping data that is not within a few standard
    devs of this fit (removing outliers)
3) Applying (b) to (ii)-(iii), to find the best results for (ii). Subsequently
    (ii) is treated at (c) within THIS dataset only
4) Applying (a) to (iii) to get final results
'''


