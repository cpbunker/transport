'''
Investigate finite size effects in spin polarized td-DMRG calculations
'''

import numpy as np
import matplotlib.pyplot as plt

minJtimes = []; # get data for time of min J vs lead length
crosstimes = []; # get data for time current crosses 0 vs lead length
llengths = [2,3,6];
datafs = []
for l in llengths:
    datafs.append("dat/DotDataDMRG/spinpol/"+str(l)+"_1_"+str(l)+"_e"+str(2*l+1)+"_B5.0_t0.0_Vg-0.5.npy");
    
for idat in range(len( datafs)): # get data from each run

    # unpack
    dataf = datafs[idat];
    print("Loading data from ",dataf);
    observables = np.load(dataf);
    t, E, Jup, Jdown, occL, occD, occR, SzL, SzD, SzR = tuple(observables); # scatter
    J = Jup + Jdown;
    
    # get min J time
    minJtimes.append(t[np.argmin(J)]);
    
    # extend last data by linear approx
    if (llengths[idat] == 6):
    	coefs = np.polyfit(t[-25:], J[-25:], 1)
    	t_ext = np.linspace(t[-1], t[-1]*3, 2*len(t) )
    	t = np.append(t, t_ext); # update t array
    	J = np.append(J, coefs[1]+coefs[0]*t_ext); # add on linear J vals
    
    # get time J crosses back over 0
    for i in range(1,len(t)): # iter over times:
    	if (J[i-1] < 0 and J[i] > 0): # this is the crossover
    		crosstimes.append(t[i]);
    		
    		
# plot results
plt.plot(llengths, minJtimes, label = "J peak");
plt.plot(llengths, crosstimes, label = "J crossover");
plt.xlabel("Lead length");
plt.ylabel("time");
plt.legend();
plt.show();

