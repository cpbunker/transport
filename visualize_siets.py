'''
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import sys
import json

# top level
case = int(sys.argv[1]);
time0 = int(sys.argv[2]);
Nupdates = int(sys.argv[3]);
datafiles = sys.argv[4:];

# plotting
obs1, color1, ticks1, linewidth1, fontsize1 = "occ_", "cornflowerblue", (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
obs2, color2 = "G_", "darkred";
obs3, color3 = "Sdz_", "darkgreen";
num_xticks = 4;
datamarkers = ["s","^","d","*"];

if(case in [0]): # standard charge density vs site snapshot
    from transport import tddmrg
    from transport.tddmrg import plot
    sys_type = json.load(open(datafiles[0]+".txt"))["sys_type"];
    tddmrg.plot.snapshot_fromdata(datafiles[0], time0, sys_type);

elif(case in [1,2]): # observable as a function of time

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    #ax.set_yticks(ticks1);
    ax.set_xlabel("Time $(\hbar/t_l)$");

    # plot observables for EACH datafile
    for datai in range(len(datafiles)):
        params = json.load(open(datafiles[datai]+".txt"));
        title_or_label = open(datafiles[datai]+"_arrays/"+obs1+"title.txt","r").read().splitlines()[0][1:];

        # time evolution params
        tupdate = params["tupdate"];
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = time0 + ti*tupdate;

        # total number of electrons
        NL, NR = params["NL"], params["NR"];
        if(params["sys_type"] == "SIAM"): NFM = 1;
        elif(params["sys_type"] == "SIETS"): NFM = params["NFM"];
        Nsites = NL+NFM+NR;
        totalNe = np.sum(np.load(datafiles[datai]+"_arrays/occ_yjs_time{:.2f}.npy".format(times[-1])));
        NsiteNestring = "$N_{sites}=$"+"{:.0f}, $N_e =${:.0f}".format(Nsites,totalNe); 
        if(len(datafiles)==1):
            the_title = title_or_label[:]; the_label = "";
            print(NsiteNestring)
        else:
            the_label = title_or_label[:];
            the_title = NsiteNestring[:];

        # current vs time
        which_imp = 0;
        yds_vs_time = np.zeros((len(times),NFM+1),dtype=float);
        for ti in range(len(times)):
            dummy_current = np.load(datafiles[datai]+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
            print(np.shape(dummy_current));
            yds_vs_time[ti] = dummy_current[:];
            
        # plot averaged or separated current
        if(case in [1]): # averaged
            current_vs_time = (yds_vs_time[:,0]+yds_vs_time[:,-1])/2;
            ax.plot(times,current_vs_time,color=color2, marker=datamarkers[datai],label=the_label);
        elif(case in [2]):
            ax.plot(times,yds_vs_time[:,0], color=color2, marker=datamarkers[datai], label=the_label);
            ax.plot(times,yds_vs_time[:,-1], color=color2, linestyle="dashed",  marker=datamarkers[datai], label=the_label);
        
        # Sdz vs time
        if(params["sys_type"] == "SIETS"):
            Sdzs_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                Sdzs_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
            ax.plot(times,Sdzs_vs_time[:,which_imp],color=color3, marker=datamarkers[datai]);


    # formatting
    ax.set_ylabel("$\langle G_{SR}\\rangle /G_0$", color=color2, fontsize=fontsize1);
    ax.set_title(the_title);
    if(len(times) > num_xticks): 
        time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1))
        ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));
    if(params["sys_type"] == "SIETS"):
        ax3 = ax.twinx();
        ax3.yaxis.set_label_position("left");
        ax3.spines.left.set_position(("axes", -0.2));
        ax3.spines.left.set(alpha=0.0);
        ax3.set_yticks([])
        ax3.set_ylabel("$\langle S_{d}^{z} \\rangle$", color=color3, fontsize=fontsize1);
    
    # show
    if(len(datafiles) > 1): ax.legend();
    plt.tight_layout();
    plt.show();

elif(case in [3,4]): # left lead, SR, right lead occupancy as a function of time
    if(case in [4]): difference=True;
    else: difference = False;

    # axes
    fig, axes = plt.subplots(2, sharex=True);
    axes[-1].set_xlabel("Time $(\hbar/t_l)$");

    # plot observables for EACH datafile
    for datai in range(len(datafiles)):
        params = json.load(open(datafiles[datai]+".txt"));
        title_or_label = open(datafiles[datai]+"_arrays/"+obs1+"title.txt","r").read().splitlines()[0][1:];
        if(len(datafiles)==1): the_title = title_or_label[:]; the_label = "";
        else: the_title = ""; the_label = title_or_label[:];

        # time evolution params
        tupdate = params["tupdate"];
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = time0 + ti*tupdate;
            
        # occ vs time
        NL, NFM, NR = params["NL"], params["NFM"], params["NR"];
        Nsites = NL+NFM+NR; 
        yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));

        # break up occupancies
        yjL_vs_time = np.sum(yjs_vs_time[:,:NL], axis=1);
        yjSR_vs_time = np.sum(yjs_vs_time[:,NL:NL+NFM], axis=1);
        yjR_vs_time = np.sum(yjs_vs_time[:,NL+NFM:], axis=1);
        if(difference): # only plot change in occupancy
            print("LL n(0) = {:.4f}".format(yjL_vs_time[0]));
            yjL_vs_time = yjL_vs_time - yjL_vs_time[0];
            print("SR n(0) = {:.4f}".format(yjSR_vs_time[0]));
            yjSR_vs_time = yjSR_vs_time - yjSR_vs_time[0];
            print("RL n(0) = {:.4f}".format(yjR_vs_time[0]));
            yjR_vs_time = yjR_vs_time - yjR_vs_time[0];
            axes[0].set_ylim(0.0,0.01);
        
        # plot occupancies
        #axes[0].plot(times, yjL_vs_time,color=color1,marker=datamarkers[datai]);
        axes[0].plot(times, yjSR_vs_time,color=color2,marker=datamarkers[datai],label=the_label);
        #axes[0].plot(times, yjR_vs_time,color=color3,marker=datamarkers[datai]);
        
        # plot discrete time deriv of SR occ
        dt = params["tupdate"]
        ddt_SR = np.gradient(yjSR_vs_time, dt)
        axes[1].plot(times, -ddt_SR, color=color2, marker=datamarkers[datai], label="$-\\frac{n_{SR}(t+dt) - n_{SR}(t-dt)}{2dt}$"+" (dt={:.1f})".format(dt));
        
         # load current vs time data
        yds_vs_time = np.zeros((len(times),NFM+1),dtype=float);
        for ti in range(len(times)):
            dummy_current = np.load(datafiles[datai]+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
            print(np.shape(dummy_current));
            yds_vs_time[ti] = dummy_current[:];
            
        # plot discrete spatial deriv of current
        del_current = (params["Vb"]/np.pi)*(yds_vs_time[:,-1]-yds_vs_time[:,0]); # convert units from conductance to current
        axes[1].plot(times,del_current, color=color3, marker=datamarkers[datai], label="$\\frac{J(j+1) - J(j)}{a}$");
             
    # formatting
    if(difference): axes[0].set_ylabel("$\Delta n_{SR}(t)$", color=color2, fontsize=fontsize1);
    else: axes[0].set_ylabel("$n_{SR}(t)$", color=color2, fontsize=fontsize1);
    axes[0].set_title(the_title);
    if(len(times)>num_xticks):
        time_ticks = np.arange(times[0], times[-1], times[-1]//max(1,num_xticks-1))
        axes[-1].set_xticks(time_ticks);
    axes[-1].set_xlim((times[0], times[-1]));

    # show
    axes[1].legend();
    plt.tight_layout();
    plt.show();

else: raise Exception("case = {:.0f} not supported".format(case));
