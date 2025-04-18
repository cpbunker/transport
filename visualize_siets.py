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
obs1, color1, ticks1, linewidth1, fontsize1 = "occ_", "cornflowerblue", (-1.0,0.0,1.0), 3.0, 16;
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
        if(params["sys_type"] in ["SIAM"]): NFM = 1;
        elif(params["sys_type"] in ["SIAM_RM","SIETS","SIETS_RM"]): NFM = params["NFM"];
        else: raise NotImplementedError;
        if(params["sys_type"] in ["SIAM_RM", "SIETS_RM"]): block2site = 2;
        else: block2site = 1;
        Ntotal = (params["NL"]+params["NR"]+NFM);
        totalNe = np.sum(np.load(datafiles[datai]+"_arrays/occ_yjs_time{:.2f}.npy".format(times[-1])));
        NsiteNestring = "$N_{sites}=$"+"{:.0f}, $N_e =${:.0f}".format(Ntotal*block2site,totalNe); 
        if(params["sys_type"] in ["SIAM_RM","SIETS_RM"]):
            Egap = 2*abs(params["w"]-params["v"]*np.sign(params["w"]/params["v"]));
            Egap_str = ", $E_{gap} =$"+"{:.2f}".format(Egap);
            assert("u" not in params.keys());
            #NsiteNestring += Egap_str;
            title_or_label = "$t_h =${:.2f}, $V_b =${:.2f}, $v =${:.2f}, $w =${:.2f}".format(params["th"],params["Vb"],params["v"],params["w"]);
            title_or_label += Egap_str;
        if(len(datafiles)==1):
            the_title = title_or_label[:]; the_label = "";
            print(NsiteNestring)
        else:
            the_label = title_or_label[:];
            the_title = NsiteNestring[:];

        # conductance vs time
        # (normalized by G0 = e^2/2\hbar, saturates at 1)
        label2 = "$\langle G_j \\rangle /G_0$"
        print(obs2,"->",label2);
        yds_vs_time = np.zeros((len(times),(NFM+1)*block2site),dtype=float);
        xds_vs_time = np.zeros((len(times),(NFM+1)*block2site),dtype=float);
        for ti in range(len(times)):
            dummy_G = np.load(datafiles[datai]+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
            if(ti==0): print("shape of G_ = ",np.shape(dummy_G));
            yds_vs_time[ti] = dummy_G[:];
            xds_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(times[ti]));
            
        # plot conductance at various interfaces
        # pay attention to which interface as it varies for monatomic/diatomic (see xds_vs_time)
        if(case in [1]): # conductance thru SR|RL interface (at same place for monatomic, diatomic blocks)
            # do not average !!
            xds_plotted = [xds_vs_time[0,NFM*block2site]]; 
            ax.plot(times,yds_vs_time[:,NFM*block2site],color=color2, linestyle="dashed",marker=datamarkers[datai],label=the_label);
        elif(case in [2]): # conductance thru LL | SR interface, and SR | RL interface
            xds_plotted = [xds_vs_time[0,0], xds_vs_time[0,NFM*block2site]];
            ax.plot(times,yds_vs_time[:,0], color=color2, marker=datamarkers[datai]);
            ax.plot(times,yds_vs_time[:,NFM*block2site], color=color2, linestyle="dashed",  marker=datamarkers[datai], label=the_label);
        for sitei in xds_plotted: print("G_ at site {:.0f}".format(sitei));
        
        # Sdz vs time
        if(params["sys_type"] == "SIETS"):
            label3 = "$\langle S_{d}^{z} \\rangle$";
            print(obs3,"->",label3);
            Sdzs_vs_time = np.zeros((len(times),NFM*block2site),dtype=float);
            for ti in range(len(times)):
                Sdzs_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
            ax.plot(times,Sdzs_vs_time[:,which_imp],color=color3, marker=datamarkers[datai]);


        # formatting
        ax.set_ylabel(label2, color=color2, fontsize=fontsize1);
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
            ax3.set_ylabel(label3, color=color3, fontsize=fontsize1);

        # end loop over datafiles
        del params, Ntotal, totalNe, NFM
    
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
        if(params["sys_type"] in ["SIAM"]): NFM = 1;
        elif(params["sys_type"] in ["SIAM_RM","SIETS","SIETS_RM"]): NFM = params["NFM"];
        else: raise NotImplementedError;
        if(params["sys_type"] in ["SIAM_RM", "SIETS_RM"]): block2site = 2;
        else: block2site = 1;
        Ntotal = (params["NL"]+NFM+params["NR"]);

        # get occupancies
        yjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));

        # break up occupancies
        yjL_vs_time = np.sum(yjs_vs_time[:,:block2site*params["NL"]], axis=1);
        yjSR_vs_time = np.sum(yjs_vs_time[:,block2site*params["NL"]:block2site*(params["NL"]+NFM)],axis=1);
        yjR_vs_time = np.sum(yjs_vs_time[:,block2site*(params["NL"]+NFM):], axis=1);
        if(difference): # only plot occupancy difference vs time 0
            print("n_LL (0) = {:.4f}".format(yjL_vs_time[0]));
            yjL_vs_time = yjL_vs_time - yjL_vs_time[0];
            print("n_SR (0) = {:.4f}".format(yjSR_vs_time[0]));
            yjSR_vs_time = yjSR_vs_time - yjSR_vs_time[0];
            print("n_RL (0) = {:.4f}".format(yjR_vs_time[0]));
            yjR_vs_time = yjR_vs_time - yjR_vs_time[0];
            axes[0].set_ylim(0.0,np.max([0.01,np.max(yjSR_vs_time)]));
        
        # plot occupancies
        axes[0].plot(times, yjSR_vs_time,color=color2,marker=datamarkers[datai],label=the_label);
        if(not difference):
            axes[0].plot(times, yjL_vs_time,color=color1,marker=datamarkers[datai]);
            axes[0].plot(times, yjR_vs_time,color=color3,marker=datamarkers[datai]);
        
        # CONTINUITY: discrete time deriv of n_SR
        tup = params["tupdate"]
        delta_nSR = np.append([np.nan],yjSR_vs_time[1:] - yjSR_vs_time[:-1]); # skip 1st time
        
         # load current vs time data
        yds_vs_time = np.zeros((len(times),block2site*(NFM+1)),dtype=float);
        xds_vs_time = np.zeros((len(times),block2site*(NFM+1)),dtype=float);
        for ti in range(len(times)):
            dummy_current = np.load(datafiles[datai]+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
            if(ti==0): print("shape G_ = ",np.shape(dummy_current));
            yds_vs_time[ti] = dummy_current[:];
            xds_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(times[ti]));
            
        # CONTINUITY: discrete spatial deriv of current
        xds_plotted = [xds_vs_time[0,0], xds_vs_time[0,NFM*block2site]];
        for sitei in xds_plotted: print("G_ -> J_ at site {:.0f}".format(sitei));
        # convert units from conductance to current
        del_current = (params["Vb"]/np.pi)*(yds_vs_time[:,NFM*block2site]-yds_vs_time[:,0]); 
        del_current_tminus = np.append([np.nan],del_current[:-1]); # del_current at previous step
        del_current_t = 1*del_current; # at this timestep
        del_current_trapezoid = (del_current_t + del_current_tminus)/2; # trapezoidal rule
        del del_current
        if(True):
            axes[1].plot(times, delta_nSR, color=color2, marker=datamarkers[datai], label="$n_SR(t)-n_{SR}(t-t_{up}) (t_{up}=$"+"{:.1f})".format(tup));
            axes[1].plot(times,-tup*del_current_trapezoid, color=color3, marker="x",label="$-t_{up} \cdot \\frac{1}{2a}\left[ (J_{j+1}-J_j)_t + (J_{j+1}-J_j)_{t-t_{up}} \\right]$");
             
        # plot conservation of charge
        else:
            axes[1].plot(times, tup*del_current_trapezoid+delta_nSR, color="black",linestyle="dashed");

    # formatting
    if(difference): axes[0].set_ylabel("$n_{SR}(t)-n_{SR}(0)$", color=color2, fontsize=fontsize1);
    else: axes[0].set_ylabel("$n_{SR}(t)$", color=color2, fontsize=fontsize1);
    axes[0].set_title(the_title);
    if(len(times)>num_xticks):
        #time_ticks = np.arange(times[0], times[-1], times[-1]//max(1,num_xticks-1))
        time_ticks = [];
        axes[-1].set_xticks(time_ticks);
    axes[-1].set_xlim((times[0], times[-1]));

    # show
    axes[1].legend();
    plt.tight_layout();
    plt.show();

else: raise Exception("case = {:.0f} not supported".format(case));
