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
obs1, color1, ticks1, linewidth1, fontsize1 = "Sdz_", "darkred", (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
obs2, color2 = "occ_", "cornflowerblue";
obs3, color3 = "sz_", "darkblue";
obs4, color4 = "pur_", "gray";
num_xticks = 4;

if(case in [0]): # standard charge density vs site snapshot
    from transport import tddmrg
    from transport.tddmrg import plot
    tddmrg.plot.snapshot_fromdata(datafiles[0], time0, "STT")

if(case in [1,2]): # observable as a function of time
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    #ax.set_yticks(ticks1);
    ax.set_xlabel("Time $(\hbar/t_l)$");
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);

    # time evolution params
    tupdate = params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = time0 + ti*tupdate;
    time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1))
    ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));

    # impurity spin vs time
    Nbuffer = 0;
    if("Nbuffer" in params.keys()): Nbuffer = params["Nbuffer"];
    Nsites = Nbuffer + params["NL"]+params["NFM"]+params["NR"]; # number of d sites
    which_imp = 0;
    yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
    for ti in range(len(times)):
        yds_vs_time[ti] = 2*np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
    ax.plot(times,yds_vs_time[:,which_imp],color=color1);
    ax.set_ylabel("$2 \langle S_{"+str(which_imp)+"}^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);

    # AVERAGE electron spin vs time
    Ne = params["Ne"]; # number deloc electrons
    yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
    for ti in range(len(times)):
        yjs_vs_time[ti] = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
    yjavg_vs_time = np.sum(yjs_vs_time, axis=1)/Ne;
    ax.plot(times, yjavg_vs_time,color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel("$2 \overline{ s_j^z } /\hbar$", color=color3, fontsize=fontsize1);

    # impurity purity vs time
    if False: # plot purity
        label4 = "$||$";
        purds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            purds_vs_time[ti] = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.fill_between(times, (-2)*purds_vs_time[:,which_imp],color=color4);
    else: # plot concurrence
        obs4, color4 = "conc_", "black";
        label4 = "$\langle C_{d,d+1} \\rangle$";
        dfile = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[0]));
        purds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            purds_vs_time[ti] = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.plot(times, purds_vs_time[:,which_imp],color=color4);
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.0));
    ax4.spines.right.set(alpha=1.0);
    ax4.set_yticks([])
    ax4.set_ylabel(label4, color=color4, fontsize=fontsize1);

    # show
    plt.tight_layout();
    plt.show();

if(case in [3,4]): # observable as a function of time
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));

    # axes
    fig, ax = plt.subplots();
    ax.set_xlabel("Time $(\hbar/t_l)$");
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);

    # time evolution params
    tupdate = params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = time0 + ti*tupdate;
    time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1));
    ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));

    # SUMMED impurity spins vs time
    Nbuffer = 0;
    if("Nbuffer" in params.keys()): Nbuffer = params["Nbuffer"];
    Nsites = Nbuffer + params["NL"]+params["NFM"]+params["NR"]; # number of d sites
    yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
    for ti in range(len(times)):
        yds_vs_time[ti] = np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
    yds_summed = np.sum(yds_vs_time, axis=1);
    ax.plot(times,yds_summed,color=color1);
    ax.set_ylabel("$ \Sigma_d \langle S_d^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);

    # SUMMED electron spin vs time
    Ne = params["Ne"]; # number deloc electrons
    yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
    for ti in range(len(times)):
        yjs_vs_time[ti] = np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
    yjs_summed = np.sum(yjs_vs_time, axis=1);
    ax.plot(times, yjs_summed,color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel("$ \Sigma_j \langle s_j^z \\rangle /\hbar$", color=color3, fontsize=fontsize1);

    # SUM of all spins
    ax.plot(times, yds_summed+yjs_summed, color="black", linestyle="dashed");
    
    # show
    plt.tight_layout();
    plt.show();

if(case in [10]): # animate time evol
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("$j(d)$", fontsize=fontsize1);
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);
    
    # time evolution params
    tupdate = params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = time0 + ti*tupdate;

    # set up impurity spin animation
    xds = np.load(datafile+"_arrays/"+obs1+"xjs_time{:.2f}.npy".format(time0));
    yds = 2*np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(time0));
    impurity_sz, = ax.plot(xds, yds, marker="s", color=color1, markersize=linewidth1**2);
    ax.set_ylabel("$2 \langle S_d^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);
    time_annotation = ax.annotate("Time = {:.2f}".format(time0), (0.0,-0.96),fontsize=fontsize1);

    # set up charge density animation
    xjs = np.load(datafile+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(time0));
    yjs = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time0));
    charge_density = ax.fill_between(xjs, yjs, color=color2);
    ax2 = ax.twinx();
    ax2.set_yticks([]);
    ax2.set_ylabel("$\langle n_j \\rangle$", color=color2, fontsize=fontsize1);

    # set up deloc spin animation
    xjs_3 = np.load(datafile+"_arrays/"+obs3+"xjs_time{:.2f}.npy".format(time0));
    yjs_3 = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time0));
    spin_density, = ax.plot(xjs_3, yjs_3, marker="o", color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel("$2 \langle s_j^z \\rangle /\hbar$", color=color3, fontsize=fontsize1);

    # plot concurrence
    obs4, color4, label4 = "conc_", "black", "$\langle C_{d,d+1} \\rangle$";
    xds_4 = np.load(datafile+"_arrays/"+obs4+"xjs_time{:.2f}.npy".format(time0));
    yds_4 = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time0));
    conc, = ax.plot(xds_4,yds_4,marker="^",color=color4);
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.05));
    ax4.spines.right.set(alpha=0.0);
    ax4.set_yticks([])
    ax4.set_ylabel(label4, color=color4, fontsize=fontsize1);

    # time evolve observables
    plt.tight_layout();
    def time_evolution(time):
        # impurity spin
        yds_t = 2*np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(time));
        impurity_sz.set_ydata(yds_t);
        time_annotation.set_text("Time = {:.2f}".format(time));
        # charge density
        yjs_t = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time));
        ax.collections.clear();
        charge_density_update = ax.fill_between(xjs, yjs_t, color=color2)
        charge_density.update_from(charge_density_update);
        # spin density
        yjs_3_t = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time));
        spin_density.set_ydata(yjs_3_t);
        # concurrence
        yds_4_t = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time));
        conc.set_ydata(yds_4_t);

    # animate
    if(Nupdates > 0): interval = 1000*(10/Nupdates); # so total animation time is 10 sec
    elif(params["time_step"]==1.0): interval = 400;
    elif(params["time_step"]==0.5): interval = 200;
    else: interval = 500;
    ani = animation.FuncAnimation(fig, time_evolution,
                                  frames = times, interval=interval,
                                  repeat=True, blit=False);

    plt.show()
    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)


