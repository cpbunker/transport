'''
'''

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

import sys
import json

# time evolution params
datafile = sys.argv[1];
params = json.load(open(datafile+".txt"));
tupdate = params["tupdate"];
Nupdates = params["Nupdates"];
times = np.zeros((Nupdates+1,),dtype=float);
for ti in range(len(times)):
    times[ti] = ti*tupdate;

# set up impurity spin animation
fig, ax = plt.subplots();
obs1, color1, ticks1, linewidth1, fontsize1 = "sz_", "darkgreen", (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
ax.set_yticks(ticks1);
time0 = 0.0;
xds = np.load(datafile+"_arrays/"+obs1+"xds_time{:.2f}.npy".format(time0));
yds = 2*np.load(datafile+"_arrays/"+obs1+"yds_time{:.2f}.npy".format(time0));
impurity_sz, = ax.plot(xds, yds, marker="^", color=color1, markersize=linewidth1**2);
ax.set_ylabel("$2 \langle S_d^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);
ax.set_xlabel("$j$", fontsize=fontsize1);

# set up charge density animation
obs2, color2 = "occ_", "cornflowerblue";
xjs = np.load(datafile+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(time0));
yjs = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time0));
charge_density = ax.fill_between(xjs, yjs, color=color2);
ax2 = ax.twinx();
ax2.set_yticks([])
ax2.set_ylabel("$\langle n_j \\rangle$", color=color2, fontsize=fontsize1);

# set up deloc spin animation
obs3, color3 = "sz_", "darkblue";
xjs_3 = np.load(datafile+"_arrays/"+obs3+"xjs_time{:.2f}.npy".format(time0));
yjs_3 = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time0));
spin_density, = ax.plot(xjs_3, yjs_3, marker="o", color=color3);
ax3 = ax.twinx();
ax3.spines.right.set_position(("axes", 1.06));
ax3.spines.right.set(alpha=0.0);
ax3.set_yticks([])
ax3.set_ylabel("$2 \langle s_j^z \\rangle /\hbar$", color=color3, fontsize=fontsize1);

# time evolve observables
def time_evolution(time):
    # impurity spin
    yds_t = 2*np.load(datafile+"_arrays/"+obs1+"yds_time{:.2f}.npy".format(time));
    impurity_sz.set_ydata(yds_t);
    # charge density
    yjs_t = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time));
    ax.collections.clear();
    charge_density_update = ax.fill_between(xjs, yjs_t, color=color2)
    charge_density.update_from(charge_density_update);
    # spin density
    yjs_3_t = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time));
    spin_density.set_ydata(yjs_3_t);

# animate
ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);
ani = animation.FuncAnimation(fig, time_evolution,
                              frames = times, interval=600,
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


