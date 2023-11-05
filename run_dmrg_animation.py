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

# set up charge density animation
fig, ax = plt.subplots();
obs1, color1, ticks1, linewidth1, fontsize1 = "occ", "cornflowerblue", (0.0,1.0), 3.0, 16;
time0 = 0.0;
xjs = np.load(datafile+"_arrays/"+obs1+"_xjs_time{:.2f}.npy".format(time0));
yjs = np.load(datafile+"_arrays/"+obs1+"_yjs_time{:.2f}.npy".format(time0));
charge_density = ax.fill_between(xjs, yjs, color=color1);
ax.set_xlim(np.min(xjs), np.max(xjs));
ax.set_xlabel("$j$", fontsize=fontsize1);
for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
ax.set_yticks(ticks1);
ax.set_ylabel("$\langle n_j \\rangle$", color=color1, fontsize=fontsize1);

# set up impurity spin animation
obs2, color2, ticks2 = "sz", "darkgreen", (-0.5,0.5);
xds = np.load(datafile+"_arrays/"+obs2+"_xds_time{:.2f}.npy".format(time0));
yds = np.load(datafile+"_arrays/"+obs2+"_yds_time{:.2f}.npy".format(time0));
impurity_sz, = ax.plot(xds, yds, marker="^", color=color2, markersize=linewidth1**2);
axright = ax.twinx();
for tick in ticks2: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
ax.set_yticks(np.append(ticks1,ticks2));
axright.set_yticks([])
axright.set_ylabel("$\langle S_d^z \\rangle$", color=color2, fontsize=fontsize1);

# time evolve observables
def time_evolution(time):
    yjs_t = np.load(datafile+"_arrays/"+obs1+"_yjs_time{:.2f}.npy".format(time));
    ax.collections.clear();
    charge_density_update = ax.fill_between(xjs, yjs_t, color=color1)
    charge_density.update_from(charge_density_update);
    yds_t = np.load(datafile+"_arrays/"+obs2+"_yds_time{:.2f}.npy".format(time));
    impurity_sz.set_ydata(yds_t);
    return charge_density, impurity_sz;

# animate
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


