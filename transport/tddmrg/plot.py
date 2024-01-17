'''
'''

import numpy as np
import matplotlib.pyplot as plt
from transport import tddmrg

# fig standardizing
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mylinewidth = 3.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def vs_site(js,psi,eris_or_driver,which_obs):
    '''
    '''
    obs_funcs = {"occ_":tddmrg.get_occ, "sz_":tddmrg.get_sz,"Sdz_":tddmrg.get_Sd_mu, 
            "conc_":tddmrg.concurrence_wrapper, "pur_":tddmrg.purity_wrapper};
    # site array
    vals = np.zeros_like(js,dtype=float)
    for ji in range(len(js)):
        if(which_obs in ["conc_"]): 
            if(ji!=len(js)-1): vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,[js[ji],js[ji+1]]);
            else: vals[ji] = np.nan; # since concur = C(d,d+1), we cannot compute C for the final d site.
        elif(which_obs in ["pur_"]):
            vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,js[ji]);
        else:
            op = obs_funcs[which_obs](eris_or_driver,js[ji]);
            vals[ji] = np.real(tddmrg.compute_obs(psi, op, eris_or_driver));

    return js, vals;

def snapshot_bench(psi_mps, driver_inst, params_dict, savename, time = 0.0):
    '''
    '''

    # unpack
    concur_sites = params_dict["ex_sites"];
    plot_fig = params_dict["plot"];
    Jsd, Jx, Jz = params_dict["Jsd"], params_dict["Jx"], params_dict["Jz"];
    NL, NFM, NR, Ne = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Ne"];
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    central_sites = np.array([j for j in range(NL,NL+NFM)]);
    all_sites = np.array([j for j in range(Nsites)]);

    # plot charge and spin vs site
    obs_strs = ["occ_","sz_","Sdz_","pur_","conc_"];
    ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{j}^{z} \\rangle $","$|\mathbf{S}_j|$","$C_{j,j+1}$"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[0.5,0.0,-0.5],[0.5,0.0],[1.0,0.0]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);

    if(psi_mps is not None): # with dmrg
        C_dmrg = tddmrg.concurrence_wrapper(psi_mps, driver_inst, concur_sites);
        for obsi in range(len(obs_strs)):
            if(obs_strs[obsi] not in ["Sdz_","conc_","pur_"]): js_pass = all_sites;
            else: js_pass = central_sites;
            x_js, y_js = vs_site(js_pass,psi_mps,driver_inst,obs_strs[obsi]);
            axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth,
                               label = ("DMRG ($C"+str(concur_sites)+"=${:.2f})").format(C_dmrg));
            
            # save DMRG data
            if(not plot_fig):
                np.save(savename[:-4]+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}".format(time), x_js);
                np.save(savename[:-4]+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}".format(time), y_js);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    title_str = "$J_{sd} = $"+"{:.4f}$t_l$".format(Jsd)+", $J_x = ${:.4f}$t_l$, $J_z = ${:.4f}$t_l$, $N_e = ${:.0f}".format(Jx, Jz, Ne);
    axes[0].set_title(title_str);
    plt.tight_layout();
    if(plot_fig): plt.show();
    else:
        np.savetxt(savename[:-4]+"_arrays/"+obs_strs[0]+"title.txt",[0.0], header=title_str);
        plt.savefig(savename[:-4]+"_arrays/time{:.2f}.pdf".format(time));
    plt.close(); # keeps figure from being stored in memory

def snapshot_fromdata(loadname, time):
    '''
    '''
    
    # plot charge and spin vs site
    obs_strs = ["occ_","sz_","Sdz_","pur_","conc_"];
    ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{j}^{z} \\rangle $","$|\mathbf{S}_j|$","$C_{j,j+1}$"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[0.5,0.0,-0.5],[0.5,0.0],[1.0,0.0]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);
    for obsi in range(len(obs_strs)):
        y_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}.npy".format(time));
        x_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}.npy".format(time));
        axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title( open(loadname+"_arrays/"+obs_strs[0]+"title.txt","r").read().splitlines()[0][1:]);
    plt.tight_layout();
    plt.show();

