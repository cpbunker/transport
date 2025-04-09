'''
'''

import numpy as np
import matplotlib.pyplot as plt
from transport import tddmrg, tdfci

# fig standardizing
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mylinewidth = 3.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def vs_site(js,psi,eris_or_driver,which_obs, is_impurity, block, prefactor):
    '''
    '''
    if(not isinstance(prefactor, float)): raise TypeError;
    obs_funcs = {"occ_":tddmrg.get_occ, "sz_":tddmrg.get_sz,"Sdz_":tddmrg.get_Sd_mu, 
                 "pur_":tddmrg.purity_wrapper, "G_":tddmrg.conductance_wrapper, "J_":tddmrg.pcurrent_wrapper,
                 "S2_":tddmrg.S2_wrapper, "MI_":tddmrg.mutual_info_wrapper};
    if(block): compute_func = tddmrg.compute_obs;
    else: compute_func = tdfci.compute_obs;

    # site array
    vals = np.zeros_like(js,dtype=float)
    for ji in range(len(js)):
        if(which_obs in ["S2_", "MI_"]): # WRAPPED operators on MULTIPLE sites -> all should have impurity/not impurity functionality
            if(ji!=len(js)-1):
                vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,[js[ji],js[ji+1]],is_impurity,block);
            else: vals[ji] = np.nan; # since op = op(d,d+1), we cannot compute for the final site.
        elif(which_obs in ["pur_","G_","J_"]): # WRAPPED operators
            vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,js[ji],block);
        else: # simple operators
            op = obs_funcs[which_obs](eris_or_driver,js[ji],block);
            ret = compute_func(psi, op, eris_or_driver);
            if(np.imag(ret) > 1e-10): print(ret); raise ValueError;
            vals[ji] = np.real(ret);

    return js, prefactor*vals;

def snapshot_bench(psi_mps, driver_inst, params_dict, savename, time, block=True):
    '''
    '''
    if(not isinstance(block, bool)): raise TypeError;

    # unpack
    sys_type = params_dict["sys_type"];
    NL, NR = params_dict["NL"], params_dict["NR"];

    # Sys_type tells what observables we want to compute and how to compute them
    if(sys_type=="STT"):
        is_impurity = True; # bool that tells us whether custom operators (Z, P, M) defining localized spins are defined
        NFM,  Ne = params_dict["NFM"], params_dict["Ne"];
        title_str = "$J_{sd} = $"+"{:.2f}$t_l$".format(params_dict["Jsd"])+", $N_e = ${:.0f}".format(Ne)+", $N_{conf} =$"+"{:.0f}".format(params_dict["Nconf"]);
        if("tp" in params_dict.keys()): title_str += ", $t_p =${:.1f}$t_l$".format(params_dict["tp"]);
        if("Vdelta" in params_dict.keys()): title_str += ", $\Delta V = ${:.3f}$t_l$".format(params_dict["Vdelta"]);
        # plot charge and spin vs site
        obs_strs = ["occ_","sz_","Sdz_", "J_", "S2_", "MI_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $",
        "$ \langle S_{d}^{z} \\rangle $","$\langle J_j\\rangle$","$(\mathbf{S}_d + \mathbf{S}_{d+1})^2 $","MI"];
        axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[0.5,0.0,-0.5],[1.0,-1.0],[2.0,0.0],[np.log(2),0.0]];
        if(params_dict["NFM"]< 2): # not enough impurities to do S2_ or MI_
            obs_strs, ylabels, axlines = obs_strs[:-2], ylabels[:-2], axlines[:-2];
    elif(sys_type=="SIAM"):
        is_impurity = False;
        NFM, Ne = 1, (NL+1+NR)//2;
        title_str = "$U =${:.2f}$t_l, t_h =$ {:.2f}$t_l, V_g =${:.2f}$t_l, V_b =${:.2f}$t_l$".format(params_dict["U"], params_dict["th"], params_dict["Vg"], params_dict["Vb"]);
        obs_strs = ["occ_", "sz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $", "$\langle G_{j} \\rangle/G_0$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[1.0,0.0]];
    elif(sys_type=="SIETS"):
        is_impurity = True; # bool that tells us whether custom operators (Z, P, M) defining localized spins are defined
        NFM, Ne = params_dict["NFM"], (NL+params_dict["NFM"]+NR)//2;
        title_str = "$J_{sd} = $"+"{:.2f}$t_l$, ".format(params_dict["Jsd"])+"$t_h = ${:.2f}$t_l$, $V_g =${:.2f}$t_l, V_b =${:.2f}$t_l$".format(params_dict["th"], params_dict["Vg"], params_dict["Vb"]);
        obs_strs = ["occ_", "sz_", "Sdz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{d}^{z} \\rangle $", "$\langle G_{j} \\rangle/G_0$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[0.5,0.0,-0.5],[1.0,0.0]];
    else:
        raise Exception("System type = "+sys_type+" not supported");
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    if("MSQ_spacer" in params_dict.keys()): # NFM has MSQs on either end only
        central_sites = np.array([NL,NL+NFM-1]);
    else: # NFM full of MSQs
        central_sites = np.arange(NL,NL+NFM);
    all_sites = np.arange(Nsites);

    # plot
    fig, axes = plt.subplots(len(obs_strs));
    if(psi_mps is not None): # with dmrg
        for obsi in range(len(obs_strs)):

            # what sites to find <operator> over
            if(obs_strs[obsi] in ["Sdz_","pur_","S2_","MI_"]): js_pass = central_sites; # operators on impurity sites only
            else: js_pass = all_sites;
            # what prefactors to apply to <operator>
            if(obs_strs[obsi] in ["G_"]): 
                if(sys_type in ["SIAM", "SIETS"]): 
                    prefactor = np.pi*params_dict["th"]/params_dict["Vb"]; # prefactor needed for G/G0
                else: raise NotImplementedError;
                js_pass = np.arange(NL,NL+NFM+1); # one extra
            if(obs_strs[obsi] in ["J_"]): 
                if(sys_type in ["SIAM", "SIETS"]):
                    prefactor = params_dict["th"]; # prefactor needed for J
                else: prefactor = params_dict["tl"];
                js_pass = [params_dict["Nconf"], NL, NL+NFM]; 
            else: prefactor = 1.0;

            # find <operator> vs sites
            #print(obs_strs[obsi], js_pass)
            x_js, y_js = vs_site(js_pass,psi_mps,driver_inst,obs_strs[obsi],is_impurity,block,prefactor);
            axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth,
                               label = "DMRG (te_type = "+str(params_dict["te_type"])+", dt= "+str(params_dict["time_step"])+")");
            print("Total <"+obs_strs[obsi]+"> = {:.6f}".format(np.sum(y_js)));            

            # save DMRG data
            if(not params_dict["plot"]):
                np.save(savename+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}".format(time), x_js);
                np.save(savename+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}".format(time), y_js);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j(d)$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title(title_str);
    plt.tight_layout();
    if(params_dict["plot"]): plt.show();
    else:
        np.savetxt(savename+"_arrays/"+obs_strs[0]+"title.txt",[0.0], header=title_str);
        #plt.savefig(savename+"_arrays/time{:.2f}.pdf".format(time));
    plt.close(); # keeps figure from being stored in memory

def snapshot_fromdata(loadname, time, sys_type):
    '''
    '''
    
    # plot charge and spin vs site
    if(sys_type == "STT"):
        obs_strs = ["occ_", "sz_", "Sdz_", "S2_", "MI_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{d}^{z} \\rangle $","$(\mathbf{S}_d + \mathbf{S}_{d+1})^2 $","MI"];
        axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[0.5,0.0,-0.5],[2.0,0.0],[np.log(2),0.0]];
    elif(sys_type == "SIAM"):
        obs_strs = ["occ_", "sz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $", "$\langle G_{j} \\rangle/G_0$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[1.0,0.0]];
    elif(sys_type == "SIETS"):
        obs_strs = ["occ_", "sz_", "Sdz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $", "$ \langle S_{d}^{z} \\rangle $","$\langle G_{j} \\rangle/G_0$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[0.5,0.0,-0.5],[1.0,0.0]];
    else:
        raise Exception("System type = "+sys_type+" not supported");

    # plot
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
    axes[-1].set_xlabel("$j(d)$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title( open(loadname+"_arrays/"+obs_strs[0]+"title.txt","r").read().splitlines()[0][1:]);
    plt.tight_layout();
    plt.show();

