'''
'''

from transport import tdfci, tddmrg

import numpy as np
import matplotlib.pyplot as plt

import sys
import json

def lorentzian(xarr, x0, Gamma):
    return (1/np.pi)*(0.5*Gamma)/((xarr-x0)**2 + (0.5*Gamma)**2);

def energies_to_dos(Evals, Gamma):
    dosvals = np.zeros_like(Evals);
    for E in Evals:
        dosvals += lorentzian(Evals, E, Gamma);

    return dosvals;
    
def snapshot(state,time):
    '''
    '''
    n_yjs = np.full((len(state),),np.nan,dtype=float);
    for ni in range(len(n_yjs)):
        nop = 0*np.eye(len(state));
        nop[ni,ni] = 1;
        n_yjs[ni] = np.real(np.dot( np.conj(state), np.matmul(nop, state)));
    plt.plot(range(len(n_yjs)), n_yjs);
    plt.suptitle("Time = {:.2f}".format(time));
    plt.show();

if(__name__ == "__main__"):
    # top level
    verbose = 2; assert verbose in [1,2,3];
    np.set_printoptions(precision = 4, suppress = True);
    case = sys.argv[1];
    json_name = sys.argv[2];
    try:
        try:
            params = json.load(open(json_name+".txt"));
        except:
            params = json.load(open(json_name));
            json_name = json_name[:-4];
        print(">>> Params = ",params);
    except:
        raise Exception(json_name+" cannot be found");

if(case=="transport"):

    # construct Time=0 single-body Hamiltonian as matrix
    h1e_t0, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
    h1e_t0, g2e_dummy = tddmrg.H_RM_polarizer(params, (h1e_t0, g2e_dummy), block=False);
    h1e_t0 = h1e_t0[::2,::2]; # <- make spinless !!
    #h1e_t0[0,0] += (-100); # gd state = 0th site only, for testing
    vals_t0, vecs_t0 = tdfci.solver(h1e_t0);
    centrals = np.arange(params["NL"],params["NL"]+params["NFM"]);
    RMdofs = 2;
    print("h1e_t0 = ");
    print(h1e_t0[:8,:8]); 
    print(h1e_t0[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1),RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]); 
    print(h1e_t0[-8:,-8:]);
    
    # selection of M^th Time=0 eigenstate as the initial state
    initindex = int(sys.argv[3]);
    initstate = vecs_t0[initindex];
    #print("Filling {:.0f}th state of {:.0f} total molecular orbs".format(initindex, len(vecs_t0)));
    #print("Init energy state = {:.4f}".format(vals_t0[initindex]));
    print("Init energy spectrum:");
    filled_str = ["[ ]","[X]"];
    for statei in range(len(vecs_t0)): print("{:.6f} ".format(vals_t0[statei])+filled_str[statei==initindex]);
    del h1e_t0, vals_t0, vecs_t0;
    
    # Time > 0 eigenstates and eigenvalues
    h1e, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
    # no polarizer !
    h1e = h1e[::2,::2]; # <- make spinless !!
    vals, vecs = tdfci.solver(h1e);
    print("h1e = ");
    print(h1e[:8,:8]);
    print(h1e[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1),RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]);
    print(h1e[-8:,-8:]);  
    
    # set up observables over time propagation
    evolvedstate = np.copy(initstate);
    time_N = params["Nupdates"]*int(params["tupdate"]/params["time_step"]);
    print("time_N = ", time_N);
    times = np.arange(time_N)*params["time_step"];
    n0_op = 0.0*np.eye(len(h1e)); n0_op[0,0] = 1;
    n0_yjs = np.full((time_N,),np.nan);
    nSR_op = 0.0*np.eye(len(h1e)); 
    for muj in [RMdofs*params["NL"],RMdofs*params["NL"]+1]: nSR_op[muj,muj] = 1;
    nSR_yjs = np.full((time_N,),np.nan);
    js_pass = np.append(centrals, [centrals[-1]+1]); # one extra
    GL_op = 0.0*np.eye(len(h1e),dtype=complex);
    GR_op = 0.0*np.eye(len(h1e),dtype=complex);
    GLR = [GL_op, GR_op];
    for Gi in range(len(GLR)):
        muA, muB_prev = RMdofs*js_pass[Gi], RMdofs*js_pass[Gi]-1;
        print("G_ at site {:.0f}".format(muA));  
        GLR[Gi][muA, muB_prev] += complex(0,-1);
        GLR[Gi][muB_prev, muA] += complex(0, 1);  
    GL_yjs = np.full((time_N,),np.nan);
    GR_yjs = np.full((time_N,),np.nan);
    
    # time propagate
    for time_stepi in range(time_N): # iter over time steps
 
        # measure observables at Time = time_step*time_stepi
        n0_yjs_ret = np.dot( np.conj(evolvedstate), np.matmul(n0_op, evolvedstate));
        assert(abs(np.imag(n0_yjs_ret)) < 1e-10);
        n0_yjs[time_stepi] = np.real(n0_yjs_ret);
        nSR_yjs_ret = np.dot( np.conj(evolvedstate), np.matmul(nSR_op, evolvedstate));
        assert(abs(np.imag(nSR_yjs_ret)) < 1e-10);
        nSR_yjs[time_stepi] = np.real(nSR_yjs_ret);
        
        # conductance
        GL_yjs_ret =  np.dot( np.conj(evolvedstate), np.matmul(GL_op, evolvedstate)); 
        assert(abs(np.imag(GL_yjs_ret)) < 1e-10);
        GL_yjs[time_stepi] = np.pi*params["th"]/params["Vb"]*np.real(GL_yjs_ret); 
        GR_yjs_ret =  np.dot( np.conj(evolvedstate), np.matmul(GR_op, evolvedstate)); 
        assert(abs(np.imag(GR_yjs_ret)) < 1e-10);
        GR_yjs[time_stepi] = np.pi*params["th"]/params["Vb"]*np.real(GR_yjs_ret);         
        
        if(time_stepi%100==0):snapshot(evolvedstate, time_stepi*params["time_step"]);
        evolvedstate = tdfci.propagator(evolvedstate, params["time_step"], vals, vecs);
        
    # plot
    plt.plot(times, n0_yjs, label = "$n_0$");
    plt.plot(times, nSR_yjs, label = "$n_{SR}$");
    plt.plot(times,GL_yjs, label = "$G_L$");
    plt.plot(times,GR_yjs, label = "$G_R$");
    
    # format
    plt.suptitle("$N_{eff} =$"+"{:.0f}, $t_h =${:.2f}, $V_g =${:.2f}, $u =${:.2f}, $v =${:.2f}, $w =${:.2f}".format(2*(initindex+1), params["th"], params["Vg"], params["u"], params["v"], params["w"]));
    
    # show
    plt.legend();
    plt.tight_layout();
    plt.show();

elif(case=="dos"):

    # construct Time=0 single-body Hamiltonian as matrix
    h1e_t0, g2e_dummy = tddmrg.H_RM_builder(params, block=False);
    h1e_t0, g2e_dummy = tddmrg.H_RM_polarizer(params, (h1e_t0, g2e_dummy), block=False);
    h1e_t0 = h1e_t0[::2,::2]; # <- make spinless !!
    #h1e_t0[0,0] += (-100); # gd state = 0th site only, for testing
    vals_t0, vecs_t0 = tdfci.solver(h1e_t0);
    centrals = np.arange(params["NL"],params["NL"]+params["NFM"]);
    RMdofs = 2;
    print("h1e_t0 = ");
    print(h1e_t0[:8,:8]); 
    print(h1e_t0[RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1),RMdofs*(centrals[0]-1):RMdofs*(centrals[-1]+1)]); 
    print(h1e_t0[-8:,-8:]);

    if(True):
        # load data from json
        u, v, w = params["u"], params["v"], params["w"];

        # Rice Mele 
        ks = np.linspace(-np.pi, np.pi, 999);
        Es = np.append(-np.sqrt(u**2 + v**2 + w**2 + 2*v*w*np.cos(ks)), np.sqrt(u**2 + v**2 + w**2 + 2*v*w*np.cos(ks))); # dispersion
        band_edges = np.array([np.sqrt(u*u+(w+v)*(w+v)),
                       np.sqrt(u*u+(w-v)*(w-v))]);

	####
	####
	# DOS from analytical dispersion
        dos_an = (2/np.pi)/abs(np.gradient(Es,np.append(ks,ks))); # handles both bands at once

	# plotting
        fig, (dispax,dosax) = plt.subplots(ncols=2, sharey=True);
        for band in [0,1]:
            this_band = Es[band*len(ks):(band+1)*len(ks)];
            this_dos = dos_an[band*len(ks):(band+1)*len(ks)];
            dispax.plot(ks/np.pi, this_band);
            dosax.plot(this_dos,this_band);

	####
	####
	# DOS from manual hamiltonian diagonalization
        Es_direct = vals_t0 #direct_eigenenergies(params);
        dos_direct = energies_to_dos(Es_direct, Gamma=0.01);
        use_dos_direct = False;

	# analyze
        direct_spacings = (Es_direct[1:] - Es_direct[:-1]);
        myGamma = np.min(direct_spacings)/2;
        print("Gamma = min(E_spacing)/2 = {:.6f}".format(myGamma));
        dos_direct = energies_to_dos(Es_direct, Gamma=myGamma);
        E_gap_tilde = np.min(Es_direct[len(Es_direct)//2:]) - np.max(Es_direct[:len(Es_direct)//2]);
        print("E_gap_tilde = {:.6f}".format(E_gap_tilde));

	# plotting
        for band in [0,1]:
            this_band = Es_direct[band*len(Es_direct)//2:(band+1)*len(Es_direct)//2];
            this_dos = dos_direct[band*len(Es_direct)//2:(band+1)*len(Es_direct)//2];
            print("***")
            print(this_band)
            print(this_dos)
            if(use_dos_direct): dosax.plot(this_dos,this_band);
            else: 
                for E in this_band: dosax.axhline(E, color=["tab:green","darkgreen"][band]);

	# title
        Egap = np.min(band_edges) - np.max(-band_edges);
        title_or_label = "$t_h =${:.2f}, $u =${:.2f}, $v =${:.2f}, $w =${:.2f}".format(params["th"],params["u"],params["v"],params["w"]);
        title_or_label += ", $E_{gap} =$"+"{:.2f}".format(Egap);
        fig.suptitle(title_or_label);

        # format
        dispax.set_xlim(-1.0,1.0);
        dispax.set_xlabel("$ka/\pi$");
        if(not use_dos_direct): dosax.set_xlim(0.0,20.0);
        else: dosax.set_xlim(0.0,np.max(dos_direct));
        dosax.set_xlabel("$\\rho$");
        dosax.set_ylabel("$\\tilde{E}_{gap} = $"+"{:.6f}".format(E_gap_tilde));
        for bedge in band_edges: 
            for edgesign in [+1,-1]:
                dispax.axhline(edgesign*bedge, color="black", linestyle="dashed");
                dosax.axhline( edgesign*bedge, color="black", linestyle="dashed");
        dispax.set_ylabel("$E$");

	# show
        plt.show();
	
else: raise Exception("case = "+case+" not supported");


