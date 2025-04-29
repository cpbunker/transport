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

def direct_eigenenergies(params):
    if(params["u"]==0.0 and (params["v"]==-1.0 and params["w"]==-1.0)):
        if(params["th"]==1.0):
            arr_in = np.array(
                [-1.9909438451461685, -1.9638573945254123, -1.918985947228994, -1.8567358660321451, -1.7776708973098465, -1.6825070656623617, -1.5721061894855748, -1.4474680762101402, -1.30972146789057, -1.160113819142396, -0.9999999999999997, -0.8308300260037728, -0.6541359266348432, -0.47151787101885423, -0.2846296765465703, -0.09516383164748421, 0.09516383164748438, 0.2846296765465699, 0.4715178710188541, 0.6541359266348434, 0.8308300260037731, 0.9999999999999997, 1.1601138191423965, 1.3097214678905704, 1.44746807621014, 1.572106189485575, 1.6825070656623617, 1.777670897309846, 1.856735866032144, 1.9189859472289945, 1.9638573945254127, 1.9909438451461687]
                 );
        elif(params["th"]==0.4):
            arr_in = np.array(
                    [-1.966455470330932, -1.957012998548398, -1.867118984382385, -1.8302189130147226, -1.7060250835611506, -1.6265693969020472, -1.4909257680775385, -1.3623653366776498, -1.2395741834025562, -1.1045310988492005, -0.9669112497515308, -0.8552182087923157, -0.6105204433750158, -0.5299015900429188, -0.21634882581259737, -0.1702454544922317, 0.17024545449223194, 0.21634882581259737, 0.5299015900429185, 0.6105204433750169, 0.8552182087923157, 0.9669112497515305, 1.1045310988492008, 1.2395741834025564, 1.3623653366776498, 1.4909257680775385, 1.6265693969020467, 1.7060250835611508, 1.8302189130147222, 1.8671189843823846, 1.9570129985483986, 1.9664554703309314]
                    );
        else: raise NotImplementedError;
    elif(params["u"]==0.2 and (params["v"]==-1.0 and params["w"]==-1.0)):
        if(params["th"]==0.4):
            arr_in = np.array(
                [-1.976599887886884, -1.9672061093051214, -1.8778001229739834, -1.8411141381122447, -1.7177082364999687, -1.6388190879222417, -1.5042804412467763, -1.3769674326507526, -1.2556050956244622, -1.1224922932140882, -0.9873790381085005, -0.8782927670486285, -0.642444714959059, -0.5663882900714076, -0.2946299618682559, -0.26264712976780613, 0.26264712976780513, 0.29462996186825485, 0.5663882900714075, 0.6424447149590584, 0.8782927670486285, 0.9873790381085, 1.122492293214088, 1.2556050956244613, 1.376967432650753, 1.5042804412467763, 1.6388190879222426, 1.7177082364999685, 1.8411141381122453, 1.8778001229739834, 1.9672061093051216, 1.9765998878868851]
                );
        elif(params["th"]==1.0):
            arr_in = np.array([-2.000964116251317, -1.9740151635770546, -1.9293799692290687, -1.8674763924103943, -1.7888861951343902, -1.6943523913294343, -1.5847769152214626, -1.4612199805804345, -1.3249038921572502, -1.1772272819490535, -1.0198039027185568, -0.8545633575747498, -0.6840276387065257, -0.5121807324472026, -0.34787074147016905, -0.22148624077768553, 0.22148624077768536, 0.347870741470169, 0.5121807324472029, 0.6840276387065258, 0.8545633575747499, 1.0198039027185575, 1.1772272819490541, 1.32490389215725, 1.461219980580434, 1.5847769152214626, 1.6943523913294343, 1.7888861951343902, 1.8674763924103945, 1.9293799692290685, 1.9740151635770546, 2.000964116251317]
                    );
        else: raise NotImplementedError;
    elif(params["u"]==0.0 and params["th"] == 1.0 and (params["v"]==-0.95 and params["w"]==-1.0)):
        arr_in = np.array(
                [-1.9411623077635933, -1.9147290470410678, -1.870938920267093, -1.810187281691921, -1.7330224177248268, -1.6401403422798115, -1.5323780888228367, -1.4107054305882416, -1.27621483525588, -1.1301091441761741, -0.9736856082116285, -0.808312289413186, -0.6353832818679812, -0.45619377750777457, -0.27133159564160964, -0.06373974285727309, 0.06373974285727317, 0.27133159564160914, 0.45619377750777435, 0.6353832818679813, 0.8083122894131854, 0.973685608211628, 1.1301091441761741, 1.27621483525588, 1.4107054305882412, 1.5323780888228358, 1.6401403422798115, 1.733022417724827, 1.8101872816919211, 1.8709389202670936, 1.914729047041068, 1.9411623077635929]
                );
    else: raise NotImplementedError;
    return arr_in;
    
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
    print("Filling {:.0f}th state of {:.0f} total molecular orbs".format(initindex, len(vecs_t0)));
    print("Init energy state = {:.4f}".format(vals_t0[initindex]));
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
    
    # show
    plt.legend();
    plt.tight_layout();
    plt.show();

elif(case=="dos"):
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
        Es_direct = direct_eigenenergies(params);
        dos_direct = energies_to_dos(Es_direct, Gamma=0.01);
        use_dos_direct = False;

	# analyze
        direct_spacings = (Es_direct[1:] - Es_direct[:-1]);
        myGamma = np.min(direct_spacings)/2;
        print("Gamma = min(E_spacing)/2 = {:.6f}".format(myGamma));
        dos_direct = energies_to_dos(Es_direct, Gamma=myGamma);
        E_gap_tilde = np.min(Es_direct[len(dos_direct)//2:]) - np.max(Es_direct[:len(dos_direct)//2]);
        print("E_gap_tilde = {:.6f}".format(E_gap_tilde));

	# plotting
        for band in [0,1]:
            this_band = direct_eigenenergies(params)[band*len(dos_direct)//2:(band+1)*len(dos_direct)//2];
            this_dos = dos_direct[band*len(dos_direct)//2:(band+1)*len(dos_direct)//2];
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


