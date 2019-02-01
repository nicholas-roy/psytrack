import numpy as np
from datetime import datetime, timedelta
from os import makedirs

from hyperOpt import hyperOpt


def generateSim(K=4, N=64000, hyper={},
                boundary=4.0, iterations=20, days=None, seed=42, 
                path='', save=False):
    '''
    Simulates weights, in addition to inputs and multiple realizations
    of responses. Simulation data is either saved to a file or returned
    directly.

    Args:
        K : int, number of weights to simulate
        N : int, number of trials to simulate
        hyper : dict, hyperparameters and initial values used to construct the prior
            Default is none, can include sigma, sigInit, sigDay
        boundary : float, weights are reflected from this boundary
            during simulation, is a symmetric +/- boundary
        iterations : int, # of behavioral realizations to simulate,
            same input and weights can render different choice due
            to probabilistic model, iterations are saved in 'all_Y'
        days : array, indices where a new day starts, only used if sigDay provided
        seed : int, random seed to make random simulations reproducible
        path : str, appened to folder name if save
        save : bool, if True creates a folder and saves simulation data
            in a file; else data is returned

    Returns:
        save_path | (save=True) : str, the name of the folder+file where
            simulation data was saved in the local directory
        save_dict | (save=False) : dict, contains all relevant info
            from the simulation 
    '''

    ### Reproducability
    np.random.seed(seed)

    ### Supply default hyperparameters if necessary
    if 'sigma' in hyper and hyper['sigma'] is not None:  
        sigma = hyper['sigma']
    else:
        sigma = 2**np.random.choice([-4.,-5.,-6.,-7.,-8.], size=K)

    if 'sigInit' in hyper and hyper['sigInit'] is not None: 
        sigInit = hyper['sigInit']
    else: sigInit = sigma

    if 'sigDay' in hyper and hyper['sigDay'] is not None:  
            sigDay = hyper['sigDay']
    else: sigDay = sigma



    if np.isscalar(sigma):
        invSigma_flat_k = np.ones(N) * sigma**2
        invSigma_flat_k[days] = sigDay**2   # add sigDay to day changes
        invSigma_flat_k[0] = sigInit**2     # add sigInit to beginning
        
        if missing_trials is not None:      # add extra sigma variance if a test trial gap
            invSigma_flat_k += missing_trials * sigma**2
        
        invSigma_flat = np.tile(invSigma_flat_k,K)
        return diags(invSigma_flat**-1)

    elif type(sigma) in [np.ndarray, list]:
        if len(sigma) != K: raise Exception('number of sigmas is not K')

        invSigma_flat = np.zeros(N*K)
        for k in range(K):
            invSigma_flat[k*N:(k+1)*N] = sigma[k]**2
            
            if np.isscalar(sigDay):
                invSigma_flat[k*N + days] = sigDay**2
            else:
                invSigma_flat[k*N + days] = sigDay[k]**2

            if np.isscalar(sigInit):
                invSigma_flat[k*N] = sigInit**2
            else:
                invSigma_flat[k*N] = sigInit[k]**2

            if missing_trials is not None:      # add extra sigma variance if a test trial gap
                invSigma_flat[k*N:(k+1)*N] += missing_trials * sigma[k]**2

        return diags(invSigma_flat**-1)
    ### Determine the sigmas for each weight
    if sigma is None:                        # Choose from good options
        sigma = 2**np.random.choice([-4.,-5.,-6.,-7.,-8.], size=K)
    elif type(sigma) in [int, float]:        # Specify one for all
        sigma = 2**float(sigma)
    elif type(sigma) in [list, np.ndarray]:  # Specify one for each
        sigma = 2**np.array(sigma)
    else:
        raise Exception("sigma must be a float, int, list, or array of appropriate size")
    

    ### -------------
    ### Simulation
    ### -------------
    
    # Simulate inputs
    X = np.random.normal(size=(N,K))

    # Simulate weights
    E = np.zeros((N,K))
    E[0] = np.random.normal(scale=sigInit,size=K)
    E[1:] = np.random.normal(scale=sigma,size=(N-1,K))
    W = np.cumsum(E,axis=0)
    
    # Impose a ceiling and floor boundary on W
    for i in range(len(W.T)):
        cross = (W[:,i] < -boundary) | (W[:,i] > boundary)
        while cross.any():
            ind = np.where(cross)[0][0]
            if W[ind,i] < -boundary:
                W[ind:,i] = -2*boundary - W[ind:,i]
            else:
                W[ind:,i] = 2*boundary - W[ind:,i]
            cross = (W[:,i] < -boundary) | (W[:,i] > boundary)
       
    # Save data
    save_dict = {'sigInit' : sigInit, 'sigma' : sigma, 'seed' : seed,
                 'W' : W, 'X' : X, 'K' : K, 'N' : N}

    
    ### Simulate behavioral realizations in advance
    pR = 1.0/(1.0 + np.exp(-np.sum(X*W, axis=1)))

    all_simy = []
    for i in range(iterations):
        sim_y = (pR > np.random.rand(len(pR))).astype(int) + 1 # 1 for L, 2 for R
        all_simy += [sim_y]

    # Update saved data to include behavior
    save_dict.update({'all_Y' : all_simy})


    ### Save & return file path OR return simulation data
    if save:
        # Creates unique file name from current datetime
        folder = datetime.now().strftime('%Y%m%d_%H%M%S') + path
        makedirs(folder)

        save_path = folder + '/sim.npz'
        np.savez_compressed(save_path, save_dict=save_dict)

        return save_path
    
    else:
        return save_dict



def recoverSim(data, N=None, iteration=0, save=False):
    '''
    Recovers weights from the simulation data generated by generateSim()
    Can take in a filepath pointing to simulation data, or the simulation
    dict directly. Specify how many trials of data should be recovered, 
    and from which behavioral iteration (only one). Output is either saved
    in same folder as geenrated data, or returned directly.

    Args:
        data : str or dict, either the filepath to data from generateSim()
            or the dict returned directly by generateSim()
        N : int, number of trials to simulate, if None then just the 
            full length of the simulation
        iterations : int, which # of the behavioral realizations to recover
        save : bool, if True saves recovery data as a file in same folder
            as generateSim data (error if True, but data is not a filepath);
            if False, recovery data is returned

    Returns:
        save_path | (save=True) : str, the name of the folder+file where
            recovery data was saved in the local directory
        save_dict | (save=False) : dict, contains all relevant info
            from the recovery 
    '''

    # Initialize saved recovery data
    save_dict = {'iteration' : iteration}

    ### Readin simulation input
    if type(data) is str:
        save_dict['simfile'] = data
        readin = np.load(data)['save_dict'].item()
    elif type(data) is dict:
        readin = data
    else: raise Exception("data must be either file name or dict")

    # If number of trials not specified, use all trials of simulation
    if N is None: N = readin['N']
    save_dict['N'] = N


    ### -------------
    ### Recovery
    ### -------------

    ### Initialization of recovery
    K = readin['K']
    hyper_guess = {
         'sigma'   : [2**-6]*K,  # 2**-6 is an arbitrary starting point for the search
         'sigInit' : readin['sigInit'],
         'sigDay'  : None
          }
    optList = ['sigma']
    weights = {'x' : K}

    dat = {'inputs' : {'x' : readin['X'][:N, :K]}, 
           'y' : readin['all_Y'][iteration][:N]}

    ### Run recovery, recording duration of recoverty
    START = datetime.now()
    hyp, evd, wMode, _ = hyperOpt(dat, hyper_guess, weights, optList)
    END = datetime.now()
    
    save_dict.update({'K' : K, 'hyp' : hyp, 'evd' : evd,
                      'wMode' : wMode, 'duration' : END-START})

    
    ### Save (only if genertaeSim was also saved) or return recovery results
    if save:
        if 'simfile' not in save_dict: 
            raise Exception("Can only save recovery if generateSim was also saved")
        save_path = save_dict['simfile'][:-4] + '_N' + str(N) + '_i' + str(iteration) +'.npz'
        np.savez_compressed(save_path, save_dict=save_dict)
        return save_path
    
    else:
        return save_dict

