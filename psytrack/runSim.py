import numpy as np
from datetime import datetime, timedelta
from os import makedirs
from .hyperOpt import hyperOpt


def generateSim(K=4,
                N=64000,
                hyper={},
                days=None,
                boundary=4.0,
                iterations=20,
                seed=None,
                savePath=None):
    '''Simulates weights, inputs, and choices under the model.

    Args:
        K : int, number of weights to simulate
        N : int, number of trials to simulate
        hyper : dict, hyperparameters and initial values used to construct the
            prior. Default is none, can include sigma, sigInit, sigDay
        days : list or array, list of the trial indices on which to apply the
            sigDay hyperparameter instead of the sigma
        boundary : float, weights are reflected from this boundary
            during simulation, is a symmetric +/- boundary
        iterations : int, # of behavioral realizations to simulate,
            same input and weights can render different choice due
            to probabilistic model, iterations are saved in 'all_Y'
        seed : int, random seed to make random simulations reproducible
        savePath : str, if given creates a folder and saves simulation data
            in a file; else data is returned

    Returns:
        save_path | (if savePath) : str, the name of the folder+file where
            simulation data was saved in the local directory
        save_dict | (if no SavePath) : dict, contains all relevant info
            from the simulation 
    '''

    # Reproducability
    np.random.seed(seed)

    # Supply default hyperparameters if necessary
    sigmaDefault = 2**np.random.choice([-4.0, -5.0, -6.0, -7.0, -8.0], size=K)
    sigInitDefault = np.array([4.0] * K)
    sigDayDefault = 2**np.random.choice([1.0, 0.0, -1.0], size=K)

    if 'sigma' not in hyper:
        sigma = sigmaDefault
    elif hyper['sigma'] is None:
        sigma = sigmaDefault
    elif np.isscalar(hyper['sigma']):
        sigma = np.array([hyper['sigma']] * K)
    elif ((type(hyper['sigma']) in [np.ndarray, list]) and
          (len(hyper['sigma']) == K)):
        sigma = hyper['sigma']
    else:
        raise Exception('hyper["sigma"] must be either a scalar or a list or '
                        'array of len K')

    if 'sigInit' not in hyper:
        sigInit = sigInitDefault
    elif hyper['sigInit'] is None:
        sigInit = sigInitDefault
    elif np.isscalar(hyper['sigInit']):
        sigInit = np.array([hyper['sigInit']] * K)
    elif (type(hyper['sigInit']) in [np.ndarray, list]) and (len(hyper['sigInit']) == K):
        sigInit = hyper['sigInit']
    else:
        raise Exception('hyper["sigInit"] must be either a scalar or a list or '
                        'array of len K.')

    if days is None:
        sigDay = None
    elif 'sigDay' not in hyper:
        sigDay = sigDayDefault
    elif hyper['sigDay'] is None:
        sigDay = sigDayDefault
    elif np.isscalar(hyper['sigDay']):
        sigDay = np.array([hyper['sigDay']] * K)
    elif ((type(hyper['sigDay']) in [np.ndarray, list]) and
          (len(hyper['sigDay']) == K)):
        sigDay = hyper['sigDay']
    else:
        raise Exception('hyper["sigDay"] must be either a scalar or a list or '
                        'array of len K.')

    # -------------
    # Simulation
    # -------------

    # Simulate inputs
    X = np.random.normal(size=(N, K))

    # Simulate weights
    E = np.zeros((N, K))
    E[0] = np.random.normal(scale=sigInit, size=K)
    E[1:] = np.random.normal(scale=sigma, size=(N - 1, K))
    if sigDay is not None:
        E[np.cumsum(days)] = np.random.normal(scale=sigDay, size=(len(days), K))
    W = np.cumsum(E, axis=0)

    # Impose a ceiling and floor boundary on W
    for i in range(len(W.T)):
        cross = (W[:, i] < -boundary) | (W[:, i] > boundary)
        while cross.any():
            ind = np.where(cross)[0][0]
            if W[ind, i] < -boundary:
                W[ind:, i] = -2*boundary - W[ind:, i]
            else:
                W[ind:, i] = 2*boundary - W[ind:, i]
            cross = (W[:, i] < -boundary) | (W[:, i] > boundary)

    # Save data
    save_dict = {
        'sigInit': sigInit,
        'sigDay' : sigDay,
        'sigma': sigma,
        'dayLength' : days,
        'seed': seed,
        'W': W,
        'X': X,
        'K': K,
        'N': N,
    }

    # Simulate behavioral realizations in advance
    pR = 1.0 / (1.0 + np.exp(-np.sum(X * W, axis=1)))

    all_simy = []
    for i in range(iterations):
        sim_y = (pR > np.random.rand(
            len(pR))).astype(int) + 1  # 1 for L, 2 for R
        all_simy += [sim_y]

    # Update saved data to include behavior
    save_dict.update({'all_Y': all_simy})

    # Save & return file path OR return simulation data
    if savePath is not None:
        # Creates unique file name from current datetime
        folder = datetime.now().strftime('%Y%m%d_%H%M%S') + savePath
        makedirs(folder)

        fullSavePath = folder + '/sim.npz'
        np.savez_compressed(fullSavePath, save_dict=save_dict)

        return fullSavePath

    else:
        return save_dict


def recoverSim(data, N=None, iteration=0, hess_calc="All", save=False):
    '''Recovers weights from the simulation data generated by generateSim().
    
    Can take in a filepath pointing to simulation data, or the simulation
    dict directly. Specify how many trials of data should be recovered, 
    and from which behavioral iteration (only one). Output is either saved
    in same folder as generated data, or returned directly.

    Args:
        data : str or dict, either the filepath to data from generateSim()
            or the dict returned directly by generateSim()
        N : int, number of trials to simulate, if None then just the 
            full length of the simulation
        iterations : int, which # of the behavioral realizations to recover
        hess_calc : str, passed to hyperOpt(), error bars to calculate
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
    save_dict = {'iteration': iteration}

    # Readin simulation input
    if type(data) is str:
        save_dict['simfile'] = data
        readin = np.load(data, allow_pickle=True)['save_dict'].item()
    elif type(data) is dict:
        readin = data
    else:
        raise Exception('data must be either file name or dict')

    # If number of trials not specified, use all trials of simulation
    if N is None:
        N = readin['N']
    save_dict['N'] = N

    # -------------
    # Recovery
    # -------------

    # Initialization of recovery
    K = readin['K']
    weights = {'x': K}
    hyper_guess = {
        # 2**-6 is an arbitrary starting point for the search
        'sigma': [2**-6] * K,
        'sigInit': [2**4] * K,
        'sigDay': None,
    }
    optList = ['sigma']
    dat = {
        'inputs': {
            'x': readin['X'][:N, :K]
        },
        'y': readin['all_Y'][iteration][:N]
    }
        
    # Detect whether to include sigDay in optimization
    if 'dayLength' in readin and readin['dayLength'] is not None:
        # 2**-1 is an arbitrary starting point for the search
        hyper_guess['sigDay'] = [2**-1] * K
        optList = ['sigma', 'sigDay']
        dat['dayLength'] = readin['dayLength']


    # Run recovery, recording duration of recoverty
    START = datetime.now()
    hyp, evd, wMode, hess_info = hyperOpt(dat, hyper_guess, weights, optList,
                                          hess_calc=hess_calc)
    END = datetime.now()

    save_dict.update({
        'K': K,
        'input' : data,
        'hyp': hyp,
        'evd': evd,
        'wMode': wMode,
        'hess_info' : hess_info,
        'duration': END - START
    })

    # Save (only if generateSim was also saved) or return recovery results
    if save:
        if 'simfile' not in save_dict:
            raise Exception(
                'Can only save recovery if generateSim was also saved')
        save_path = (save_dict['simfile'][:-4] + '_N' + str(N) + '_i' +
                     str(iteration) + '.npz')
        np.savez_compressed(save_path, save_dict=save_dict)
        return save_path

    else:
        return save_dict
