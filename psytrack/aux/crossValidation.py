import numpy as np
from .auxFunctions import read_input


def Kfold_crossVal(D, F=10, seed=None):
    """
    Splits a dataset into F folds, then save each individual fold
    as a test set with the other F-1 folds as a training set. Returns
    a list of F training datasets and F corresponding testing datasets

    Args:
        D : dict, data to be split into folds for cross-validation
        F : int, number of folds
        seed : int, random seed to reproduce xval fold split

    Returns:
        K_trainD : list, contains each fold's training dataset
        K_testD : list, contains each fold's testing dataset
    """

    ### Initialize randomness
    np.random.seed(seed)

    # Determine number of trials, and shuffle the order
    N = D["y"].shape[0]
    shuffled_array = np.arange(N)
    np.random.shuffle(shuffled_array)

    ### Iterate through the folds
    K_trainD = []
    K_testD = []
    for k in range(F):

        ### Define the k^th train/test split
        N_array = np.arange(N)
        chunk = int(N / F)

        ### Select the k^th chunk of shuffled trial indices
        test = np.sort(shuffled_array[k * chunk : (k + 1) * chunk])
        train = np.delete(N_array, test)

        ### Collect counts of where gaps will be in training set from missing test trials
        train_array = np.zeros(N)
        test2 = test.copy()
        while len(test2) > 0:
            train_array[test2 - 1] += 1
            test2 = np.array([i for i in test2 if i + 1 in test2])
        train_array = train_array[train]

        ### Shift any overnight gaps in test set back into training set
        if "dayLength" in D:
            day_array = np.zeros(N)
            cumDays = np.cumsum(D["dayLength"], dtype=int)[:-1]
            day_array[cumDays] = 1
            overlap = np.array([i for i in test if i in cumDays])
            while len(overlap) > 0:
                day_array[overlap + 1] = 1
                overlap = np.array([i + 1 for i in overlap if i + 1 in test])
            day_array = day_array[train]
            days = np.hstack((np.where(day_array)[0], [len(day_array)]))
            new_dayLength = np.hstack((days[0], np.diff(days)))
        else:
            new_dayLength = np.array([])

        ### Iterate through all keys in the original dict, save test/train copies
        trainD = {}
        testD = {}
        for key in D.keys():

            if key == "inputs":
                trainD[key] = {}
                testD[key] = {}
                continue

            try:
                if N == D[key].shape[0]:
                    trainD[key] = D[key][train]
                    testD[key] = D[key][test]
                else:
                    trainD[key] = D[key].copy()
                    testD[key] = D[key].copy()
            except:
                trainD[key] = D[key]
                testD[key] = D[key]

        for i in D["inputs"].keys():
            trainD["inputs"][i] = D["inputs"][i][train]
            testD["inputs"][i] = D["inputs"][i][test]

        trainD.update({"missing_trials": train_array, "dayLength": new_dayLength})
        testD.update({"test_inds": test})

        ### Append train/test dicts to list of dicts from all folds
        K_trainD += [trainD]
        K_testD += [testD]

    return K_trainD, K_testD


def Kfold_crossVal_check(testD, wMode, missing_trials, weights):
    """
    Calculates the log-likelihood and gw value of each trial in a
    test set given the wMode recovered from a corresponding training set. 

    Args:
        testD : dict, test data
        wMode : (K, trainN) array, weights recovered from training set
        missing_trials : (N*(F-1)/F,) array, indices corresponding to 
            each trial in the training set, with the value indicating
            how many test trials followed it in the original dataset
        weights : dict, name and count of weights in testD['inputs'] 
            to fit

    Returns:
        logli : array, each test trial's log-likelihood
        all_gw : array, each test trial's gw value
    """

    ### Form input matrix g from test set
    g = read_input(testD, weights)
    _, trainN = wMode.shape

    logli = []
    all_gw = []
    test_count = 0  # trial in the test set
    for t in range(trainN):  # iterate through each training trial
        
        # if training trial followed by one or more test trials
        for _ in range(int(missing_trials[t])):  

            ### Currently use the weights form the nearest prior training
            ### trial, could do interpolation...
            gw = g[test_count] @ wMode[:, t]
            yt = int(testD["y"][test_count]) - 1

            ### Save loglikelihood and gw value of each term in test set
            logli += [yt * gw - np.logaddexp(0, gw)]
            all_gw += [gw]

            ### Increment tracker of test trial index
            test_count += 1

    return np.array(logli), np.array(all_gw)
