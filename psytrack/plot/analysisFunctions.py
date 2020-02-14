import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

cmap = plt.get_cmap('vlag') #vlag0 = #2369bd; vlag1 = #a9373b
COLORS = {'bias' : '#FAA61A', 
          's1' : cmap(1.0), 's2' : cmap(0.0), 
          'sR' : cmap(1.0), 'sL' : cmap(0.0),
          'cR' : cmap(1.0), 'cL' : cmap(0.0),
          'c' : '#59C3C3', 'h' : '#9593D9', 's_avg' : '#99CC66',
          'emp_perf': '#E32D91', 'emp_bias': '#9252AB'}
ZORDER = {'bias' : 2, 
          's1' : 3, 's2' : 3, 
          'sR' : 3, 'sL' : 3,
          'cR' : 3, 'cL' : 3,
          'c' : 1, 'h' : 1, 's_avg' : 1}
BIAS_COLORS = {50 : 'None', 20 : COLORS['sR'], 80 : COLORS['sL']}


def plot_weights(W, weights, figsize=(3.75,1.4),
                 colors=None, zorder=None, errorbar=None, days=None):
    
    # Some useful values to have around
    N = len(W[0])
    maxval = np.max(np.abs(W))*1.1  # largest magnitude of any weight
    if colors is None: colors = COLORS
    if zorder is None: zorder = ZORDER

    ### Plotting
    fig = plt.figure(figsize=figsize)        

    # Infer (alphabetical) order of weights from dict
    labels = []
    for j in sorted(weights.keys()):
        labels += [j]*weights[j]

    for i, w in enumerate(labels):

        plt.plot(W[i], lw=1.5, alpha=0.8, ls='-', c=colors[w], zorder=zorder[w])

        # Plot errorbars on weights
        if errorbar is not None:
            plt.fill_between(np.arange(N),
                             W[i]-2*errorbar[i], W[i]+2*errorbar[i], 
                             facecolor=colors[w], zorder=zorder[w], alpha=0.2)

    # Plot vertical session lines
    if days is not None:
        if type(days) not in [list, np.ndarray]:
            raise Exception("days must be a list or array.")
        if days[-1] < N/2:  # this means day lengths were passed
            days = np.cumsum(days)
        for d in days:
            plt.axvline(d, c='black', ls='-', lw=0.5, alpha=0.5, zorder=0)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.axhline(0, c="black", ls="--", lw=1, alpha=0.5, zorder=0)
    plt.ylim(-maxval, maxval); plt.xlim(0, N)
    plt.gca().set_yticks(np.arange(-int(maxval), int(maxval)+1,1))
    plt.xlabel("Trial #"); plt.ylabel("Weights")
    
    return fig


def addBiasBlocks(fig, pL):
    plt.sca(fig.gca())
    i = 0
    while i < len(pL):
        start = i
        while i+1 < len(pL) and np.linalg.norm(pL[i] - pL[i+1]) < 0.0001:
            i += 1
        fc = BIAS_COLORS[int(100 * pL[start])]
        plt.axvspan(start, i+1, facecolor=fc, alpha=0.2, edgecolor=None)
        i += 1
    return fig
    
    
def plot_performance(dat, prediction, sigma=50, figsize=None):

    ### Data from xval, arranges the inferred gw values from heldout data
    N = len(dat['y'])
    X = np.array([i['gw'] for i in prediction]).flatten()
    test_inds = np.array([i['test_inds'] for i in prediction]).flatten()
    inds = [i for i in np.argsort(test_inds)]
    X = X[inds]
    pL = 1 / (1 + np.exp(X))
    answerR = (dat['answer'] == 2).astype(float)

    ### Plotting
    fig = plt.figure(figsize=figsize)        
    
    # Smoothing vector for errorbars
    QQQ = np.zeros(10001)
    QQQ[5000] = 1
    QQQ = gaussian_filter(QQQ, sigma)

    # Calculate smooth representation of binary accuracy
    raw_correct = dat['correct'].astype(float)
    smooth_correct = gaussian_filter(raw_correct, sigma)
    plt.plot(smooth_correct, c=COLORS['emp_perf'], lw=3, zorder=4)

    # Calculate errorbars on empirical performance
    perf_errorbars = np.sqrt(
        np.sum(QQQ**2) * gaussian_filter(
            (raw_correct - smooth_correct)**2, sigma))
    plt.fill_between(range(N),
                     smooth_correct - 2 * perf_errorbars,
                     smooth_correct + 2 * perf_errorbars,
                     facecolor=COLORS['emp_perf'], alpha=0.3, zorder=3)

    # Calculate the predicted accuracy
    pred_correct = np.abs(answerR - pL)
    smooth_pred_correct = gaussian_filter(pred_correct, sigma)
    plt.plot(smooth_pred_correct, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if "dayLength" in dat and dat["dayLength"] is not None:
        days = np.cumsum(dat["dayLength"])
        for d in days:
            plt.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
    
    # Add plotting details
    plt.axhline(0.5, c="k", ls="--", lw=1, alpha=0.5, zorder=1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(0, N); plt.ylim(0.3, 1.0)
    
    return fig


def plot_bias(dat, prediction, sigma=50, figsize=None):

    ### Data from xval, arranges the inferred gw values from heldout data
    N = len(dat['y'])
    X = np.array([i['gw'] for i in prediction]).flatten()
    test_inds = np.array([i['test_inds'] for i in prediction]).flatten()
    inds = [i for i in np.argsort(test_inds)]
    X = X[inds]
    pL = 1 / (1 + np.exp(X))
    choiceR = (dat['y'] == 2).astype(float)
    answerR = (dat['answer'] == 2).astype(float)

    ### Plotting
    fig = plt.figure(figsize=figsize)        
    
    # Smoothing vector for errorbars
    QQQ = np.zeros(10001)
    QQQ[5000] = 1
    QQQ = gaussian_filter(QQQ, sigma)

    # Calculate smooth representation of empirical bias
    raw_bias = choiceR - answerR
    smooth_bias = gaussian_filter(raw_bias, sigma)
    plt.plot(smooth_bias, c=COLORS['emp_bias'], lw=3, zorder=4)

    # Calculate errorbars on empirical performance
    bias_errorbars = np.sqrt(
        np.sum(QQQ**2) * gaussian_filter((raw_bias - smooth_bias)**2, sigma))
    plt.fill_between(range(N),
                     smooth_bias - 2 * bias_errorbars,
                     smooth_bias + 2 * bias_errorbars,
                     facecolor=COLORS['emp_bias'], alpha=0.3, zorder=3)

    ### Calculate the predicted bias
    pred_bias = (1 - pL) - answerR
    smooth_pred_bias = gaussian_filter(pred_bias, sigma)
    plt.plot(smooth_pred_bias, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if "dayLength" in dat and dat["dayLength"] is not None:
        days = np.cumsum(dat["dayLength"])
        for d in days:
            plt.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
    
    # Add plotting details
    plt.axhline(0, c="k", ls="--", lw=1, alpha=0.5, zorder=1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(0, N); plt.ylim(-0.5, 0.5)
    
    return fig