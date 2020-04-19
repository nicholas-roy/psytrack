import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

COLORS = {'bias' : '#FAA61A', 
          's1' : "#A9373B", 's2' : "#2369BD", 
          's_a' : "#A9373B", 's_b' : "#2369BD", 
          'sR' : "#A9373B", 'sL' : "#2369BD",
          'cR' : "#A9373B", 'cL' : "#2369BD",
          'c' : '#59C3C3', 'h' : '#9593D9', 's_avg' : '#99CC66',
          'emp_perf': '#E32D91', 'emp_bias': '#9252AB'}
ZORDER = {'bias' : 2, 
          's1' : 3, 's2' : 3, 
          's_a' : 3, 's_b' : 3, 
          'sR' : 3, 'sL' : 3,
          'cR' : 3, 'cL' : 3,
          'c' : 1, 'h' : 1, 's_avg' : 1}


def plot_weights(W, weight_dict=None, figsize=(5, 2),
                 colors=None, zorder=None, errorbar=None, days=None):
    '''Plots weights in a quick and reasonable way.
    
    Args:
        W: weights to plot.
        weight_dict: names of weights in W, used to color and label lines.
        figsize: size of figure.
        colors: a dict mapping weight names from `weight_dict` to colors.
            Defaults to nice preset values for common weight names.
        zorder: a dict mapping weight names from `weight_dict` to zorder.
            Defaults to nice preset values for common weight names.
        errorbar: optional array for size a 1 standard error at each trial
            for each weight (same shape as W).
        days: list of session lengths or trials index of session boundaries.
    
    Returns:
        fig: The figure, to be modified further if necessary.
    '''
    
    # Some useful values to have around
    K, N = W.shape
    maxval = np.max(np.abs(W))*1.1  # largest magnitude of any weight
    if colors is None: colors = COLORS
    if zorder is None: zorder = ZORDER

    # Infer (alphabetical) order of weights from dict
    if weight_dict is not None:
        labels = []
        for j in sorted(weight_dict.keys()):
            labels += [j]*weight_dict[j]
    else:
        labels = [i for i in range(K)]
        colors = {i: np.unique(list(COLORS.values()))[i] for i in range(K)}
        zorder = {i: i+1 for i in range(K)}

    # Plot weights and credible intervals
    fig = plt.figure(figsize=figsize)        
    for i, w in enumerate(labels):
        plt.plot(W[i], lw=1.5, alpha=0.8, ls='-', c=colors[w],
                 zorder=zorder[w], label=w)
        if errorbar is not None:  # Plot 95% credible intervals on weights
            plt.fill_between(np.arange(N),
                             W[i]-1.96*errorbar[i], W[i]+1.96*errorbar[i], 
                             facecolor=colors[w], zorder=zorder[w], alpha=0.2)

    # Plot vertical session lines
    if days is not None:
        if type(days) not in [list, np.ndarray]:
            raise Exception('days must be a list or array.')
        if np.sum(days) <= N:  # this means day lengths were passed
            days = np.cumsum(days)
        for d in days:
            if d < N:
                plt.axvline(d, c='black', ls='-', lw=0.5, alpha=0.5, zorder=0)

    # Further tweaks to make plot nice
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.axhline(0, c='black', ls='--', lw=1, alpha=0.5, zorder=0)
    plt.gca().set_yticks(np.arange(-int(2*maxval), int(2*maxval)+1,1))
    plt.ylim(-maxval, maxval); plt.xlim(0, N)
    plt.xlabel('Trial #'); plt.ylabel('Weights')
    
    return fig
    
    
def plot_performance(dat, xval_pL=None, sigma=50, figsize=(5, 1.5)):
    '''Plots empirical and (optional) cross-validated prediction of performance.
    
    Args:
        dat: a standard Psytrack input dataset.
        xval_pL: array of cross-validated P(y=0) for each trial in dat, the
            output of crossValidation().
        sigma: option passed to gaussian_filter controling smoothing of
            performance curve.
        figsize: size of figure.
    
    Returns:
        fig: The figure, to be modified further if necessary.
    '''

    if "correct" not in dat or "answer" not in dat:
        raise Exception("Please define a `correct` {0,1} and an `answer` {1,2} "
                        "field in `dat`.")
    
    N = len(dat['y'])
    if 2 in np.unique(dat['answer']):
        answerR = (dat['answer'] == 2).astype(float)
    else:
        answerR = (dat['answer'] == 1).astype(float)

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
                     smooth_correct - 1.96 * perf_errorbars,
                     smooth_correct + 1.96 * perf_errorbars,
                     facecolor=COLORS['emp_perf'], alpha=0.3, zorder=3)

    # Calculate the predicted accuracy
    if xval_pL is not None:
        pred_correct = np.abs(answerR - xval_pL)
        smooth_pred_correct = gaussian_filter(pred_correct, sigma)
        plt.plot(smooth_pred_correct, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if 'dayLength' in dat and dat['dayLength'] is not None:
        days = np.cumsum(dat['dayLength'])
        for d in days:
            plt.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
    
    # Add plotting details
    plt.axhline(0.5, c='k', ls='--', lw=1, alpha=0.5, zorder=1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(0, N); plt.ylim(0.3, 1.0)
    plt.xlabel('Trial #'); plt.ylabel('Performance')

    return fig


def plot_bias(dat, xval_pL=None, sigma=50, figsize=(5, 1.5)):
    '''Plots empirical and (optional) cross-validated prediction of bias.
    
    Args:
        dat: a standard Psytrack input dataset.
        xval_pL: array of cross-validated P(y=0) for each trial in dat, the
            output of crossValidation().
        sigma: option passed to gaussian_filter controling smoothing of
            performance curve.
        figsize: size of figure.
    
    Returns:
        fig: The figure, to be modified further if necessary.
    '''
    
    if "answer" not in dat:
        raise Exception("Please define an `answer` {1,2} field in `dat`.")
        
    N = len(dat['y'])
    if 2 in np.unique(dat['y']):
        choiceR = (dat['y'] == 2).astype(float)
    else:
        choiceR = (dat['y'] == 1).astype(float)
    if 2 in np.unique(dat['answer']):
        answerR = (dat['answer'] == 2).astype(float)
    else:
        answerR = (dat['answer'] == 1).astype(float)

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
                     smooth_bias - 1.96 * bias_errorbars,
                     smooth_bias + 1.96 * bias_errorbars,
                     facecolor=COLORS['emp_bias'], alpha=0.3, zorder=3)

    ### Calculate the predicted bias
    if xval_pL is not None:
        pred_bias = (1 - xval_pL) - answerR
        smooth_pred_bias = gaussian_filter(pred_bias, sigma)
        plt.plot(smooth_pred_bias, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if 'dayLength' in dat and dat['dayLength'] is not None:
        days = np.cumsum(dat['dayLength'])
        for d in days:
            plt.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
    
    # Add plotting details
    plt.axhline(0, c='k', ls='--', lw=1, alpha=0.5, zorder=1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.yticks([-0.5,0,0.5])
    plt.xlim(0, N); plt.ylim(-0.5, 0.5)
    plt.xlabel('Trial #'); plt.ylabel('Bias')

    return fig