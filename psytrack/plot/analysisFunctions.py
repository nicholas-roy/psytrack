import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from psytrack.aux.auxFunctions import read_input


def makeWeightPlot(wMode,
                   outData,
                   weights_dict,
                   START=0,
                   END=0,
                   perf_plot=False,
                   bias_plot=False,
                   prediction=None,
                   errorbar=None):
    """Generates a visualization of weights and related plots
    
    Displays a plot (returns nothing) of the weights over trials. Optionally,
    include a plot of bias and/or performance over time as accompanying
    subplots. Also show errorbars on weights optionally.
    
    Args:
        wMode (array): the weights to plot
        outData (dict): the dataset generating the weights
        weights_dict (dict): the weights in the dataset that were fit
        START (int): Trial to start plotting from
        END (int): Trial to stop plotting, 0 is end, negative accepted
        perf_plot (bool): Show a subplot with performance tracked over time
        bias_plot (bool): Show a subplot with choice bias tracked over time
        prediction (array OR dict): array corresponds to xval gw values from
            which to infer performance or bias. dict means infer from
            non-cross-validated weights instead. Overlaid on perf and bias plot
        errorbar (array): size of errorbars to include on each weight
    """

    ### Initialization
    K, N = wMode.shape

    if START < 0: START = N + START
    if START > N: raise Exception("START > N : " + str(START) + ", " + str(N))
    if END <= 0: END = N + END
    if END > N: END = N
    if START >= END:
        raise Exception("START >= END : " + str(START) + ", " + str(END))

    # Some useful values to have around
    maxval = np.max(np.abs(wMode)) * 1.1  # largest magnitude of any weight
    cumdays = np.cumsum(outData['dayLength'])
    myrange = np.arange(START, END)
    sigma = 20  # for smoothing performance and bias estimates

    # Custom labels for weights based on dataset
    label_names = {
        'bias': 'Bias',
        's1': 'Tone A',
        's2': 'Tone B',
        'sL': 'Left contrast',
        'sR': 'Right contrast',
        's_avg': 'Avg. Tone',
        'sBoth': 'Both contrasts',
        'h': r'Answer',
        'r': r'Reward',
        'c': r'Choice'
    }

    # Set label ordering for legend display
    label_order = {
        's1': 0,
        'sL': 0,
        's2': 1,
        'sR': 1,
        'sBoth': 1,
        'bias': 2,
        's_avg': 3,
        'h': 4,
        'c': 5,
        'r': 6
    }

    # Manually set good colors
    colors = {
        'bias': '#1982C4',
        's1': '#FF595E',
        'sL': '#FF595E',
        's2': '#FFCA3A',
        'sR': '#FFCA3A',
        's_avg': 'hotpink',
        'sBoth': 'hotpink',
        'h': '#A4036F',
        'r': '#7353BA',
        'c': '#8AC926'
    }

    # Determine species of animal being plotted for title
    if 'dataset' not in outData: plot_title = ""
    elif "Human" in outData['dataset']: plot_title = "Human "
    elif "Rat" in outData['dataset']: plot_title = "Rat "
    elif "Mouse" in outData['dataset']: plot_title = "Mouse "
    else: plot_title = ""

    ##### Plotting |
    #####----------+
    if bias_plot and perf_plot:
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={'height_ratios': [2, 1, 1]})
        ax = axs[0]
    elif bias_plot or perf_plot:
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(12, 6),
            sharex=True,
            gridspec_kw={'height_ratios': [2, 1]})
        ax = axs[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Turn weights dict into weights list
    weights = []
    for i in sorted(weights_dict.keys()):
        weights += [i]*weights_dict[i]
    

    ### Top Plot, weight trajectories
    ###----------
    plt.sca(ax)

    # Plot individual weights, arrange according to order specified above
    _, order = zip(
        *sorted(zip(weights, np.arange(K)), key=lambda t: label_order[t[0]]))

    w_count = 0
    for i in order:

        if weights[i] == weights[i - 1]: w_count += 1
        else: w_count = 0
        linestyles = ['-', '--', ':', '-.']

        # Order labels and append history # if needed
        label = label_names[weights[i]]
        if weights[i] in ['h', 'r', 'c', 's_avg']:
            label += r'$^{ %s }$' % str(-(w_count + 1))

        plt.plot(
            wMode[i],
            label=label,
            lw=3,
            alpha=0.8,
            linestyle=linestyles[w_count],
            c=colors[weights[i]])

        # Plot errorbars (2SD) on weights if option is passed
        if errorbar is not None:
            plt.fill_between(
                np.arange(len(wMode[i])),
                wMode[i] - 2 * errorbar[i],
                wMode[i] + 2 * errorbar[i],
                facecolor=colors[weights[i]],
                alpha=0.2)

    # Plot vertical session lines + write text if enough space and option passed
    for i in range(len(cumdays)):
        start = cumdays[i - 1] * int(i != 0)
        plt.axvline(start, color='grey', linestyle='-', alpha=0.4)

    # Draw horizontal line at y=0
    plt.axhline(0, color='black', linestyle=':')

    # Make legend, format nicely
    (lines, labels) = plt.gca().get_legend_handles_labels()
    plt.legend(
        lines,
        labels,
        fontsize=14,
        loc='upper left',
        framealpha=0.7,
        ncol=(K + 1) // 2)

    # Add labels, adjuts other formatting
    plt.title(plot_title + outData['name'], fontsize=20)
    plt.ylabel("Weight", fontsize=18)
    plt.ylim(-maxval, maxval)
    plt.xlim(START, END + 1)
    plt.tick_params(
        axis='both', which='major', labelsize=15, direction='inout')
    ax.set_yticks(np.arange(-int(maxval), int(maxval) + 1))

    ### Test if performance or bias plot; if not, set xlabel only if this is final subplot
    xval_mask = np.ones(len(myrange)).astype(bool)

    if not (perf_plot or bias_plot):
        plt.xlabel("Trial #", fontsize=18)

    ### Data from cross-validation, arranges the inferred gw values from heldout data
    elif prediction is not None:

        X = np.array([i['gw'] for i in prediction]).flatten()
        test_inds = np.array([i['test_inds'] for i in prediction]).flatten()

        inrange = np.where((test_inds >= START) & (test_inds < END))[0]
        inds = [i for i in np.argsort(test_inds) if i in inrange]

        X = X[inds]

        untested_inds = [j for j in myrange if j not in test_inds]
        untested_inds = [np.where(myrange == i)[0][0] for i in untested_inds]
        
        xval_mask[untested_inds] = False

    # Data simply about which weights were used in the model, to reconstruct g
    else:
        g = read_input(outData, weights_dict)
        g = g[START:END]

        X = np.sum(g.T * wMode[:, START:END], axis=0)


    ### Perforamce Plot
    ###----------
    if perf_plot:
        plt.sca(axs[1])

        # Calculate smooth representation of binary accuracy, plot
        raw_correct = outData['correct'][START:END].astype(float)
        smth_correct = gaussian_filter(raw_correct, sigma=sigma)
        plt.plot(
            myrange,
            smth_correct,
            color='red',
            alpha=0.4,
            lw=3,
            linestyle='-',
            label="Empirical")

        # Calculate errorbars on empirical performance
        QQQ = np.zeros(10001)
        QQQ[5000] = 1
        QQQ = gaussian_filter(QQQ, sigma=sigma)
        perf_errorbars = np.sqrt(
            np.sum(QQQ**2) * gaussian_filter(
                (raw_correct - smth_correct)**2, sigma=sigma))
        plt.fill_between(
            myrange,
            smth_correct - 2 * perf_errorbars,
            smth_correct + 2 * perf_errorbars,
            facecolor='red',
            alpha=0.2)

        ### If prediction data is supplied, calculate the predicted accuracy
        ansR = (outData['answer'][START:END] == 2).astype(float)
        pL = 1 / (1 + np.exp(X))
        pCor = np.abs(ansR[xval_mask] - pL)
        pred_correct = gaussian_filter(pCor, sigma=sigma)
        plt.plot(
            myrange[xval_mask],
            pred_correct,
            color='maroon',
            alpha=0.8,
            lw=3,
            linestyle="-",
            label="Predicted")

        # Plot vertical session lines + write text if enough space and option passed
        for i in range(len(cumdays)):
            start = cumdays[i - 1] * int(i != 0)
            plt.axvline(start, color='grey', linestyle='-', alpha=0.4)

        # Draw horizontal line at y=0.5, random choice
        plt.axhline(0.5, color='black', linestyle=':')

        # Add labels, adjuts other formatting
        plt.ylabel("Accuracy", fontsize=18, color='red')
        plt.ylim(0.3, 1.0)
        plt.xlim(START, END + 1)
        plt.tick_params(
            axis='y',
            which='major',
            labelsize=12,
            direction='inout',
            color='red',
            labelcolor='red')
        plt.tick_params(
            axis='x',
            which='major',
            labelsize=15,
            direction='inout',
            color='black',
            labelcolor='black')
        plt.legend(fontsize=14, loc='upper left', framealpha=0.7, ncol=2)

        # Set xlabel only if this is final subplot
        if not bias_plot:
            plt.xlabel("Trial #", fontsize=18)

    ### Bias Plot
    ###----------
    if bias_plot:
        plt.sca(axs[-1])

        # Calculate smooth representation of empirical bias, plot
        choiceR = (outData['y'][START:END] == 2).astype(float)
        ansR = (outData['answer'][START:END] == 2).astype(float)
        raw_bias = choiceR - ansR
        smth_bias = gaussian_filter(raw_bias, sigma=sigma)
        plt.plot(
            myrange,
            smth_bias,
            color='blue',
            alpha=0.4,
            lw=3,
            label="Empirical")

        # Calculate errorbars on empirical performance
        QQQ = np.zeros(10001)
        QQQ[5000] = 1
        QQQ = gaussian_filter(QQQ, sigma=sigma)
        bias_errorbars = np.sqrt(
            np.sum(QQQ**2) * gaussian_filter(
                (raw_bias - smth_bias)**2, sigma=sigma))
        plt.fill_between(
            myrange,
            smth_bias - 2 * bias_errorbars,
            smth_bias + 2 * bias_errorbars,
            facecolor='blue',
            alpha=0.2)

        ### If prediction data is supplied, calculate the predicted bias
        pL = 1 / (1 + np.exp(X))
        pred_bias = (1 - pL) - ansR[xval_mask]
        smth_pred_bias = gaussian_filter(pred_bias, sigma=sigma)
        plt.plot(
            myrange[xval_mask],
            smth_pred_bias,
            color='purple',
            alpha=0.8,
            lw=3,
            label="Predicted")

        # Plot vertical session lines + write text if enough space and option passed
        for i in range(len(cumdays)):
            start = cumdays[i - 1] * int(i != 0)
            plt.axvline(start, color='grey', linestyle='-', alpha=0.4)

        # Draw horizontal line at y=0, no bias
        plt.axhline(0, color='black', linestyle=':')

        # Add labels, adjuts other formatting
        plt.ylabel("Bias", fontsize=18, color='blue')
        plt.ylim(-0.45, 0.45)  #plt.ylim(-0.6,0.6)
        plt.xlim(START, END + 1)
        plt.tick_params(
            axis='y',
            which='major',
            labelsize=12,
            direction='inout',
            color='blue',
            labelcolor='blue')
        plt.tick_params(
            axis='x',
            which='major',
            labelsize=15,
            direction='inout',
            color='black',
            labelcolor='black')
        plt.legend(fontsize=14, loc='upper left', framealpha=0.7, ncol=2)

        # Final plot, so include xlabel
        plt.xlabel("Trial #", fontsize=18)

    ### Final plotting adjustments
    ###----------
    fig.align_labels()
    plt.tight_layout(h_pad=0.1)