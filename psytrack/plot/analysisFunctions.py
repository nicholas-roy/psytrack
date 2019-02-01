import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from AUX.AUX_functions import read_input
import pdb


from scipy.ndimage import gaussian_filter      
def makeWeightPlot(wMode, outData, weights,
                    START=0, END=0, perf_plot=True, bias_plot=True, include_stimbias=False,
                    prediction=None, errorbar=None, add_text=False):
    
    ### Initialization
    K, N = wMode.shape
    
    if START <0: START = N + START
    if START > N: raise Exception("START > N : " + str(START) + ", " + str(N))
    if END <=0: END = N + END
    if END > N: END = N
    if START >= END: raise Exception("START >= END : " + str(START) + ", " + str(END))
    
    # Some useful values to have around
    maxval = np.max(np.abs(wMode))*1.1 # largest magnitude of any weight at any point
    min_count = int((END-START)*.03)   # minimum session length to have text fit in figure
    cumdays = np.cumsum(outData['dayLength'])
    myrange = np.arange(START,END)
    sigma = 20  # for smoothing performance and bias estimates

    # Set labels for weights based on dataset
    label_names = {'bias' : 'Bias',
              's1' : 'Tone A', 's2' : 'Tone B',
              'sL' : 'Left contrast', 'sR' : 'Right contrast',
              's_avg' : 'Avg. Tone', 'sBoth' : 'Both contrasts',
              'h' : r'Answer', 'r' : r'Reward', 'c' : r'Choice'}
    
    # Set labels ordering for legend display
    label_order = {'s1' : 0,'sL' : 0, 's2' : 1,'sR' : 1, 'sBoth' : 1,
                   'bias' : 2, 's_avg' : 3, 'h' : 4, 'c' : 5, 'r' : 6}

    # Manually set nice colors
    colors = {'bias' : '#1982C4',
              's1' : '#FF595E', 'sL' : '#FF595E',
              's2' : '#FFCA3A', 'sR' : '#FFCA3A',
              's_avg' : 'hotpink', 'sBoth' : 'hotpink',
              'h' : '#A4036F',
              'r' : '#7353BA',
              'c' : '#8AC926'}

    # Determine species of animal being plotted for title
    if "Human" in outData['dataset']: plot_title="Human "
    elif "Rat" in outData['dataset']: plot_title="Rat "
    elif "Mouse" in outData['dataset']: plot_title="Mouse "
    else: plot_title=""


    ##### Plotting |
    #####----------+
    if bias_plot and perf_plot:
        fig,axs = plt.subplots(3,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
        ax = axs[0]
    elif bias_plot or perf_plot:
        fig,axs = plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        ax = axs[0]
    else:
        fig,ax = plt.subplots(1,1,figsize=(12,4))

    
    ### Top Plot, weight trajectories
    ###----------
    plt.sca(ax)

    # Plot individual weights
    _, order = zip(*sorted(zip(weights, np.arange(K)), key=lambda t: label_order[t[0]]))

    w_count = 0
    for i in order:

        if weights[i] == weights[i-1]: w_count += 1
        else: w_count = 0
        linestyles = ['-','--',':','-.']

        # Order labels and append history #
        label = label_names[weights[i]]
        if weights[i] in ['h','r','c','s_avg']:
            label += r'$^{ %s }$' % str(-(w_count+1))

        plt.plot(wMode[i], label=label, lw=3, alpha=0.8, linestyle=linestyles[w_count], c=colors[weights[i]])

        # Plot errorbars on weights if option is passed
        if errorbar is not None: # and weights[i] in ['s1','s2','bias']:
            plt.fill_between(np.arange(len(wMode[i])), wMode[i]-2*errorbar[i], wMode[i]+2*errorbar[i], 
                             facecolor=colors[weights[i]], alpha=0.2)

    # Plot vertical session lines + write text if enough space and option passed
    for i in range(len(cumdays)):
        start = cumdays[i-1] * int(i!=0)
        end = cumdays[i]
        
        plt.axvline(start, color='grey', linestyle = '-', alpha=0.8)

        if start < START or end > END: continue
        
        if end-start > min_count and add_text:
            plt.text((end-start)/2 + start, -maxval*.97, str(end-start), 
                    horizontalalignment='center')
    
    if include_stimbias:
        cmap = plt.get_cmap('RdBu_r') #('OrRd')
        probL = outData['probL']
        i = START
        while i < END:
            _start = i
            while i+1 < END and np.linalg.norm(probL[i] - probL[i+1]) < 0.0001: i+=1
            plt.axvspan(_start, i+1, facecolor=cmap(probL[i]), alpha=0.2, edgecolor=None)
            
            # Optimal bias lines
            # if probL[i] < 0.5:
            #     plt.plot([_start,i+1], [2.208]*2, 'k--')
            # else:
            #     plt.plot([_start,i+1], [-0.829]*2, 'k--')

            i += 1

        ### Add color bar
        ax_color = fig.add_axes([0.92, 0.12, 0.02, 0.76])
        cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cmap, orientation='vertical', alpha=0.2)
        # cb1.solids.set_rasterized(True) 
        cb1.ax.set_ylabel('P(Left)', rotation=90, fontsize=16)

        plt.sca(ax)


    # Draw horizontal line at y=0
    plt.axhline(0, color='black', linestyle=':')
    
    # Make legend, format nicely
    (lines, labels) = plt.gca().get_legend_handles_labels()
    # labels, lines = zip(*sorted(zip(labels, lines), key=lambda t: label_order[t[0]]))
    # labels = [label_names[i] for i in labels]
    plt.legend(lines,labels,fontsize=14, loc='upper left', framealpha=0.7, ncol=(K+1)//2)
    
    # Add labels, adjuts other formatting
    plt.title(plot_title + outData['name'],fontsize=20)
    plt.ylabel("Weight",fontsize=18)
    plt.ylim(-maxval,maxval)
    # plt.ylim(-3.6,3.6)
    plt.xlim(START,END+1)    
    plt.tick_params(axis='both', which='major', labelsize=15, direction='inout')
    ax.set_yticks(np.arange(-int(maxval), int(maxval)+1))

    ### Test if performance or bias plot; if not, set xlabel only if this is final subplot
    if not (perf_plot or bias_plot):
        plt.xlabel("Trial #",fontsize=18)

    
    ### If performace or bias plot AND prediction data is supplied, calculate required gw values
    elif prediction is not None:
        
        xval_mask = np.ones(len(myrange)).astype(bool)

        # Data from cross-validation, arranges the inferred gw values from heldout data
        if type(prediction) is np.ndarray:
            X = np.array([i['gw'] for i in prediction]).flatten()
            test_inds = np.array([i['test_inds'] for i in prediction]).flatten()

            inrange = np.where((test_inds >= START) & (test_inds < END))[0]
            inds = [i for i in np.argsort(test_inds) if i in inrange]

            X = X[inds]

            untested_inds = [j for j in myrange if j not in test_inds]
            untested_inds = [np.where(myrange==i)[0][0] for i in untested_inds]
            xval_mask[untested_inds] = False

        # Data simply about which weights were used in the model, to reconstruct g
        elif type(prediction) is dict:
            g = read_input(outData, prediction)
            g = g[START:END]
            
            X = np.sum(g.T * wMode[:,START:END], axis=0)

        else:
            raise Exception("prediction must be an array if xval data or a dict if the weights \
                used by the best model, not type " + str(type(prediction)))

    
    ### Perforamce Plot
    ###----------
    if perf_plot:
        plt.sca(axs[1])
        
        # Calculate smooth representation of binary accuracy, plot
        raw_correct = outData['correct'][START:END].astype(float)
        smth_correct = gaussian_filter(raw_correct, sigma=sigma)
        plt.plot(myrange, smth_correct, color='red', alpha=0.4, lw=3, linestyle='-', label="Empirical")

        # Calculate errorbars on empirical performance
        QQQ = np.zeros(10001); QQQ[5000] = 1; QQQ = gaussian_filter(QQQ, sigma=sigma)
        perf_errorbars = np.sqrt(np.sum(QQQ**2)*gaussian_filter((raw_correct-smth_correct)**2, sigma=sigma))
        plt.fill_between(myrange, smth_correct - 2*perf_errorbars, smth_correct + 2*perf_errorbars, 
                             facecolor='red', alpha=0.2)

        ### If prediction data is supplied, calculate the predicted accuracy
        if prediction is not None:
            ansR = (outData['answer'][START:END]==2).astype(float)
            pL = 1/(1 + np.exp(X))
            pCor = np.abs(ansR[xval_mask] - pL)
            pred_correct = gaussian_filter(pCor, sigma=sigma)
            plt.plot(myrange[xval_mask], pred_correct, color='maroon',
                        alpha=0.8, lw=3, linestyle="-", label="Predicted")       
        

        # Plot vertical session lines + write text if enough space and option passed
        for i in range(len(cumdays)):
            start = cumdays[i-1] * int(i!=0)
            end = cumdays[i]
            
            plt.axvline(start, color='grey', linestyle = '-', alpha=0.2)

            if start < START or end > END: continue
            
            acc = np.average(outData['correct'][start:end])

            ### Make a flat line for session average
            # plt.plot([start,end], [acc]*2, color='red')

            if end-start > min_count and add_text:
                plt.text((end-start)/2 + start, 0.32, 
                     str(int(100*acc))+"%",
                     horizontalalignment='center', verticalalignment='bottom', color='red')
                    
        # Draw horizontal line at y=0.5, random choice
        plt.axhline(0.5, color='black', linestyle=':')
        
        # Add labels, adjuts other formatting
        plt.ylabel("Accuracy", fontsize=18, color='red')
        # plt.ylim(0.27,0.73)
        plt.ylim(0.3,1.0)
        plt.xlim(START,END+1)    
        plt.tick_params(axis='y', which='major', labelsize=12, 
                        direction='inout', color='red', labelcolor='red')
        plt.tick_params(axis='x', which='major', labelsize=15,
                        direction='inout', color='black', labelcolor='black')
        plt.legend(fontsize=14, loc='upper left', framealpha=0.7, ncol=2)

        # Set xlabel only if this is final subplot
        if not bias_plot:
            plt.xlabel("Trial #",fontsize=18)
  

    ### Bias Plot
    ###----------
    if bias_plot:
        plt.sca(axs[-1])

        # Calculate smooth representation of empirical bias, plot
        choiceR = (outData['y'][START:END]==2).astype(float)
        ansR = (outData['answer'][START:END]==2).astype(float)
        raw_bias = choiceR - ansR
        smth_bias = gaussian_filter(raw_bias, sigma=sigma)
        plt.plot(myrange, smth_bias, color='blue', alpha=0.4, lw=3, label="Empirical")

        # Calculate errorbars on empirical performance
        QQQ = np.zeros(10001); QQQ[5000] = 1; QQQ = gaussian_filter(QQQ, sigma=sigma)
        bias_errorbars = np.sqrt(np.sum(QQQ**2)*gaussian_filter((raw_bias-smth_bias)**2, sigma=sigma))
        plt.fill_between(myrange, smth_bias - 2*bias_errorbars, smth_bias + 2*bias_errorbars, 
                             facecolor='blue', alpha=0.2)

        ### If prediction data is supplied, calculate the predicted bias
        if prediction is not None:
            pL = 1/(1 + np.exp(X))
            pred_bias = (1-pL) - ansR[xval_mask]
            smth_pred_bias = gaussian_filter(pred_bias, sigma=sigma)
            plt.plot(myrange[xval_mask], smth_pred_bias, color='purple', alpha=0.8, lw=3, label="Predicted")


        # Plot vertical session lines + write text if enough space and option passed
        for i in range(len(cumdays)):
            start = cumdays[i-1] * int(i!=0)
            end = cumdays[i]
            
            plt.axvline(start, color='grey', linestyle = '-', alpha=0.8)

            if start < START or end > END: continue
            
            choiceR = np.average(outData['y'][start:end]==2)
            ansR = np.average(outData['answer'][start:end]==2)
        
            ### Make flat lines for session average of % trials given right vs. chosen right
            # plt.plot([start,end], [choiceR]*2, color='blue')
            # plt.fill_between([start,end], [choiceR]*2, [ansR]*2, 
            #                 color='blue', edgecolor='none', alpha=0.1)
            
            if end-start > min_count and add_text:
                bias_diff = int(100*(choiceR - ansR))
                if bias_diff >= 0:
                    bias_diff = "+" + str(bias_diff) + "%"
                else:
                    bias_diff = str(bias_diff) + "%"
                plt.text((end-start)/2 + start, -0.58, bias_diff,
                     horizontalalignment='center', verticalalignment='bottom', color='blue')


        # Draw horizontal line at y=0, no bias
        plt.axhline(0, color='black', linestyle=':')

        # Add labels, adjuts other formatting
        plt.ylabel("Bias", fontsize=18, color='blue')
        plt.ylim(-0.45,0.45) #plt.ylim(-0.6,0.6)
        plt.xlim(START,END+1)    
        plt.tick_params(axis='y', which='major', labelsize=12,
                        direction='inout', color='blue', labelcolor='blue')    
        plt.tick_params(axis='x', which='major', labelsize=15,
                        direction='inout', color='black', labelcolor='black')
        plt.legend(fontsize=14, loc='upper left', framealpha=0.7, ncol=2)

        # Final plot, so include xlabel
        plt.xlabel("Trial #",fontsize=18)


    ### Final plotting adjustments
    ###----------
    # fig.align_labels()
    # plt.tight_layout(h_pad=0.1)


from scipy.stats import sem
from matplotlib import colors as clrs
def makeComparisonPlot(models, ratname, weights, include_xval=False):
    
    colors = {'bias' : '#1982C4',
              's1' : '#FF595E',
              's2' : '#FFCA3A',
              's_avg' : 'hotpink',
              'h' : '#A4036F',
              'r' : '#7353BA',
              'c' : '#8AC926'}
    
    # Set labels for weights based on dataset
    labels = {'s_avg' : 'Avg. Stim',
              'h'  : 'Answer',
              'r'  : 'Reward',
              'c'  : 'Choice'}

    ### Extract model evidence information
    models_evd = models['results']        
    N = models_evd[0]['wMode'].shape[1]
    M = len(models_evd)
    K = len(weights)

    allEvds = [i['evd'] for i in models_evd]
    allEvds_norm = (allEvds - np.min(allEvds))/N

    ### Extract model cross-validation information
    if include_xval:
        models_xval = models['test_results']
        # completed_models = [i for i in range(len(models_xval)) if models_xval[i] is not None]
        # print(completed_models)       
        testN = len(models_xval[1][0]['gw'])

        all_LLs = [[j['logli']/testN for j in mod] for mod in models_xval]
        LLs_mean = [np.average(i) for i in all_LLs]
        LLs_mean_norm = LLs_mean - np.min(LLs_mean)
        LLs_sem = [sem(i) for i in all_LLs]

        
    ##### Plotting |
    #####----------+
    fig,[ax0,ax1] = plt.subplots(2,1,figsize=(M,4.5+0.5*K),sharex=True,
                                    gridspec_kw={'height_ratios':[4.5,0.5*K]})
    
    ### Top Plot, evidence bar plot
    ###----------
    plt.sca(ax0)
    
    xvals = np.arange(M) + 1
    plt.bar(xvals-0.5, allEvds_norm, align='edge', width=1, 
            color='#ebe9e7', edgecolor='k', linewidth=3)

    best_model = np.argmax(allEvds_norm)
    plt.plot(xvals[best_model], allEvds_norm[best_model]*.92, 'r*', markersize=18)

    up_adjust = np.max(allEvds_norm)*0.03
    plt.text(xvals[np.argmin(allEvds_norm)]-0.25, up_adjust,
             "Baseline log-Evd: " + str(round(np.min(allEvds)/N,3)), rotation=90,
            horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    # for i in range(M):
    #     if allEvds_norm[i]<np.max(allEvds_norm)*.4: continue
    #     plt.text(xvals[i]-0.25, allEvds_norm[i]-up_adjust,
    #         str(round(allEvds_norm[i],4)), rotation=90,
    #         horizontalalignment='center', verticalalignment='top', fontsize=12)
    
    plt.title(ratname,fontsize=20)
    plt.ylabel("log-Evidence / trial (norm)", fontsize=15)
    plt.ylim(np.max(allEvds_norm)*0.01,np.max(allEvds_norm)*1.1)
    plt.tick_params(axis='y', which='major', labelsize=14, direction='inout')
    plt.tick_params(axis='x', which='both', color='none')

    ### Top Plot, xval bar plot
    ###----------
    if include_xval:
        ax0b = plt.twinx(ax0)
        plt.sca(ax0b)

        xvals = np.arange(M) + 1
        plt.bar(xvals, LLs_mean_norm, #yerr=LLs_sem,
                error_kw=dict(ecolor='red', lw=2, capsize=0, alpha=0.4), 
                align='edge', width=0.5, 
                color='red', edgecolor='red', alpha=0.2, linewidth=1)

        # adj_height = allEvds_norm[np.argmin(LLs_mean_norm)] / np.max(allEvds_norm) * np.max(LLs_mean_norm)
        plt.text(xvals[np.argmin(LLs_mean_norm)]+0.25, np.max(LLs_mean_norm)*0.03,
             "Baseline xval-LL: " + str(round(np.min(LLs_mean),3)), rotation=90,
            horizontalalignment='right', verticalalignment='bottom', fontsize=12,
            color='red')

        plt.ylabel("avg. xval-LL / trial (norm)", fontsize=15, color='r')
        plt.ylim(np.max(LLs_mean_norm)*.01,np.max(LLs_mean_norm)*1.1)
        plt.tick_params(axis='y', which='major', labelsize=14, direction='inout', 
                        color='r', labelcolor='r')

    ### Bottom Plot, which weights active
    ###----------
    plt.sca(ax1)
    
    gridVals = np.zeros((K,M))
    for i in range(M):
        for j in range(K):
            used = models_evd[i]['weights'][weights[j]]
            if weights[j]=='s1' : used -= 1  # adjust for sensory weight of 2
            gridVals[j,i] = (j+1)*bool(used)
            
    
    my_cmap = clrs.ListedColormap(['white'] + [colors[i] for i in weights])
    plt.imshow(gridVals, interpolation='nearest', cmap=my_cmap, 
               extent=[0.5, M+0.5, -0.5, K-0.5], aspect='auto')
    plt.grid(which='minor', axis='both',
             linestyle='-', color='k', linewidth=2)
    
    yvals = np.arange(K)
    ax1.set_xticks(xvals)
    ax1.set_yticks(yvals)
    ax1.set_xticks(xvals[:-1]+0.5, minor=True);
    ax1.set_yticks(yvals[:-1]+0.5, minor=True);
    ax1.set_yticklabels([labels[i] for i in weights[::-1]])
    plt.xlabel("Model #", fontsize=20)
    plt.tick_params(axis='y', which='major', labelsize=15, size=0)
    plt.tick_params(axis='x', which='major', labelsize=15, size=0)
    plt.tick_params(axis='both', which='minor', color='none')

    
    ### Final plotting adjustments
    ###----------
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.02)




from scipy.stats import sem
def makeLapsePlot(model, dat, START=0, END=0, fig=None, xval=False, **kwargs):
    
    ### Initialization
    N = len(dat['y'])
    if xval: N = len(model[0]['testY'])*len(model)

    if START <0: START = N + START
    if START > N: raise Exception("START > N : " + str(START) + ", " + str(N))
    if END <=0: END = N + END
    if END > N: END = N
    if START >= END: raise Exception("START >= END : " + str(START) + ", " + str(END))
        
        
    ### Set up model parameters
    if xval:
        X = np.array([i['gw'] for i in model]).flatten()
        Y = np.array([i['testY'] for i in model]).flatten() - 1
        test_inds = np.array([i['test_inds'] for i in model]).flatten()

        inrange = np.where((test_inds >= START) & (test_inds <= END))[0]
        inds = [i for i in np.argsort(test_inds) if i in inrange]

        X = X[inds]
        Y = Y[inds]

    else:
        wMode = model['wMode']
        g = read_input(dat, model['weights'])
        g = g[START:END]
        
        Y = dat['y'][START:END] - 1
        X = np.sum(g.T * wMode, axis=0)
  
    
    ### Plotting
    start=-6; end=-start; step=.5
    edges = np.arange(start,end+step,step)

    choices = []
    for i in edges[:-1]:
        mask = (X >= i) & (X < i+step)
        choices += [Y[mask]]

    centers = edges[:-1] + step/2
    means = [np.mean(i) if len(i) > 4 else np.nan for i in choices]
    sems = [sem(i) if len(i) > 4 else np.nan for i in choices]

    if fig is None:
        fig = plt.figure(figsize=(8,4))

        xgrid = np.arange(start,end+step,.01)
        ygrid = 1/(1 + np.exp(-xgrid))
        plt.plot(xgrid, ygrid, 'k-', alpha=0.2)#, label="True Logistic")

        plt.ylim(-0.01,1.01)
        plt.xlim(start-0.1,end+0.1)
        plt.ylabel("Frac. Went Right", fontsize=18)
        # plt.title(dat['name'] + " tuning curve")

        xlabel = r"$g \cdot w$"
        # if xval: xlabel = r"cross-validated " + xlabel
        plt.xlabel(xlabel, fontsize=18)
        plt.tick_params(axis='both', labelsize=14)

        
        ax2 = plt.twinx(plt.gca())
        ax2.set_ylabel("Trial Count", fontsize=18)
        plt.tick_params(axis='both', labelsize=14)


    else:
        fig = plt.figure(fig.number)
    
    [ax1, ax2] = fig.axes
 
    plt.sca(ax1)
    plt.errorbar(centers, means, yerr=sems, alpha=1.0, fmt='o', **kwargs)
    plt.legend(loc="upper left", fontsize=14)
   
    plt.sca(ax2)
    count,_,_ = plt.hist(X, bins=edges, alpha=0.2, **kwargs)
    ymin, ymax = ax2.get_ylim()
    plt.ylim(0,np.max([ymax, np.max(count)/0.7]))
    
    return fig






def compare(dat, w, day_thresh=50):
    
    K, N = w.shape
    labels = ['Bias','Left Contrast','Right Contrast','History']
    
    days = dat['dayLength']
    validDays = days>20
    cumm_days = np.hstack(([0],np.cumsum(days)))
    starts = (cumm_days[:-1])[validDays]
    ends = (cumm_days[1:] - 1)[validDays]
    
    for i in range(K):
        
        withinDay = w[i][ends] - w[i][starts]
        betweenDay = w[i][starts[1:]] - w[i][ends[:-1]]
        
        plt.figure()
        plt.plot(withinDay[:-1],betweenDay, 'o')
        plt.axis([-4,4,-4,4])
        plt.axhline(0,linestyle='--', color='black')
        plt.axvline(0,linestyle='--', color='black')
        plt.title(labels[i])
        plt.xlabel("within Day"); plt.ylabel("between Day")




# if split_perf:
#     LeftTrials = outData['answer'][start:end]==1
#     accL = np.average(outData['correct'][start:end][LeftTrials])
#     accR = np.average(outData['correct'][start:end][~LeftTrials])
#     plt.plot([start,end], [accL]*2, color='orange', linestyle='--', lw=4)
#     plt.plot([start,end], [accR]*2, color='purple', linestyle='--', lw=4)
    
#     plt.text((end-start)/2 + start - 50, 0.98, 
#          str(int(100*accL))+"%",
#          horizontalalignment='center', verticalalignment='top', color='orange')

#     plt.text((end-start)/2 + start + 50, 0.98, 
#          str(int(100*accR))+"%",
#          horizontalalignment='center', verticalalignment='top', color='purple')