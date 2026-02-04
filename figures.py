from matplotlib import pyplot as plt
import numpy as np

# line plot for firing probability when activate multiple spines
def line_firingprobability(data, save=False, xpeak=[40, 175]) :
    x_ctl_nspines = data['ctl_actspines']
    y_ctl_prob = data['ctl_prob']
    y_ctl_stdev = data['ctl_stdev']
    # x_fcd_nspines = data['fcd_actspines']
    y_fcd_prob = data['fcd_prob']
    y_fcd_stdev = data['fcd_stdev']

    fig, (ax_fcd, ax_ctl) = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0.05)  # adjust space between Axes

    ax_ctl.fill_between(x_ctl_nspines, np.clip(y_ctl_prob-y_ctl_stdev, 0, None), np.clip(y_ctl_prob+y_ctl_stdev, None, 1), color='cyan', alpha=.55, linewidth=1)  
    ax_ctl.plot(x_ctl_nspines, y_ctl_prob, linewidth=2, color='darkcyan', label="Baseline")
    ax_fcd.plot(x_ctl_nspines, y_ctl_prob, linewidth=2, color='darkcyan', label="Baseline")
    
    ax_fcd.fill_between(x_ctl_nspines, np.clip(y_fcd_prob-y_fcd_stdev, 0, None), np.clip(y_fcd_prob+y_fcd_stdev, None, 1), color='magenta', alpha=.55, linewidth=1)
    ax_fcd.plot(x_ctl_nspines, y_fcd_prob, linewidth=2, color='darkmagenta', label="Altered")
    ax_ctl.plot(x_ctl_nspines, y_fcd_prob, linewidth=2, color='darkmagenta', label="Altered")  #just for the legend

    
    # xpeak = [40,175]
    ax_fcd.set_xlim(xpeak[0]-10, xpeak[0]+10)  
    ax_ctl.set_xlim(xpeak[1]-10, xpeak[1]+10)  
    ax_fcd.set_ylim(0, 1)
    ax_ctl.set_ylim(0, 1)

    ax_fcd.spines.right.set_visible(False)
    ax_ctl.spines.left.set_visible(False)
    ax_fcd.yaxis.tick_left()
    ax_ctl.yaxis.tick_right()

    ax_fcd.set_xticks([xpeak[0]-5, xpeak[0]+5], labels=[str(xpeak[0]-5), str(xpeak[0]+5)], fontsize=12) 
    ax_ctl.set_xticks([xpeak[1]-5, xpeak[1]+5], labels=[str(xpeak[1]-5), str(xpeak[1]+5)], fontsize=12) 
    ax_fcd.set_yticks([0, 1.0], labels=["0.0", "1.0"], fontsize=12)
    
    d = .2  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=10,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_fcd.plot([1, 1], [0, 1], transform=ax_fcd.transAxes, **kwargs)
    ax_ctl.plot([0, 0], [0, 1], transform=ax_ctl.transAxes, **kwargs)

    ax_fcd.set_ylabel("Firing Probability", fontsize=14)
    ax_fcd.set_xlabel('Number of Activated Spines', fontsize=14)
    ax_fcd.xaxis.set_label_coords(0.5, 0.05, transform=fig.transFigure)

    if save :
        plt.savefig(f"syn_activation.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    plt.show()  

# line plot for io rate curve
def line_io_curve(stats_df, save=False) :
    plt.figure()
    
    # Control 
    plt.plot(stats_df['input_rate'], stats_df['avg_ctl'], label='Baseline', color='darkcyan')
    plt.fill_between(stats_df['input_rate'], stats_df['min_ctl'], stats_df['max_ctl'], color='cyan', alpha=.55, linewidth=1)
    
    # FCD 
    plt.plot(stats_df['input_rate'], stats_df['avg_fcd'], label='Altered', color='darkmagenta')
    plt.fill_between(stats_df['input_rate'], stats_df['min_fcd'], stats_df['max_fcd'], color='magenta', alpha=.55, linewidth=1)
    
    plt.ylabel("Firing Rate (Hz)", fontsize=14)
    plt.ylim((0, 170))
    plt.xlim((0.2,5))
    plt.yticks(ticks=[0, 50, 100, 150], labels=["0", "50", "100", "150"], fontsize=12)
    plt.xlabel("Input Frequency (Hz)", fontsize=14)
    plt.xticks(ticks=[0.2, 1.0, 2.0, 3.0, 4.0, 5.0], labels=["0.2", "1.0", "2.0", "3.0", "4.0", "5.0"], fontsize=12)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False)#,         # ticks along the top edge are off
    if save :
        plt.savefig(f"io_curve.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    plt.show()  

# line plot about deltaV(head-base) when neck diameter changed
def line_neckdiam_vs_deltaV(data, ra, save=False, show=True, legend=False) :
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # neck_range_min = data['range'][0]
    # neck_range_max = data['range'][1]
    neck_diameter = data['neck_diameter']

    deltaV_min = data['deltaV_min']
    deltaV_max = data['deltaV_max']

    fig, ax = plt.subplots()

    # plt.fill_between(neck_diameter, deltaV_min, deltaV_max, color="lightgrey", alpha=0.9)

    cmap = cm.Reds
    # norm = Normalize(vmin=min(ra), vmax=max(ra))
    norm = Normalize(vmin=0, vmax=max(ra))
    for i in range(len(neck_diameter) - 1):
        res_val = (ra[i] + ra[i+1]) / 2
        color = cmap(norm(res_val))
        ax.fill_between(neck_diameter[i:i+2], deltaV_min[i:i+2], deltaV_max[i:i+2], color=color, edgecolor=None)

    plt.xlabel("Spine Neck Diameter (µm)", fontsize=14)
    plt.ylabel('ΔV (Head - Base, mV)', fontsize=14)
    plt.ylim((0, max(deltaV_max)))
    plt.xlim((0.1, 0.5))
    plt.yticks(ticks=[0, 5], labels=["0", "5"], fontsize=12)
    plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5], labels=["0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=12)

    plt.axvline(x = 0.15, ymin = 0, ymax = deltaV_max[neck_diameter.index(0.15)]/max(deltaV_max), color = 'cyan', label = 'Baseline', linestyle='-')
    plt.axvline(x = 0.3, ymin = 0, ymax = deltaV_max[neck_diameter.index(0.3)]/max(deltaV_max), color = 'magenta', label = 'Altered', linestyle='-')

    plt.scatter(0.3, deltaV_max[neck_diameter.index(0.3)], s=20, c='magenta', marker='X')
    plt.scatter(0.15, deltaV_max[neck_diameter.index(0.15)], s=20, c='cyan', marker='X')
    if legend :
        plt.text(0.3, deltaV_max[neck_diameter.index(0.3)], 'Altered', fontdict={'size': 9, 'color': 'magenta'})
        plt.text(0.15, deltaV_max[neck_diameter.index(0.15)], 'Baseline', fontdict={'size': 9, 'color': 'cyan'})

    # Force all spines on
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    ax_ins = inset_axes(ax, width="5%", height="40%", loc='upper right', borderpad=2)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 

    cbar = fig.colorbar(sm, cax=ax_ins)
    cbar.set_label('Neck Resistance (MΩ)')
    # cbar.ax.yaxis.set_label_position('left') 
    cbar.ax.yaxis.set_ticks_position('left')  

    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 330])

    if save :
        plt.savefig(f"neckdiam_deltaV.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    if show :
        plt.show()

# line plot about multipoints EPSP amplitude (head, base, soma) when head diameter(fixed neck length) changed 
def line_headdiam_vs_amplitude_multipoints(data, save=False) :
    # head_diam_range_min = data['range'][0]
    # head_diam_range_max = data['range'][1]
    #x axis
    head_diameters = data['head_diameter']

    #averaged values
    head_ = data['head']
    base_ = data['base']
    soma_ = data['soma']

    #min and max for shading
    head_min = data['head_min']
    base_min = data['base_min']
    soma_min = data['soma_min']

    head_max = data['head_max']
    base_max = data['base_max']
    soma_max = data['soma_max']

    #plot 그리기
    fig, (ax_spine, ax_soma) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [0.7, 0.3]})
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

    ax_spine.plot(head_diameters, head_, color="red",label="Head")
    ax_spine.plot(head_diameters, base_, color="blue", label="Base")
    ax_spine.plot(head_diameters, soma_, color="black", label="Soma")

    ax_soma.plot(head_diameters, head_, color="red",label="Head")
    ax_soma.plot(head_diameters, base_, color="blue", label="Base")
    ax_soma.plot(head_diameters, soma_, color="black", label="Soma")

    ax_spine.fill_between(head_diameters, head_min, head_max, color="red", alpha=0.3)
    ax_spine.fill_between(head_diameters, base_min, base_max, color="blue", alpha=0.3)
    ax_spine.fill_between(head_diameters, soma_min, soma_max, color="black", alpha=0.3)

    ax_soma.fill_between(head_diameters, head_min, head_max, color="red", alpha=0.3)
    ax_soma.fill_between(head_diameters, base_min, base_max, color="blue", alpha=0.3)
    ax_soma.fill_between(head_diameters, soma_min, soma_max, color="black", alpha=0.3)

    
    ax_spine.spines.bottom.set_visible(False)

    ax_soma.xaxis.tick_bottom()
    ax_soma.spines.top.set_visible(False)

    ax_spine.set_xticks(ticks=[])
    ax_spine.set_yticks(ticks=[3,5,7], labels=["3", "5", "7"])
    ax_spine.set_ylim([1,9])
    

    ax_soma.set_xticks(ticks=[0.2, 0.5, 1.0, 1.5], labels=["0.2", "0.5", "1.0", "1.5"], fontsize=12)
    ax_soma.set_yticks(ticks=[0, 0.4], labels=["0", "0.4"], fontsize=12)
    ax_soma.set_ylim([0, 0.5])
    
    ax_spine.axvline(x = 0.55, ymin = 0, ymax = 1, color = 'cyan', linestyle='-')
    ax_spine.axvline(x = 0.70, ymin = 0, ymax = 1, color = 'magenta',linestyle='-')
    ax_soma.axvline(x = 0.55, ymin = 0, ymax = 1, color = 'cyan', linestyle='-')
    ax_soma.axvline(x = 0.70, ymin = 0, ymax = 1, color = 'magenta', linestyle='-')


    d = .5 # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=10,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_spine.plot([0, 1], [0, 0], transform=ax_spine.transAxes, **kwargs)
    ax_soma.plot([0, 1], [1, 1], transform=ax_soma.transAxes, **kwargs)

    ax_soma.set_xlabel("Spine Head Diameter (µm)", fontsize=14)
    ax_spine.legend(loc="upper center", ncol=3, fontsize=12)

    fig.text(0.04, 0.5, 'EPSP Amplitude (mV)', va='center', rotation='vertical', fontsize=14)

    if save :
        plt.savefig(f"head_diam_amplitude.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)
    plt.show()

# violin plot for head, base, soma EPSP amplitude between groups
def violin_singlespine_EPSPamp_multipoints_between_groups(control_data, altered_data, alter_what, N=100, save=False, show=True, legend=False) :
    amp_head_ctl = control_data['head']
    amp_base_ctl = control_data['base']
    amp_soma_ctl = control_data['soma']

    amp_head_fcd = altered_data['head']
    amp_base_fcd = altered_data['base']
    amp_soma_fcd = altered_data['soma']

    #plot 그리기
    fig, ax = plt.subplots(figsize=(4,4))

    # #Head, Base, Soma position base
    box_colors = ["cyan", "magenta", "cyan", "magenta", "cyan", "magenta"]
    bplot1 = ax.violinplot([amp_head_ctl, amp_head_fcd], 
                           positions=[1,2], widths=0.7, showmedians=True)
    bplot3 = ax.violinplot([amp_base_ctl, amp_base_fcd], 
                           positions=[3,4], widths=0.7, showmedians=True)

    bplot1['cmaxes'].set_color("black")
    bplot1['cmins'].set_color("black")
    bplot1['cmedians'].set_color('black') 

    bplot3['cmaxes'].set_color("black")
    bplot3['cmins'].set_color("black")  
    bplot3['cmedians'].set_color('black') 
    
    ax.vlines(1, min(amp_head_ctl), max(amp_head_ctl), color="black", linestyle='-', lw=1.5)
    ax.vlines(2, min(amp_head_fcd), max(amp_head_fcd), color="black", linestyle='-', lw=1.5)
    ax.vlines(3, min(amp_base_ctl), max(amp_base_ctl), color="black", linestyle='-', lw=1.5)
    ax.vlines(4, min(amp_base_fcd), max(amp_base_fcd), color="black", linestyle='-', lw=1.5)
    for pc, color in zip(bplot1['bodies'], box_colors[0:2]):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.5)
    for pc, color in zip(bplot3['bodies'], box_colors[2:4]):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.5)
        
    if legend :
        plt.legend(["Baseline","Altered"], loc="lower left")

    ax.set_xticks([1.5, 3.5])
    ax.set_xticklabels(["Head", "Base"], fontsize=12)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)

    ax.set_ylabel('EPSP Amplitude (mV)', fontsize=14)
    ax.set_yticks(ticks=[0, 2, 4, 6, 8, 10], labels=["0", "2", "4", "6", "8", "10"], fontsize=12)
    ax.set_ylim([0, 10.5])
    # if alter == "density" :
    #     ax.set_ylim([0, 10.5])
    ax.set_title('Spine', fontsize=14)
    
    # Force all spines on
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    plt.savefig(f"violin_{alter_what}_ampl_spine.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    fig, ax2 = plt.subplots(figsize=(2,4))
    bplot2 = ax2.violinplot([amp_soma_ctl, amp_soma_fcd], positions=[5,6], widths=0.7, showmedians=True)
    bplot2['cmaxes'].set_color("black")
    bplot2['cmins'].set_color("black")
    bplot2['cmedians'].set_color('black')
    ax2.vlines(5, min(amp_soma_ctl), max(amp_soma_ctl), color="black", linestyle='-', lw=1.5)
    ax2.vlines(6, min(amp_soma_fcd), max(amp_soma_fcd), color="black", linestyle='-', lw=1.5)
    for pc, color in zip(bplot2['bodies'], box_colors[4:6]):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.5)


    ax2.set_yticks(ticks=[0, 0.2, 0.4], labels=["0", "0.2", "0.4"], fontsize=12)
    ax2.set_xticks([])
    ax2.set_ylim([0, 0.45])
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.set_title('Soma', fontsize=14)
    # Set the y-axis label position to the right
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()


    # Force all spines on
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    if save :
        plt.savefig(f"violin_{alter_what}_ampl_soma.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    if show :
        plt.show()

#line plot with scatter overlay for input resistance when differing spine factor 
# spine density vs input resistance
def line_density_vs_RI(data, save=False, show=True, legend=False) :
    x = data['x_sf']
    y = data['y_ri']

    fig, ax = plt.subplots()
    plt.plot(x, y, color="black", linewidth=2.5)

    plt.ylim((70, 230))
    # Set the y-axis label position to the right
    ax.yaxis.set_label_position("right")

    # Move the y-axis ticks and tick labels to the right
    ax.yaxis.tick_right()

    ax.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=True)
    plt.yticks(ticks=[100, 150, 200], labels=["100", "150", "200"], fontsize=12)

    plt.xticks(ticks=[x[0], x[-1]], labels=["Sparse", "Dense"], fontsize=12)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False)#),         # ticks along the top edge are off

    # Force all spines on
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    #flip X axis 
    plt.gca().invert_xaxis()
        
    plt.scatter(1.2, y[x.index(1.20)], s=120, c='magenta', marker='X')
    plt.scatter(2.0, y[x.index(2.00)], s=120, c='cyan', marker='X')

    if legend :
        plt.text(2.0, y[x.index(2.00)], 'Baseline', fontdict={'size': 9, 'color': 'cyan'})
        plt.text(1.2, y[x.index(1.20)], 'Altered', fontdict={'size': 9, 'color': 'magenta'})

    plt.xlabel("Spine Density", fontsize=14)
    plt.ylabel("Input Resistance (MΩ)", fontsize=14)
    
    if save:
        plt.savefig("density_RI.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    if show :
        plt.show()

#boxplot for input resistance when differing nbasal
def box_nbasal_vs_ri(data, save=False): 
    y_ctl = data["ri_ctl"]
    y_fcd = data["ri_fcd"]


    fig, ax = plt.subplots(figsize=(2,2))

    bplot = ax.boxplot([y_ctl, y_fcd], widths=0.5, medianprops=dict(color="black"), patch_artist=True)  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot['boxes'], ["cyan", "magenta"]):
        patch.set_facecolor(color)
    plt.ylim((80, 170))
    ax.set_yticks(ticks=[100, 140], labels=["100", "140"], fontsize=12)
    plt.xticks([])
    if save :
        plt.savefig(f"ri_box.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    plt.show()

# distance from soma vs somatic amplitude 
# scatter plot and fitting line
# does not include dendrite information
# should adjust nseg of the SF dendrite by dend_nseglevel
# upper line from basal dendrite, lower line from apical dendrite 
def scatter_distance_from_soma_vs_somatic_amplitude(data, fit_params, save=False): 
    import numpy as np
    from scipy.optimize import curve_fit

    #unpakcing
    dist_spine_ctl = data['dst_ctl']
    dist_spine_fcd = data['dst_fcd']
    amplitude_ctl = data['amp_ctl']
    amplitude_fcd = data['amp_fcd']

    a = fit_params['a']
    b = fit_params['b']
    c = fit_params['c']
    a_ = fit_params['a_']
    b_ = fit_params['b_']
    c_ = fit_params['c_']


    fig, ax = plt.subplots()

    plt.scatter(dist_spine_ctl, amplitude_ctl, marker="o", s=20, color="cyan", alpha=0.7, label="Baseline")
    plt.scatter(dist_spine_fcd, amplitude_fcd, marker="o", s=20, color="magenta", alpha=0.7, label="Altered")


    ## exponential fitting 
    # Define the exponential function
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) +c

    # Fit the curve
    params, covar_ = curve_fit(exp_func, dist_spine_ctl, amplitude_ctl, p0=[a, b, c])
    fit_a, fit_b, fit_c = params
    x_fit = np.linspace(min(dist_spine_ctl), max(dist_spine_ctl), 1000) # for a smooth curve
    y_fit = exp_func(x_fit, fit_a, fit_b, fit_c)
    plt.plot(x_fit, y_fit, color='darkcyan')
    print(f'Baseline: y={fit_a:.4f}e^({fit_b:.5f}x)+{fit_c:.4f}')

    # Fit the curve
    params_, covar_ = curve_fit(exp_func, dist_spine_fcd, amplitude_fcd, p0=[a_, b_, c_])
    fit_a_fcd, fit_b_fcd, fit_c_fcd = params_
    y_fit_ = exp_func(x_fit, fit_a_fcd, fit_b_fcd, fit_c_fcd)
    plt.plot(x_fit, y_fit_, color='darkmagenta')
    print(f'Altered: y={fit_a_fcd:.4f}e^({fit_b_fcd:.5f}x)+{fit_c_fcd:.4f}')

    plt.xlabel("Spine Location Relative to Soma (µm)", fontsize=14)
    plt.ylabel("Somatic EPSP Amplitude (mV)", fontsize=14)

    plt.xlim((30, 390))
    plt.ylim((0.1, 0.42))

    ax.set_xticks(ticks=[100, 200, 300], labels=["100", "200", "300"], fontsize=12)
    ax.set_yticks(ticks=[0.1, 0.2, 0.3, 0.4], labels=["0.1", "0.2", "0.3", "0.4"], fontsize=12)
    plt.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        right=False)
    # if legend :
    #     plt.legend(loc="upper right")

    # Force all spines on
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    if save :
        plt.savefig(f"distance_vs_soma_amp.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)

    plt.show()


if __name__ == "__main__" :

    pass