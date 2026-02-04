import argparse
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

import figures 

import sseEPSP
import morphologyandproperty
from neck_function.plot_neckfunction import compute_axialresistance


#save as pkl file 
def save_as_pkl(fname, **kwargs) :
    joblib.dump(kwargs, fname)

def export_as_excel(fname, **kwargs) :
    df = pd.DataFrame(kwargs)
    df.to_excel(fname, index=False, engine='openpyxl')

#load excel file and preprocess for io rate curve
#new version will give two file names
def preprocess_iocurve(fname, fname2=None) :
    dataframe = pd.read_excel(fname, engine='openpyxl')

    if 'I_hz' in dataframe : #old version, single file contains firing rates for both models
        dataframe['I_hz_round'] = dataframe['I_hz'].round(1)

        #should as_indx=false to append i_hz in stats
        stats = dataframe.groupby('I_hz_round', as_index=False).agg({
            'fr_CTL': ['mean', 'max', 'min'],
            'fr_FCD': ['mean', 'max', 'min']
        })

        stats.columns = ['input_rate', 'avg_ctl', 'max_ctl', 'min_ctl', 'avg_fcd', 'max_fcd', 'min_fcd']
        return stats
    else :  #new version
        dataframe2 = pd.read_excel(fname2, engine='openpyxl')

        #header : nbasal	ihz	trial	firingrate
        dataframe['I_hz_round'] = dataframe['ihz'].round(1)
        dataframe2['I_hz_round'] = dataframe2['ihz'].round(1)

        stats_ctl = dataframe.groupby('I_hz_round').agg({'firingrate': ['mean', 'max', 'min']})
        stats_fcd = dataframe2.groupby('I_hz_round').agg({'firingrate': ['mean', 'max', 'min']})

        stats = pd.merge(stats_ctl, stats_fcd, left_index=True, right_index=True).reset_index()
        stats.columns = ['input_rate', 'avg_ctl', 'max_ctl', 'min_ctl', 'avg_fcd', 'max_fcd', 'min_fcd']
        return stats


#load excel file and preprocess for firing probability plot
def preprocess_firing_prob(fname):
    #get all sheet
    all_sheets = pd.read_excel(fname, sheet_name=None, engine='openpyxl')

    for sheet_name, df in all_sheets.items():
        if sheet_name == 'ctl' :
            x_activated_spines_ctl = df.iloc[:, 0]
            y_firing_probability_ctl = df.iloc[:, -2] / (df.shape[1]-3)
            y_stdev_ctl = df.iloc[:, -1]
            pass
        elif sheet_name == 'fcd' :
            x_activated_spines_fcd = df.iloc[:, 0]
            y_firing_probability_fcd = df.iloc[:, -2] / (df.shape[1]-3)
            y_stdev_fcd = df.iloc[:, -1]

    return {"ctl_actspines" : x_activated_spines_ctl, "ctl_prob" : y_firing_probability_ctl, "ctl_stdev" : y_stdev_ctl,
            "fcd_actspines" : x_activated_spines_fcd, "fcd_prob" : y_firing_probability_fcd, "fcd_stdev" : y_stdev_fcd }

#compute neck axial resistance based on diameter and given length
def get_neckresistance(diam_start, diam_limit) :
    diams, ra = compute_axialresistance(1.3, diam_start, diam_limit)
    return diams, ra
    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="controller for make figures")
    parser.add_argument('--nsim', type=int, default=10, help='number of runs')
    parser.add_argument('--which', type=str, default=None, 
                        choices=['attenuation', 'nbasal_ri', 'sf_ri', 'violin', 
                                 'head_effect', 'neck_effect', 'io_curve', 'firing_prob'], 
                        help='select which plot to make')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    if args.which == 'attenuation' :
        ##################################### dist vs somatic EPSP 
        #single spine evoked EPSP amplitude plot 
        dist_ctl, amp_ctl = sseEPSP.distance_from_soma_vs_somatic_amplitude(N=args.nsim, model='ctl')
        dist_fcd, amp_fcd = sseEPSP.distance_from_soma_vs_somatic_amplitude(N=args.nsim, model='fcd')
        save_as_pkl(fname='pkls/dist_vs_somaticamp.pkl', dst_ctl=dist_ctl, amp_ctl=amp_ctl, dst_fcd=dist_fcd, amp_fcd=amp_fcd)
        load_data = joblib.load('pkls/dist_vs_somaticamp.pkl')
        fit_params = {'a':0.3, 'b':-0.025, 'c':0.1, 
                    'a_':0.4, 'b_':-0.02, 'c_':0.2}
        figures.scatter_distance_from_soma_vs_somatic_amplitude(load_data, fit_params, save=False)

    elif args.which == 'nbasl_ri' :
        ##################################### group vs Ri when nbasal changed 
        nbasal_ctl, ri_ctl = morphologyandproperty.measure_inputresistance_with_various_dendrite_combination(model='ctl', min_basal=5, max_basal=6)
        nbasal_fcd, ri_fcd = morphologyandproperty.measure_inputresistance_with_various_dendrite_combination(model='fcd', min_basal=5, max_basal=6)
        save_as_pkl(fname='pkls/nbasal_ri.pkl', nbasal_ctl=nbasal_ctl, ri_ctl=ri_ctl, nbasal_fcd=nbasal_fcd, ri_fcd=ri_fcd)
        load_data = joblib.load('pkls/nbasal_ri.pkl')
        figures.box_nbasal_vs_ri(load_data, save=False)

    elif args.which == 'sf_ri' :
        #################################### spine factor(density) vs Ri
        sf, ri = morphologyandproperty.measure_inputresistance_with_various_spine_density()
        save_as_pkl(fname='pkls/sf_ri.pkl', x_sf=sf, y_ri=ri)
        load_data = joblib.load('pkls/sf_ri.pkl')
        figures.line_density_vs_RI(load_data)

    elif args.which == 'violin' :
        # ##################################### single spine evoked EPSP at head, base, soma 
        head, base, soma = sseEPSP.multipoints_EPSP_with_alteration_between_groups(model='ctl', alter_component=None, N=args.nsim)
        save_as_pkl(fname='pkls/multipoints_EPSP_baseline.pkl', head=head, base=base, soma=soma)

        head, base, soma = sseEPSP.multipoints_EPSP_with_alteration_between_groups(model='fcd', alter_component='density', N=args.nsim)
        save_as_pkl(fname='pkls/multipoints_EPSP_alterdensity.pkl', head=head, base=base, soma=soma)
        
        head, base, soma = sseEPSP.multipoints_EPSP_with_alteration_between_groups(model='fcd', alter_component='neck', N=args.nsim)
        save_as_pkl(fname='pkls/multipoints_EPSP_alterneck.pkl', head=head, base=base, soma=soma)

        head, base, soma = sseEPSP.multipoints_EPSP_with_alteration_between_groups(model='fcd', alter_component='head', N=args.nsim)
        save_as_pkl(fname='pkls/multipoints_EPSP_alterhead.pkl', head=head, base=base, soma=soma)
        
        #load from pkl
        baseline = joblib.load('pkls/multipoints_EPSP_baseline.pkl')
        alterdensity = joblib.load('pkls/multipoints_EPSP_alterdensity.pkl')
        alterneck = joblib.load('pkls/multipoints_EPSP_alterneck.pkl')
        alterhead = joblib.load('pkls/multipoints_EPSP_alterhead.pkl')

        #export simulation results as excel file
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_as_excel(fname=f'Results/sseEPSPamplitude_multipoints_{now}.xlsx', 
                        baseline_head=baseline['head'], baseline_base=baseline['base'], baseline_soma=baseline['soma'],
                        alterdensity_head=alterdensity['head'], alterdensity_base=alterdensity['base'], alterdensity_soma=alterdensity['soma'],
                        alterneck_head=alterneck['head'], alterneck_base=alterneck['base'], alterneck_soma=alterneck['soma'],
                        alterhead_head=alterhead['head'], alterhead_base=alterhead['base'], alterhead_soma=alterhead['soma'])
        
        #plotting
        figures.violin_singlespine_EPSPamp_multipoints_between_groups(baseline, alterdensity, 'density', N=args.nsim, save=args.save, show=False)
        figures.violin_singlespine_EPSPamp_multipoints_between_groups(baseline, alterneck, 'neck', N=args.nsim, save=args.save, show=False)
        figures.violin_singlespine_EPSPamp_multipoints_between_groups(baseline, alterhead, 'head', N=args.nsim, save=args.save, show=False)

    elif args.which == 'head_effect' :
        ##################################### single spine evoked EPSP, comparing between different head diameter 
        result = sseEPSP.multipoints_EPSP_with_alter_head(args.nsim)
        save_as_pkl(fname='pkls/head_effect_EPSP.pkl', **result)
        load_data = joblib.load('pkls/head_effect_EPSP.pkl')
        figures.line_headdiam_vs_amplitude_multipoints(load_data)
    
    elif args.which == 'neck_effect' : 
        ##################################### single spine evoked EPSP: deltaV (head-base), comparing between different neck diameter
        # result = sseEPSP.multipoints_EPSP_with_alter_neck(args.nsim, neck_diam_start=0.1, neck_diam_limit=0.5)
        # save_as_pkl(fname='pkls/neck_effect_EPSP2.pkl', **result)
        load_data = joblib.load('pkls/neck_effect_EPSP2.pkl')
        diams, ra = get_neckresistance(0.1, 0.5)
        figures.line_neckdiam_vs_deltaV(load_data, ra, save=args.save)
    
    elif args.which == 'io_curve' :
        # stats = preprocess_iocurve(fname="Results/FI_for_various_dendcomb.xlsx")    #old version 
        stats = preprocess_iocurve(fname="Results/merged_iorate_baseline.xlsx", fname2="Results/merged_iorate_altered.xlsx")
        figures.line_io_curve(stats, save=args.save)

    elif args.which == 'firing_prob' :
        result = preprocess_firing_prob(fname="Results/merge_sum.xlsx")
        figures.line_firingprobability(result)

