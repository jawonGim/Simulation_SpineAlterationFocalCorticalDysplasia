from neuron import gui, h
from neuron.units import ms, sec, mV
import morphology

import time
import random

import argparse
from datetime import datetime
from controller import save_as_pkl

from conductance_distribution import get_baseline_weight_distribution, get_altered_weight_distribution


#measure firing rate by deliver signals on spine heads with specific weight
def measure_firing_rate(cell, stim_hz, weight_ampa, weight_nmda, vinit=-70*mV, dt=0.1*ms, tstop=12000*ms) :
    # stim_start = 0 *ms
    stim_intv = 1_000 * ms / stim_hz
    stim_num = tstop / stim_intv
    drop_both = stim_intv*2

    drop_both = min(drop_both, 500*ms)  # ensure drop_both maximum is 500ms

    random.seed(time.time())
    stims = []
    ncl_ampa = []
    ncl_nmda = []
    #set stimulation 
    for ampa, nmda, w1, w2 in zip(cell.ampar, cell.nmdar, weight_ampa, weight_nmda) :
        stims.append(h.NetStim())
        stims[-1].start = random.randrange(0, int(stim_intv))
        stims[-1].interval = stim_intv
        stims[-1].number = stim_num 
        stims[-1].noise = 1 

        ncl_ampa.append(h.NetCon(stims[-1], ampa))    
        ncl_ampa[-1].weight[0] = w1
        ncl_ampa[-1].delay = 0 * ms
    
        ncl_nmda.append(h.NetCon(stims[-1], nmda))
        ncl_nmda[-1].weight[0] = w2
        ncl_nmda[-1].delay = 0 * ms

    h.dt = dt
    h.v_init = vinit
    h.finitialize(vinit)
    h.continuerun(tstop)

    firing = list(cell.spike_times)

    cropped = [sp for sp in firing if sp < tstop-drop_both and sp > drop_both]
    if len(cropped) == 0 :
        fr = 0
    else :
        fr = len(cropped)/(tstop-drop_both*2)*sec
    return fr


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="batch processing of simulation for IO-rate curve")
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'altered'], help='select model (baseline or altered)')
    parser.add_argument('--ihz', nargs=3, type=float, help='input hz range (start, tick, end)')
    parser.add_argument('--nbasal', nargs=2, type=int, help='input nbasal range (start, end)')
    args = parser.parse_args()


    if args.model == 'baseline' :
        cellspec = dict()
        
        ampaw, nmdaw = get_baseline_weight_distribution()

    elif args.model == 'altered' :
        cellspec = dict()
        cellspec["nspines_perdend"] = 30
        cellspec["spine_len"] = 1.696
        cellspec["spine_head_diam"] = 0.718
        cellspec["spine_neck_diam"] = 0.301 
        cellspec["model"] = "epilepsy" 

        ampaw, nmdaw = get_altered_weight_distribution()

    #set default value
    ntrial = 3
    ihz_config = {'start': 0.2, 'tick': 0.2, 'limit':5.0}
    nbasal_config = {'start' : 1, 'limit' : 10}

    #overwrite if argument input was exist
    if args.ihz != None : 
        if args.ihz[0] > args.ihz[2] : 
            parser.error(f'@@error, check ihz input')
        ihz_config = {'start': args.ihz[0], 'tick': args.ihz[1], 'limit':args.ihz[2]}

    if args.nbasal != None :
        if args.nbasal[0] > args.nbasal[1] :
            parser.error(f'@@error, check nbasal input')
        nbasal_config = {'start' : args.nbasal[0], 'limit' : args.nbasal[1]}


    #result 
    rsl_nbasal = []
    rsl_ihz = []
    rsl_trial = []
    rsl_firing = []

    nbasal = nbasal_config['start']
    while True :
        ihz = ihz_config['start']
        cellspec["nbasal"] = nbasal

        while True :
            for trial in range(0,ntrial) :
                #make cell 
                cell = morphology.cell(verbose=False)
                cell.create_cell_with_spec(cellspec=cellspec)
                cell.set_cell_properties()
            
                h.celsius = 37
               
                firing = measure_firing_rate(cell, stim_hz=ihz, weight_ampa=ampaw, weight_nmda=nmdaw, vinit=-70, dt=0.1)

                del cell
                print(f'nbasal:{nbasal}_ihz:{ihz}_trial#{trial}_firing:{firing:.2f}')

                rsl_nbasal.append(nbasal)
                rsl_ihz.append(ihz)
                rsl_trial.append(trial)
                rsl_firing.append(firing)

            #save as pkl for every ihz
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_as_pkl(fname=f'pkls/ioratecurve_{args.model}_{now}.pkl', 
                        nbasal=rsl_nbasal, ihz=rsl_ihz, trial=rsl_trial, firingrate=rsl_firing)

            ihz = ihz + ihz_config['tick']
            if round(ihz,1) > round(ihz_config['limit'],1) :
                break

        ihz = ihz_config['start']
        nbasal = nbasal +1
        if nbasal > nbasal_config['limit'] :
            break
    