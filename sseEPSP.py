from neuron import h
from neuron.units import mV, ms
import cell_singlespine_randomloc as cellrl
# import numpy as np
# from scipy.optimize import curve_fit


# distance from soma vs somatic amplitude 
# does not include dendrite information
# higher amplitude line from basal dendrite, lower amplitude line from apical dendrite 
def distance_from_soma_vs_somatic_amplitude(N=100, model='ctl') :
    h.load_file("stdrun.hoc")

    vinit = -70 * mV

    h.celsius = 37
    h.dt = 0.1 *ms
    h.v_init = vinit

    stim = h.NetStim()
    stim.start = 300 *ms
    stim.number = 1

    compute_start = (stim.start - 10)/h.dt
    compute_end = (stim.start + 200)/h.dt

    ampa_weight = 31
    nmda_weight = 14

    if model == "ctl" :
        sf = 2.0
        cell_name = "control"
    else :
        sf = 1.2
        cell_name = "epileptogenic"


    amplitude = []
    dist_spine = []

    for i in range(1, N+1) :
        cell = cellrl.cell(name=cell_name, gid=i, dend_nseglevel=3)
        cell.create_cell()

        ampa_ctl = h.NetCon(stim, cell.ampar[0])
        ampa_ctl.weight[0] = ampa_weight     #ampa_weight
        ampa_ctl.delay = 0 * ms
        nmda_ctl = h.NetCon(stim, cell.nmdar[0])
        nmda_ctl.weight[0] = nmda_weight     #nmda_weight
        nmda_ctl.delay = 0 * ms

        tstop = 1_000 * ms

        cell.set_cell_properties(sf=sf)

        h.finitialize(vinit)
        h.continuerun(tstop)      

        amplitude.append(cell.v_soma.max(compute_start, compute_end) - cell.v_soma.min(compute_start, compute_end))
        dist_spine.append(h.distance(cell.soma(1), cell.spine_necks[0](0)))

        del cell

    #sort
    dist_spine, amplitude = zip(*sorted(zip(dist_spine, amplitude)))

    return dist_spine, amplitude

# measure EPSP at multiple points (head, base, soma)
#this function uses default weight values of control(baseline) for both models
#this function considers only between groups, not vast range of values
def multipoints_EPSP_with_alteration_between_groups(model='ctl', alter_component=None, N=100) :
    h.load_file("stdrun.hoc")

    vinit = -70 * mV

    h.celsius = 37
    h.dt = 0.1 *ms
    h.v_init = vinit

    stim = h.NetStim()
    stim.start = 300 *ms
    stim.number = 1

    compute_start = (stim.start - 10)/h.dt
    compute_end = (stim.start + 300)/h.dt

    amp_head = []
    amp_base = []
    amp_soma = []

    cellspec = dict()

    if model == 'ctl' :
        cellname = 'control'
        sf = 2.0
    else :
        cellname = 'epileptogenic' 
        cellspec["model"] = "epilepsy"
        sf = 2.0

    if alter_component == 'neck' :
        cellspec["spine_neck_diam"] = 0.301
    elif alter_component == 'head' :
        cellspec["spine_head_diam"] = 0.718
        cellspec["spine_len"] = 1.823-0.551+0.718
    elif alter_component == 'density' :
        sf = 1.2
    elif alter_component == None :
        pass
    else :
        print(f"@@@ unknown alter_component: {alter_component}, will run with default values")

    tstop = 1_000 * ms

    for i in range(0, N) :
        cell = cellrl.cell(name=cellname, gid=i, dend_nseglevel=3)    
        cell.create_cell_with_spec(cellspec)

        ampa_fcd = h.NetCon(stim, cell.ampar[0])
        ampa_fcd.weight[0] = 31     #ampa_weight
        ampa_fcd.delay = 0 * ms
        nmda_fcd = h.NetCon(stim, cell.nmdar[0])
        nmda_fcd.weight[0] = 14     #nmda_weight
        nmda_fcd.delay = 0 * ms


        vhead_ = h.Vector().record(cell.spine_heads[0](0.5)._ref_v)
        vbase_ = h.Vector().record(cell.spine_loc_dend(cell.spine_loc_in_dends)._ref_v)

        cell.set_cell_properties(sf=sf)

        h.finitialize(vinit)
        h.continuerun(tstop)      

        amp_head.append(vhead_.max(compute_start, compute_end) - vhead_.min(compute_start, compute_end))
        amp_base.append(vbase_.max(compute_start, compute_end) - vbase_.min(compute_start, compute_end))
        amp_soma.append(cell.v_soma.max(compute_start, compute_end) - cell.v_soma.min(compute_start, compute_end))

        del cell
    
    return amp_head, amp_base, amp_soma

# measure EPSP at multiple points (head, base, soma)
#this function uses default weight values of control(baseline) for both models
#this function only considers vast range of head diameter 
def multipoints_EPSP_with_alter_head(N=20, head_diam_start=0.2, head_diam_limit=1.5) :
    h.load_file("stdrun.hoc")

    vinit = -70 * mV

    h.celsius = 37
    h.dt = 0.1 *ms
    h.v_init = vinit

    stim = h.NetStim()
    stim.start = 300 *ms
    stim.number = 1

    compute_start = (stim.start - 10)/h.dt
    compute_end = (stim.start + 300)/h.dt

    
    head_ = []
    base_ = []
    soma_ = []
    head_max = []
    base_max = []
    soma_max = []
    head_min = []
    base_min = []
    soma_min = []

    cellspec = dict()

    x_headdiam = []
    tick = 0.01
    head_diam = head_diam_start     #start

    #for head_d in range(0, diam_ticks) :
    while True :
        amp_head = []
        amp_base = []
        amp_soma = []

        x_headdiam.append(head_diam)
        for i in range(0, N) :
            cell = cellrl.cell(name="control")
            cellspec["spine_head_diam"] = head_diam
            cellspec["spine_len"] = 1.3 + head_diam #1.3 is for neck length
            cell.create_cell_with_spec(cellspec)
            
            ampa_receptor = h.NetCon(stim, cell.ampar[0])
            ampa_receptor.weight[0] = 31     #ampa_weight
            ampa_receptor.delay = 0 * ms
            nmda_receptor = h.NetCon(stim, cell.nmdar[0])
            nmda_receptor.weight[0] = 14     #nmda_weight
            nmda_receptor.delay = 0 * ms

            tstop = 1_000 * ms

            vhead = h.Vector().record(cell.spine_heads[0](0.5)._ref_v)
            vbase = h.Vector().record(cell.spine_loc_dend(cell.spine_loc_in_dends)._ref_v)

            cell.set_cell_properties(sf=2.0)  

            h.finitialize(vinit)
            h.continuerun(tstop)      

            amp_head.append(vhead.max(compute_start, compute_end) - vhead.min(compute_start, compute_end))
            amp_base.append(vbase.max(compute_start, compute_end) - vbase.min(compute_start, compute_end))
            amp_soma.append(cell.v_soma.max(compute_start, compute_end) - cell.v_soma.min(compute_start, compute_end))

            del cell

        head_.append(sum(amp_head)/N)
        base_.append(sum(amp_base)/N)
        soma_.append(sum(amp_soma)/N)
        head_max.append(max(amp_head))
        base_max.append(max(amp_base))
        soma_max.append(max(amp_soma))
        head_min.append(min(amp_head))
        base_min.append(min(amp_base))
        soma_min.append(min(amp_soma))

        head_diam = head_diam + tick
        head_diam = round(head_diam, 2)
        
        if head_diam > head_diam_limit :
            break
    
    return {'range' : [head_diam_start, head_diam_limit], 
            'head_diameter' : x_headdiam,
            'head': head_, 'base' : base_, 'soma' : soma_,
            'head_max' : head_max, 'base_max' : base_max, 'soma_max' : soma_max,
            'head_min' : head_min, 'base_min' : base_min, 'soma_min' : soma_min} 

# measure EPSP at multiple points (head, base, soma)
#this function uses default weight values of control(baseline) for both models
#this function only considers vast range of neck diameter 
#this function returns deltaV also
def multipoints_EPSP_with_alter_neck(N=20, neck_diam_start=0.05, neck_diam_limit=0.52) :
    h.load_file("stdrun.hoc")

    vinit = -70 * mV

    h.celsius = 37
    h.dt = 0.1 *ms
    h.v_init = vinit

    stim = h.NetStim()
    stim.start = 300 *ms
    stim.number = 1

    compute_start = (stim.start - 10)/h.dt
    compute_end = (stim.start + 50)/h.dt

    x_neckdiam = []
    amp_head = []
    amp_base = []
    amp_soma = []
    
    deltaV_max = []
    deltaV_min = []
    
    neck_factor = 0
    neck_diam_tick = 0.01
    while True :
        neck_diam = round(neck_diam_start + neck_factor*neck_diam_tick, 2)
        if neck_diam > neck_diam_limit: 
            break
        x_neckdiam.append(neck_diam)
        print(f"neck_diam {neck_diam}")
        diff_ = []
        for i in range(0, N) :
            cell = cellrl.cell(name="control")
            cellspec = dict()
            cellspec["spine_neck_diam"] = neck_diam
            cell.create_cell_with_spec(cellspec)

            ampa_ctl = h.NetCon(stim, cell.ampar[0])
            ampa_ctl.weight[0] = 31     #ampa_weight
            ampa_ctl.delay = 0 * ms
            nmda_ctl = h.NetCon(stim, cell.nmdar[0])
            nmda_ctl.weight[0] = 14     #nmda_weight
            nmda_ctl.delay = 0 * ms

            tstop = 1_000 * ms

            vhead = h.Vector().record(cell.spine_heads[0](0.5)._ref_v)
            vbase = h.Vector().record(cell.spine_loc_dend(cell.spine_loc_in_dends)._ref_v)

            cell.set_cell_properties(sf=2.0)

            h.finitialize(vinit)
            h.continuerun(tstop)      

            amp_head.append(vhead.max(compute_start, compute_end) - vhead.min(compute_start, compute_end))
            amp_base.append(vbase.max(compute_start, compute_end) - vbase.min(compute_start, compute_end))
            amp_soma.append(cell.v_soma.max(compute_start, compute_end) - cell.v_soma.min(compute_start, compute_end))

            diff_.append(vhead.max(compute_start, compute_end) - vbase.max(compute_start, compute_end))

            del cell

        deltaV_max.append(max(diff_))
        deltaV_min.append(min(diff_))

        neck_factor = neck_factor+1
    
    return {'range' : [neck_diam_start, neck_diam_limit], 
            'neck_diameter' : x_neckdiam,
            'amp_head': amp_head, 'amp_base' : amp_base, 'amp_soma' : amp_soma,
            'deltaV_max' : deltaV_max, 'deltaV_min' : deltaV_min} 



if __name__ == "__main__" :
    pass