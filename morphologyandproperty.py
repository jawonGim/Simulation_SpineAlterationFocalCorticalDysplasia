#this file contains functions for measuring whole cell level properties when differing morphology

from neuron import h
from neuron.units import ms, mV
import cell_singlespine_randomloc as cellrl


#measure input resistance (at soma) when differing dendrite combination 
#this function uses default weight values of control(baseline) for both model
#surface area difference applied
def measure_inputresistance_with_various_dendrite_combination(model='ctl', min_basal=1, max_basal=10) :
    print(f"measure input resistance with various dendrite combination(basal {min_basal} to {max_basal})")

    from utils import measure_input_resistance

    h.load_file("stdrun.hoc")
    vinit = -70 * mV
    h.celsius = 37
    h.dt = 0.1 *ms
    h.v_init = vinit

    stim = h.NetStim()
    stim.start = 300 *ms
    stim.number = 1

    x_nbasal = []
    y_ri = []
    if model == 'ctl' :
        cellname = 'control'
        sf = 2.0
    else :
        cellname = 'epilepsy'
        sf = 1.2
        

    for nbasal in range(min_basal, max_basal+1) :
        x_nbasal.append(nbasal)

        cell_ = cellrl.cell(name=cellname)
        spec_ = dict()
        spec_["nbasal"] = nbasal

        cell_.create_cell_with_spec(spec_)

        ampa_ctl = h.NetCon(stim, cell_.ampar[0])
        ampa_ctl.weight[0] = 31 #ampa_weight
        ampa_ctl.delay = 0 * ms
        nmda_ctl = h.NetCon(stim, cell_.nmdar[0])
        nmda_ctl.weight[0] = 14 #nmda_weight
        nmda_ctl.delay = 0 * ms

        tstop = 1_000 * ms

        #mesure RI with varius factor 
        cell_.set_cell_properties(sf=sf)
        ri = measure_input_resistance(cell_)
        y_ri.append(ri)


        h.finitialize(vinit)
        h.continuerun(tstop)

        del cell_
    
    return x_nbasal, y_ri

#measure input resistance (at soma) when differing spine density, so differing surface area
#this function uses default weight values of control(baseline)
def measure_inputresistance_with_various_spine_density(sf_min=0.8, sf_max=2.5) :
    print(f"measure input resistance wi sf {sf_min} to {sf_max})")
    from utils import measure_input_resistance

    h.load_file("stdrun.hoc")

    vinit = -70 * mV

    h.celsius = 37
    h.dt = 0.1 *ms
    h.v_init = vinit

    stim = h.NetStim()
    stim.start = 300 *ms
    stim.number = 1

    x_sf = []
    y_ri = []
    sf_iter = 0
    while True:
        sf = round(sf_min + sf_iter*0.05, 2)
        if sf>sf_max :
            break

        x_sf.append(sf)

        cell_ = cellrl.cell(name="control")
        cell_.create_cell()

        ampa_ctl = h.NetCon(stim, cell_.ampar[0])
        ampa_ctl.weight[0] = 31 #ampa_weight
        ampa_ctl.delay = 0 * ms
        nmda_ctl = h.NetCon(stim, cell_.nmdar[0])
        nmda_ctl.weight[0] = 14 #nmda_weight
        nmda_ctl.delay = 0 * ms

        tstop = 1_000 * ms

        #mesure RI with varius factor 
        cell_.set_cell_properties(sf=sf)
        ri = measure_input_resistance(cell_)
        y_ri.append(ri)


        h.finitialize(vinit)
        h.continuerun(tstop)

        del cell_
        sf_iter = sf_iter +1
    
    return x_sf, y_ri
