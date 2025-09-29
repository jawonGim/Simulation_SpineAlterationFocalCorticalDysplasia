from neuron import h, gui
import math 
import time
import random
import numpy 

class cell() :
    def __init__(self, verbose=True):
        #random.seed(1) #use the same seed to get same number in every run
        random.seed(time.time())
        
        cellspec = dict()
        cellspec["soma_diam"] = 17
        cellspec["soma_len"] = 17

        cellspec["ndend"] = 20
        cellspec["nbasal"] = 6        

        cellspec["nspines_perdend"] = 80
        cellspec["spine_len"] = 1.823
        cellspec["spine_head_diam"] = 0.551
        cellspec["spine_neck_diam"] = 0.148

        cellspec["apical_trunk_len"] = 30
        cellspec["basal_prxm_len"] = 30
        cellspec["hypo_dend_len"] = 250
        cellspec["active_dend_len"] = 101.5

        cellspec["model"] = "control" #control or epilepsy

        self.cellspec = cellspec
        self.verbose = verbose

    def create_cell(self) :
        self.__cellspec_validtest()

        model = self.cellspec["model"]
        if self.verbose :
            print(f"{model} model creating...")

        self.__createSoma() 
        self.__createDendrites()

        #make spines on most distal dendrites
        self.__spine_on_dendrites("apical") 
        self.__spine_on_dendrites("basal") 

        #deploy receptprs on spine heads 
        self.__deploy_receptors()

        
    def create_cell_with_spec(self, cellspec) :
        self.cellspec.update(cellspec)
        self.create_cell()

    def set_cell_properties(self, cm=1.0, ra=203.23, rm=38907, na_bar=2000, k_bar=600, vleak=-70) :
        try :
            self.soma
        except AttributeError :
            print("create cell first\n")
            return
        
        
        if self.cellspec["model"] == "control" :
            sf = 2.0
        else:
            sf = 1.2

        #add ion channels and passive properties to the cell
        for part in self.soma.wholetree():
            part.cm = cm
            part.Ra = ra
            part.insert("pas")
            part.g_pas = (1/rm)
            part.e_pas = vleak

        #apply spine factor to hypothetic spiny dendrites 
        for dend in self.sf_dends :
            dend.g_pas = (1/rm) * sf
            dend.cm = cm * sf

        #insert soma specific channel (HH like Na+, K+ channel)
        self.soma.insert("na")
        self.soma.insert("kv")
        self.soma.gbar_na = na_bar
        self.soma.gbar_kv = k_bar


    def __cellspec_validtest(self) :
        assert self.cellspec["soma_diam"] > 0, "soma_diam should be bigger than 0\n"
        assert self.cellspec["soma_len"] > 0, "soma_len should be bigger than 0\n"
        assert self.cellspec["ndend"] > 0, "ndend should be bigger than 0\n"
        assert self.cellspec["ndend"] > self.cellspec["nbasal"], "ndend should be bigger than nbasal\n"

        assert self.cellspec["nspines_perdend"] > 0, "nspines_perdend should be bigger than 0\n"
        assert self.cellspec["spine_len"] > 0, "spine_len should be bigger than 0\n"
        assert self.cellspec["spine_head_diam"] > 0, "spine_head_diam should be bigger than 0\n"
        assert self.cellspec["spine_neck_diam"] > 0, "spine_neck_diam should be bigger than 0\n"
        assert self.cellspec["spine_len"] > self.cellspec["spine_head_diam"], "spine_len should be bigger than head diameter\n"
 
        assert self.cellspec["apical_trunk_len"] > 0, "apical_trunk_len should be bigger than 0\n"
        assert self.cellspec["basal_prxm_len"] > 0, "basal_prxm_len should be bigger than 0\n"
        assert self.cellspec["hypo_dend_len"] > 0, "hypo_dend_len should be bigger than 0\n"
        assert self.cellspec["active_dend_len"] > 0, "active_dend_len should be bigger than 0\n"

        assert self.cellspec["model"] in ["control", "epilepsy"], "model should be control or epilepsy\n"


    #compute coordinates with start point, angle and length
    #WARNING: z point will be bypassed
    def __get_coord_with_angle_and_length(self, start_point, angle, length) :
        assert length > 0, "length should be bigger than 0\n"
        assert angle >= 0, "angle should be in range 0-360\n"
        assert angle <=360, "angle should be in range 0-360\n"
        assert len(start_point) == 3, "start_point dimension error\n"
    
        x = start_point[0] + length * math.cos(math.radians(angle))
        y = start_point[1] + length * math.sin(math.radians(angle))
        z = start_point[2]
    
        return x, y, z


    def __createSoma(self) :         
        leng = self.cellspec["soma_len"]
        diam = self.cellspec["soma_diam"]

        if self.verbose :
            print(f"\tsoma {leng}um x {diam}um")

        #number of point to draw soma
        npoint = 100
        hlen_soma = leng / 2.0
        e = 0.1
        
        soma = h.Section(name="soma")
        soma.nseg = 3
        soma.L = leng
        h.pt3dclear(sec=soma)
        #sphere-like shaped
        for idx in range(int(npoint / 2)- 1, 0, -1) :
            x = hlen_soma * math.cos(2 * math.pi * idx / npoint)
            y = diam * math.sin(2 * math.pi * idx / npoint)
            h.pt3dadd(x-hlen_soma+1,1,1,y+e, sec=soma)
        
        h.pt3dadd(1,1,1,e, sec=soma)
        self.soma = soma

        # setup recording 
        self.v_soma = h.Vector().record(soma(0.5)._ref_v)
        self.spike_times = h.Vector()
        self.nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        self.nc.threshold = -10
        self.nc.record(self.spike_times)
        
    
    def __createDendrites(self) :
        #make dendrites and connect it to the soma
        self.__apical_dendrites()
        self.__basal_dendrites() 


    def __deploy_receptors(self) :
        ampar = []
        nmdar = []

        for head in self.spine_heads :
            ampar.append(h.ampa(head(0.5)))
            nmdar.append(h.nmda(head(0.5)))

        self.ampar = ampar
        self.nmdar = nmdar 


    def __spine_on_dendrites(self, dend_type):
        nbasal = self.cellspec["nbasal"]
        ndend_total = self.cellspec["ndend"]


        if dend_type == "apical" :
            letter = "A"
            ydirection = +1         
            dend_idx = [0, ndend_total-nbasal];   
        else :
            letter = "B"
            ydirection = -1
            dend_idx = [ndend_total-nbasal, ndend_total]

        spinylen = self.cellspec["active_dend_len"]
        nseg = 135
        seglen = (1 / nseg) * spinylen
        dend_diam = 1

        headdiam = self.cellspec["spine_head_diam"] 
        necklen = self.cellspec["spine_len"] - headdiam
        neckdiam = self.cellspec["spine_neck_diam"]    

        npoint_spine = 50
        head_half = headdiam / 2
        e_spine = 0.001

        spiny_start = self.spiny_start[dend_idx[0]:dend_idx[1]]
        dends = self.spiny_dends[dend_idx[0]:dend_idx[1]]
        ndend = dend_idx[1]-dend_idx[0]

        nindd = self.cellspec["nspines_perdend"]
        if self.verbose :
            print(f"\t\t{nindd} spines for each {dend_type} dendrite, head diam: {headdiam}, neck diam:{neckdiam}")

        heads = []
        necks = []
        for idend in range(0, ndend) :
            #determine spine location
            segidx = []
            while 1:
                idx = random.randrange(1, nseg-2, 1)
                segidx.append(idx)
                segidx = list(set(segidx))
                if len(segidx) >= nindd :
                    break

            assert len(segidx) == nindd, "check spine number again"

            #attatch each spine to spiny dendrites
            for ispine in range(0, nindd) :
                lind = idend * nindd + ispine

                xloc_mu = spiny_start[idend][0] + (dend_diam/2)
                yloc_norm = (segidx[ispine]/nseg) + (1/(nseg*2))
                yloc_mu = seglen * segidx[ispine] + seglen/ 2
                yloc_mu = ydirection * yloc_mu + spiny_start[idend][1]
                zloc_mu = spiny_start[idend][2]

                necks.append(h.Section(name="spine_neck{}_{}_{}".format(letter, idend, ispine)))
                necks[lind].nseg = 3
                necks[lind].connect(dends[idend](yloc_norm), 0) 
                h.pt3dclear(sec=necks[lind])
                h.pt3dadd(xloc_mu, yloc_mu, zloc_mu, neckdiam, sec=necks[lind])
                h.pt3dadd(xloc_mu + necklen, yloc_mu, zloc_mu, neckdiam, sec=necks[lind])

                heads.append(h.Section(name="spine_head{}_{}_{}".format(letter, idend, ispine)))
                heads[lind].nseg = 3
                heads[lind].connect(necks[lind](1), 0)
                h.pt3dclear(sec=heads[lind])

                pos =[xloc_mu+necklen, yloc_mu, zloc_mu]
                #sphere-like shaped
                for idx in range(int(npoint_spine / 2)- 1, 0, -1) :
                    x = head_half * math.cos(2 * math.pi * idx / npoint_spine)
                    y = headdiam * math.sin(2 * math.pi * idx / npoint_spine)
                    h.pt3dadd(x-head_half+pos[0],pos[1],pos[2],y+e_spine, sec=heads[lind])

                h.pt3dadd(pos[0],pos[1], pos[2],e_spine, sec=heads[lind])


        if dend_type == "apical" :
            self.spine_heads = heads
            self.spine_necks = necks
        else :
            self.spine_heads.extend(heads)
            self.spine_necks.extend(necks)


    def __apical_dendrites(self) :
        napical = self.cellspec["ndend"] - self.cellspec["nbasal"]
        prxmlen = self.cellspec["apical_trunk_len"]
        hypospinylen = self.cellspec["hypo_dend_len"]
        spinylen = self.cellspec["active_dend_len"]

        nseg = 135
        seglen = (1 / nseg) * spinylen

        angle_intv = 7
        dend_diam = 1
        

        #make apical dendrites
        if self.verbose :
            print(f"\t{napical} apical dendrites, {prxmlen}um of trunk, {hypospinylen}um of SF dendrites, {spinylen}um of spiny dendrites")
        
        #apical trunk (4 to 1.8 um of diameter)
        trunk = h.Section(name = "apical_trunk_dendrite")
        trunk.nseg = 25
        trunk.connect(self.soma(0.8), 0)
        
        dend_start = [-int(self.cellspec["soma_len"]/2), int(self.cellspec["soma_diam"]), 1]
        angle = 90

        h.pt3dclear(sec = trunk)
        h.pt3dadd(dend_start[0], dend_start[1], dend_start[2], 4, sec=trunk)

        ncoord = self.__get_coord_with_angle_and_length(dend_start, angle, prxmlen/2) 
        h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 2, sec=trunk)

        ncoord = self.__get_coord_with_angle_and_length(dend_start, angle, prxmlen) 
        h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 1.8, sec=trunk)

        self.apical_trunk = trunk


        dend_start = ncoord 
        angles = numpy.arange(angle-(napical//2)*angle_intv, angle+(napical//2)*angle_intv+1, angle_intv) 

        if len(angles) > napical : 
            angles = angles[:-1]

        if napical/2 == napical//2 :    #add an angle when even number of apical dendrites
            angles = angles + angle_intv//2

        hypo_spiny = []
        active_spiny = []
        coord_spiny = []
        for idend in range(0,napical) :
            #hypothetic spiny dendrites (middle of trunk-distal spiny)
            hypo_spiny.append(h.Section(name="apicalSF_dendrite_"+str(idend)))
            hypo_spiny[idend].nseg = 25
            hypo_spiny[idend].connect(self.apical_trunk(1), 0)

            ncoord = self.__get_coord_with_angle_and_length(dend_start, angles[idend], hypospinylen)
            h.pt3dclear(sec = hypo_spiny[idend])
            h.pt3dadd(dend_start[0], dend_start[1], dend_start[2], dend_diam, sec=hypo_spiny[idend])
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=hypo_spiny[idend])

            #active spiny (most distal, inputs will be comming into)
            coord_spiny.append(ncoord)
            spiny_start = ncoord
            active_spiny.append(h.Section(name="apical_spiny_dendrite_"+str(idend)))
            active_spiny[idend].nseg = nseg
            active_spiny[idend].connect(hypo_spiny[idend](1), 0)
            
            h.pt3dclear(sec = active_spiny[idend])
            h.pt3dadd(spiny_start[0], spiny_start[1], spiny_start[2], dend_diam, sec=active_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen-seglen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=active_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 0.00001, sec=active_spiny[idend])     #sealed end
        
        self.sf_dends = hypo_spiny
        self.spiny_dends = active_spiny
        self.spiny_start = coord_spiny


    def __basal_dendrites(self) :
        nbasal = self.cellspec["nbasal"]
        prxmlen = self.cellspec["basal_prxm_len"]
        hypo_spinylen = self.cellspec["hypo_dend_len"]
        spinylen = self.cellspec["active_dend_len"]

        nseg = 135
        seglen = (1 / nseg) * spinylen

        angle_intv = 7 #15
        dend_diam = 1

        if self.verbose :
            print(f"\t{nbasal} basal dendrites, {prxmlen}um of proximal, {hypo_spinylen}um of SF dendrites, {spinylen}um of spiny dendrites")
        
        dend_start = [-int(self.cellspec["soma_len"]/2), -int(self.cellspec["soma_diam"]), 1]
        angle = 270

        angles = numpy.arange(angle-(nbasal//2)*angle_intv, angle+(nbasal//2)*angle_intv+1, angle_intv) 

        if len(angles) > nbasal : 
            angles = angles[:-1]

        if nbasal/2 == nbasal//2 :    #add an angle when even number of basal dendrites
            angles = angles + angle_intv//2

        proximal = []
        hypo_spiny = []
        active_spiny = []
        coord_spiny = []
        for idend in range(0,nbasal) :
            #proximal dendrites (2 to 1um of diameter) 
            proximal.append(h.Section(name="basal_proximal_dendrite_"+str(idend))) 
            proximal[idend].nseg = 25
            proximal[idend].connect(self.soma(0.7), 0)

            h.pt3dclear(sec = proximal[idend])
            h.pt3dadd(dend_start[0], dend_start[1], dend_start[2], 2, sec=proximal[idend])
            ncoord = self.__get_coord_with_angle_and_length(dend_start, angles[idend], prxmlen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 1, sec=proximal[idend])

            prxm_end = ncoord

            #hypothetic spiny dendrites (middle of proximal-distal spiny)
            hypo_spiny.append(h.Section(name="basalSF_dendrite_"+str(idend)))
            hypo_spiny[idend].nseg = 25
            hypo_spiny[idend].connect(proximal[idend](1), 0)

            ncoord = self.__get_coord_with_angle_and_length(prxm_end, angles[idend], hypo_spinylen)
            h.pt3dclear(sec = hypo_spiny[idend])
            h.pt3dadd(prxm_end[0], prxm_end[1], prxm_end[2], dend_diam, sec=hypo_spiny[idend])
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=hypo_spiny[idend])

            #active spiny (most distal, inputs will be comming into)
            coord_spiny.append(ncoord)
            spiny_start = ncoord
            active_spiny.append(h.Section(name="basal_spiny_dendrite_"+str(idend)))
            active_spiny[idend].nseg = nseg
            active_spiny[idend].connect(hypo_spiny[idend](1), 0)
            
            h.pt3dclear(sec = active_spiny[idend])
            h.pt3dadd(spiny_start[0], spiny_start[1], spiny_start[2], dend_diam, sec=active_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen-seglen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=active_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 0.00001, sec=active_spiny[idend])     #sealed end
       
        self.basal_proximal_dends = proximal
        self.sf_dends.extend(hypo_spiny)
        self.spiny_dends.extend(active_spiny)
        self.spiny_start.extend(coord_spiny)

    def draw(self) :
        try :
            self.soma
        except AttributeError :
            print("create cell first\n")
            return

        s = h.Shape()
        s.show(False)
        #time.sleep(30)
