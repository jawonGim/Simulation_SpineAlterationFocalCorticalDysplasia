from neuron import h, gui
import math 
import time
import random
import numpy 

from matplotlib import pyplot as plt

#cell that contains only 1 spine which is randomly located on a randomly selected dendrite
class cell() :
    def __init__(self, name="cell", gid=0):
        #random.seed(1) #use the same seed to get same number in every run
        random.seed(time.time())
        
        cellspec = dict()
        cellspec["soma_diam"] = 17
        cellspec["soma_len"] = 17

        cellspec["ndend"] = 20
        cellspec["nbasal"] = 6        

        cellspec["nspines_perdend"] = 1
        cellspec["spine_len"] = 1.823
        cellspec["spine_head_diam"] = 0.551
        cellspec["spine_neck_diam"] = 0.148

        cellspec["apical_trunk_len"] = 30
        cellspec["basal_prxm_len"] = 30
        cellspec["hypo_dend_len"] = 250
        cellspec["explicit_dend_len"] = 101.5

        cellspec["model"] = "control" #control or epilepsy

        self.cellspec = cellspec
        self.name = name
        self.gid = gid

    def create_cell(self) :
        self.__cellspec_validtest()

        model = self.cellspec["model"]
        print(f"{model} cell {self.name} {self.gid} creating...")

        self.__createSoma() 
        self.__createDendrites()

        h.distance(sec=self.soma)   #set origin as 0 end of soma to compute distance from the soma

        #make spines on randomly selected dendrites
        self.__spine_on_dendrites() 

        #deploy receptprs on spine heads 
        self.__deploy_receptors()


        
    def create_cell_with_spec(self, cellspec) :
        self.cellspec.update(cellspec)
        self.create_cell()


    def __cellspec_validtest(self) :
        assert self.cellspec["soma_diam"] > 0, "soma_diam should be bigger than 0\n"
        assert self.cellspec["soma_len"] > 0, "soma_len should be bigger than 0\n"
        assert self.cellspec["ndend"] > 0, "ndend should be bigger than 0\n"
        assert self.cellspec["ndend"] > self.cellspec["nbasal"], "ndend should be bigger than nbasal\n"
 
        assert self.cellspec["apical_trunk_len"] > 0, "apical_trunk_len should be bigger than 0\n"
        assert self.cellspec["basal_prxm_len"] > 0, "basal_prxm_len should be bigger than 0\n"
        assert self.cellspec["hypo_dend_len"] > 0, "hypo_dend_len should be bigger than 0\n"
        assert self.cellspec["explicit_dend_len"] > 0, "explicit_dend_len should be bigger than 0\n"

        assert self.cellspec["model"] in ["control", "epilepsy"], "model should be control or epilepsy\n"

    #where these values came from?
    def set_cell_properties(self, cm=1.0, ra=203.23, rm=38907, na_bar=2000, k_bar=600, vleak=-70, sf=2.0) :
        try :
            self.soma
        except AttributeError :
            print("create cell first\n")
            return
        
        #add ion channels and passive properties to the cell
        for part in self.soma.wholetree():
            part.cm = cm
            part.Ra = ra
            part.insert("pas")
            part.g_pas = (1/rm)
            part.e_pas = vleak

        #apply spine factor to implicit spiny dendrites 
        for dend in self.sf_dends :
            dend.g_pas = (1/rm) * sf
            dend.cm = cm * sf

        for dend in self.spiny_dends :
            dend.g_pas = (1/rm) * sf
            dend.cm = cm * sf

        #insert soma specific channel (HH like Na+, K+ channel)
        self.soma.insert("na")
        self.soma.insert("kv")
        self.soma.gbar_na = na_bar
        self.soma.gbar_kv = k_bar

    def __spine_on_dendrites(self):
        nbasal = self.cellspec["nbasal"]
        ndend_total = self.cellspec["ndend"]

        dend_type = random.choice(["apical", "basal"])
        if dend_type == "apical" :
            letter = "A"
            ydirection = +1         
            dend_idx = [0, ndend_total-nbasal];   
        else :
            letter = "B"
            ydirection = -1
            dend_idx = [ndend_total-nbasal, ndend_total]

        #select dendrties randomly
        select_dend_idx = random.sample(range(dend_idx[0], dend_idx[1]), 1)
        select_dend_type = random.choice(["hypo", "spiny"])

        if select_dend_type == "hypo" :
            spinylen = self.cellspec["hypo_dend_len"]
            nseg = 250 
            spiny_start = self.sf_start[select_dend_idx[0]]
            dends = self.sf_dends[select_dend_idx[0]]

        else : #distal spiny
            spinylen = self.cellspec["explicit_dend_len"]
            nseg = 135
            spiny_start = self.spiny_start[select_dend_idx[0]]
            dends = self.spiny_dends[select_dend_idx[0]]
                
        print(f"    A spine is on {dends}")
        
        seglen = (1 / nseg) * spinylen
        dend_diam = 1

        headdiam = self.cellspec["spine_head_diam"] 
        necklen = self.cellspec["spine_len"] - headdiam
        neckdiam = self.cellspec["spine_neck_diam"]    

        npoint_spine = 50
        head_half = headdiam / 2
        e_spine = 0.001

        nindd = self.cellspec["nspines_perdend"]

        heads = []
        necks = []

        idend = select_dend_idx[0]

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
        lind = 0
        ispine = 0

        xloc_mu = spiny_start[0] + (dend_diam/2)
        yloc_norm = (segidx[ispine]/nseg) + (1/(nseg*2))
        yloc_mu = seglen * segidx[ispine] + seglen/ 2
        yloc_mu = ydirection * yloc_mu + spiny_start[1]
        zloc_mu = spiny_start[2]

        # to check spine base location
        self.spine_loc_dend = dends
        self.spine_loc_in_dends = yloc_norm

        necks.append(h.Section(name="spine_neck{}_{}_{}".format(letter, idend, ispine)))
        necks[lind].nseg = 3
        necks[lind].connect(dends(yloc_norm), 0) 
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


        self.spine_heads = heads
        self.spine_necks = necks

        self.spines_on = letter

    def __deploy_receptors(self) :
        ampar = []
        nmdar = []

        for head in self.spine_heads :
            ampar.append(h.ampa(head(0.5)))
            nmdar.append(h.nmda(head(0.5)))

        self.ampar = ampar
        self.nmdar = nmdar 

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
        self.nc_spike = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        self.nc_spike.threshold = -10
        self.nc_spike.record(self.spike_times)
        
    
    def __createDendrites(self) :
        #make dendrites and connect it to the soma
        self.__apical_dendrites()
        self.__basal_dendrites() 

    def __apical_dendrites(self) :
        napical = self.cellspec["ndend"] - self.cellspec["nbasal"]
        prxmlen = self.cellspec["apical_trunk_len"]
        hypospinylen = self.cellspec["hypo_dend_len"]
        spinylen = self.cellspec["explicit_dend_len"]

        nseg = 21   
        seglen = (1 / nseg) * spinylen

        angle_intv = 7
        dend_diam = 1
               
        #apical trunk (4 to 1.8 um of diameter)
        trunk = h.Section(name = "apical_trunk_dendrite")
        trunk.nseg = 21
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
        explicit_spiny = []
        coord_spiny = []
        coord_hypo = []
        for idend in range(0,napical) :
            #implicit spiny dendrites (middle of trunk-distal spiny)
            hypo_spiny.append(h.Section(name="apicalSF_dendrite_"+str(idend)))
            hypo_spiny[idend].nseg = 250
            hypo_spiny[idend].connect(self.apical_trunk(1), 0)

            ncoord = self.__get_coord_with_angle_and_length(dend_start, angle, hypospinylen)
            h.pt3dclear(sec = hypo_spiny[idend])
            h.pt3dadd(dend_start[0], dend_start[1], dend_start[2], dend_diam, sec=hypo_spiny[idend])
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=hypo_spiny[idend])

            coord_hypo.append(dend_start)

            #explicit spiny (most distal, inputs will be comming into)
            coord_spiny.append(ncoord)
            spiny_start = ncoord
            explicit_spiny.append(h.Section(name="apical_spiny_dendrite_"+str(idend)))
            explicit_spiny[idend].nseg = nseg
            explicit_spiny[idend].connect(hypo_spiny[idend](1), 0)
            
            h.pt3dclear(sec = explicit_spiny[idend])
            h.pt3dadd(spiny_start[0], spiny_start[1], spiny_start[2], dend_diam, sec=explicit_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen-seglen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=explicit_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 0.00001, sec=explicit_spiny[idend])     #sealed end
        
        self.sf_dends = hypo_spiny
        self.sf_start = coord_hypo
        self.spiny_dends = explicit_spiny
        self.spiny_start = coord_spiny 


    def __basal_dendrites(self) :
        nbasal = self.cellspec["nbasal"]
        prxmlen = self.cellspec["basal_prxm_len"]
        hypo_spinylen = self.cellspec["hypo_dend_len"]
        spinylen = self.cellspec["explicit_dend_len"]

        nseg = 21  
        seglen = (1 / nseg) * spinylen

        angle_intv = 7 
        dend_diam = 1

        dend_start = [-int(self.cellspec["soma_len"]/2), -int(self.cellspec["soma_diam"]), 1]
        angle = 270

        angles = numpy.arange(angle-(nbasal//2)*angle_intv, angle+(nbasal//2)*angle_intv+1, angle_intv) 

        if len(angles) > nbasal : 
            angles = angles[:-1]

        if nbasal/2 == nbasal//2 :    #add an angle when even number of basal dendrites
            angles = angles + angle_intv//2

        proximal = []
        hypo_spiny = []
        explicit_spiny = []
        coord_spiny = []
        coord_hypo = []
        for idend in range(0,nbasal) :
            #proximal dendrites (2 to 1um of diameter) 
            proximal.append(h.Section(name="basal_proximal_dendrite_"+str(idend))) 
            proximal[idend].nseg = 135
            proximal[idend].connect(self.soma(0.7), 0)

            h.pt3dclear(sec = proximal[idend])
            h.pt3dadd(dend_start[0], dend_start[1], dend_start[2], 2, sec=proximal[idend])
            ncoord = self.__get_coord_with_angle_and_length(dend_start, angle, prxmlen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 1, sec=proximal[idend])

            prxm_end = ncoord
            coord_hypo.append(prxm_end)

            #implicit spiny dendrites (middle of proximal-distal spiny)
            hypo_spiny.append(h.Section(name="basalSF_dendrite_"+str(idend)))
            hypo_spiny[idend].nseg = 250
            hypo_spiny[idend].connect(proximal[idend](1), 0)

            ncoord = self.__get_coord_with_angle_and_length(prxm_end, angle, hypo_spinylen)
            h.pt3dclear(sec = hypo_spiny[idend])
            h.pt3dadd(prxm_end[0], prxm_end[1], prxm_end[2], dend_diam, sec=hypo_spiny[idend])
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=hypo_spiny[idend])

            #explicit spiny (most distal, inputs will be comming into)
            coord_spiny.append(ncoord)
            spiny_start = ncoord
            explicit_spiny.append(h.Section(name="basal_spiny_dendrite_"+str(idend)))
            explicit_spiny[idend].nseg = nseg
            explicit_spiny[idend].connect(hypo_spiny[idend](1), 0)
            
            h.pt3dclear(sec = explicit_spiny[idend])
            h.pt3dadd(spiny_start[0], spiny_start[1], spiny_start[2], dend_diam, sec=explicit_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen-seglen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], dend_diam, sec=explicit_spiny[idend])
            ncoord = self.__get_coord_with_angle_and_length(spiny_start, angle, spinylen)
            h.pt3dadd(ncoord[0], ncoord[1], ncoord[2], 0.00001, sec=explicit_spiny[idend])     #sealed end
       
        self.basal_proximal_dends = proximal
        self.sf_dends.extend(hypo_spiny)
        self.sf_start.extend(coord_hypo)
        self.spiny_dends.extend(explicit_spiny)
        self.spiny_start.extend(coord_spiny)

    def draw(self) :
        try :
            self.soma
        except AttributeError :
            print("create cell first\n")
            return

        s = h.Shape()
        s.show(False)
        time.sleep(30)
    
    def draw_3d(self):        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Draw morphology with diameter as line width
        for sec in self.soma.wholetree():
            print(sec)
            n3d = int(h.n3d(sec=sec))
            if n3d < 2:
                continue

            xs = [h.x3d(i, sec=sec) for i in range(n3d)]
            ys = [h.y3d(i, sec=sec) for i in range(n3d)]
            zs = [h.z3d(i, sec=sec) for i in range(n3d)]
            ds = [h.diam3d(i, sec=sec) for i in range(n3d)]

            for i in range(n3d - 1):
                ax.plot(
                    xs[i:i+2],
                    ys[i:i+2],
                    zs[i:i+2],
                    color='black',
                    linewidth=ds[i] * 2
                )


        ax.set_axis_off()  # turn off everything including background panes

        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.view_init(elev=-90, azim=90)

        plt.tight_layout()
        plt.show()