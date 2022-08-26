import numpy as np
def convert_strtoseclist(cell,location):
    if location.lower()=='soma':
        location = cell.soma
    elif location.lower()=='axon':
        location = cell.axon
    elif location.lower()=='all':
        location = cell.allsec
    elif location.lower()=='alldend':
        location = cell.alldend
    elif location.lower() in ['apic','apical']:
        location = cell.apical
    elif location.lower() in ['basal','dend']:
        location = cell.dend
    elif location.lower() == 'apicaltrunk':
        location = cell.apicalTrunk
    elif location.lower() == 'apicaltrunk_ext':
        location = cell.apicalTrunk_ext
    elif location.lower() == 'apicaltuft':
        location = cell.apicalTuft
    elif location.lower() == 'obliques':
        location = cell.apical_obliques
    elif location.lower() == 'apicalnotuft':
        location = cell.apicalTrunk_ext+cell.apical_obliques
    else:
        try:
            location = getattr(cell,location)
        except:
            raise ValueError(location)
    return location

def convert_strtoseg(cell,mystr):
    str_split = mystr.split('(',1)
    sec = str_split[0]

    secnr = int(sec.split('[',1)[-1][:-1])
    sec = sec.split('[',1)[0]
    loc = float(str_split[-1][:-1]) # :-1 to exclude closing bracket
    return getattr(cell,sec)[secnr](loc)

def convert_strtosec(cell,mystr):
    str_split = mystr.split('[',1)
    sec = str_split[0]
    if len(str_split)>1:
        secnr = float(str_split[-1][:-1])
        return getattr(cell,sec)[secnr]
    else:
        return getattr(cell,sec)

def get_dts_simdurs(simdur,dt_init,edts,esimtimes,odts,osimtimes):
    '''
    #test case succesful
    odts = [25,5,25,5,25,5,25]
    osimtimes = [100,101,110,111,120,121,130]
    edts = [25,4,20,4,20,4,25]
    esimtimes = [103,104,111,112,120.5,121.5,130]
    simdur = 135
    dt_init = 19
    '''
    if len(osimtimes)==0 and len(esimtimes)==0:
        dts = [dt_init]; simdurs = [simdur]
        return dts, simdurs
    elif len(osimtimes)==0:
        simtimes = esimtimes
        xdts = edts
    elif len(esimtimes)==0:
        simtimes = osimtimes
        xdts = odts

    else:
        xdts = np.array(odts)
        simtimes = np.array(osimtimes)

        for edt,est0,est1 in zip(edts,[0]+esimtimes[:-1],esimtimes):
            idx_l = simtimes<=est0
            idx = (simtimes>est0) & (simtimes<est1)
            idx_h = simtimes>=est1
            idx_sh = simtimes>est1
            dts_intm = np.concatenate((xdts[idx_l],np.minimum(np.array(xdts)[idx],edt)))
            dts_intm = np.append(dts_intm,min(xdts[idx_h][0],edt))
            xdts = np.concatenate((dts_intm,xdts[idx_sh]))
            simtimes = np.concatenate((simtimes[idx_l],simtimes[idx],np.array([est1]),simtimes[idx_sh]))
        xdts = list(xdts)
        simtimes = list(simtimes)

    # append simdur
    if simtimes[-1]<simdur:
        dts = list(np.minimum(xdts,dt_init))+[dt_init]
        simdurs = simtimes+[simdur]
    else:
        simtimes = np.array(simtimes)
        xdts = np.array(xdts)
        idx_l = simtimes<simdur
        idx_h= simtimes>=simdur
        dts=np.minimum(np.array(xdts)[idx_l],dt_init)
        simdurs = simtimes[idx_l]
        dts = np.append(dts,min(xdts[idx_h][0],dt_init))
        simdurs = np.append(simdurs,simdur)

    return dts,simdurs

