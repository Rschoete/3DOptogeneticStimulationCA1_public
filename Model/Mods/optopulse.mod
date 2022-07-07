: $Id: optopulse.mod, v1.0 2021/04/12 11:30AM Ruben Schoeters$

COMMENT

pointprocess optic pulse


ENDCOMMENT

NEURON{
    POINT_PROCESS OPTO_pulse
    RANGE osstart, osprp, osdc, osdur, Amp, Io
	
}

UNITS {
  (W) = (watt)
} 

PARAMETER{
    osstart = 0 (ms)
    osprp = 0 (ms)
    osdc = 1 
    osdur = -1 (ms)
    Amp = 0 (W/m2)
}

ASSIGNED{
    Io (W/m2)
    on (1)
	
}

INITIAL{    
    on = 0
    Io=0
	

    if (osdur>0){
        net_send(osstart,1)
    }    
}
BREAKPOINT{

if(on==1){
    Io = Amp
}else{
    Io=0
}
}

NET_RECEIVE(w){
    if (flag == 1){
        if (on==0){
            if (t>(osstart+osdur)){
                on = 0                
            }else{
            on = 1
            if (osdc == 1){
                net_send(osdur, 1)
            }else{
                net_send(osprp*osdc, 1)
            }
            }
            
        } else {
            on = 0            
            if (osdc != 1){               
                net_send(osprp*(1-osdc), 1)
            }
        }
    }
}



