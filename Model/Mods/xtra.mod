: $Id: xtra.mod,v 1.4 2014/08/18 23:15:25 ted Exp ted $
: 2018/05/20 Modified by Aman Aberra

NEURON {
	SUFFIX xtra
	RANGE es, os : (es = max amplitude of the potential, os = max intensity of the light field)
	RANGE x, y, z, type, order
	GLOBAL estim, ostim : (stim = normalized waveform)
	POINTER ex, ox
}

UNITS {
  (W) = (watt)
}

PARAMETER {
	es = 0 (mV)
	os = 0 (W/m2)
	x = 0 (1) : spatial coords
	y = 0 (1)
	z = 0 (1)
	type = 0 (1) : numbering system for morphological category of section - unassigned is 0
	order = 0 (1) : order of branch/collateral.
}

ASSIGNED {
	v (millivolts)
	ex (millivolts)
	ox (W/m2)
	estim (unitless)
	ostim (unitless)
	area (micron2)
}

INITIAL {
	ex = estim*es
	ox = ostim*os
}


BEFORE BREAKPOINT { : before each cy' = f(y,t) setup
  ex = estim*es
  ox = ostim*os
}

