: $Id: ChR2H134R.mod, v1.0 2021/04/12 11:30AM Ruben Schoeters$

COMMENT

ChR2H134R 


ENDCOMMENT

NEURON{
    SUFFIX chr2h134r
    NONSPECIFIC_CURRENT i
    RANGE gchr2bar, gchr2, o, r, echr2, o0, r0, i
    RANGE tauoIV, taurIV, oinf, rinf, Iopto

}

UNITS {
  (mA) = (milliamp)
  (mV) = (millivolt)
  (mS) = (millisiemens)
  (um) = (micron)
  (S)  = (siemens)
  (W) = (watt)
}

PARAMETER{

    gchr2bar = 10.77 (mS/cm2)
    echr2 = 0 (mV)
    r0 = 1
    o0 = 0

    p1oI = 1.81
    p2oI = 1.17
    p3oI = 0.021

    p1oV = 23.14
    p2oV = -0.39
    p3oV = 13.19

    p1rI = 10
    p2rI = 0.56
    p3rI = -1.58
    p4rI = 0.87
    p5rI = 1.96
    p6rI = 0.11

    p1rV = 99.74
    p2rV = -38.69
    p3rV = 12.02

    p1oinf = 3.38
    p2oinf = 0.62

    p1rinf = 1.96
    p2rinf = 0.12
    p3rinf = 0.77

    p1GV = 1
    p2GV = 1.25
    p3GV = 44.52
}

ASSIGNED{
    v (mV)
    i (mA/cm2)
    gchr2 (S/cm2)
    tauoIV (ms)
    taurIV (ms)
    oinf
    rinf

    Iopto (W/m2)

}

STATE{
    o r
}

INITIAL{
    o = o0
    r = r0

    Iopto=0
}
BREAKPOINT{

    SOLVE states METHOD cnexp :derivimplicit is not necessary for this model structure
    :SOLVE kin METHOD sparse
    gchr2 = (1e-3)*gchr2bar*o*r
    i = gchr2*driv(v)
    
}


DERIVATIVE states{
    rates(v,Iopto)

    o' = (oinf - o)/tauoIV
    r' = (rinf - r)/taurIV
}


PROCEDURE rates(v(mV),Iopto(W/m2)){

    LOCAL tauoI, tauoV, taurI, taurV
    tauoI = p3oI/(1+exp(p1oI/p2oI)*Iopto^(1/(p2oI*log(10))))
    tauoV = p1oV/(1+exp(-(v-p2oV)/p3oV))
    tauoIV = (1000)*1/(1/tauoI + 1/tauoV)
    taurI = p1rI*(1-p2rI/(1+exp(p3rI/p4rI)*Iopto^(-1/(p4rI*log(10))))-(1-p2rI)/(1+exp(p5rI/p6rI)*Iopto^(-1/(p6rI*log(10)))))
    taurV = p1rV/(1+exp(-(v-p2rV)/p3rV))
    taurIV = (1000)*1/(1/taurI + 1/taurV)

    oinf = 1/(1+exp(p1oinf/p2oinf)*Iopto^(-1/(p2oinf*log(10))))
    rinf = 1 - p3rinf/(1+exp(p1rinf/p2rinf)*Iopto^(-1/(p2rinf*log(10))))

}
FUNCTION driv(v (mV)) (mV){
    driv = p1GV*(1-p2GV*exp(-(v-echr2)/p3GV))
}

