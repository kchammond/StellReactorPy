#!/usr/bin/env python

import numpy as np
import scipy.special as sp

#----- Physical constants ------------------------------------------------------

qe  = 1.602176634e-19
mu0 = 4. * np.pi * 1.e-7

#----- Geometric quantities ----------------------------------------------------

def a_avg(R_maj, aspect):
    '''
    Averaged minor radius (m) for the plasma LCFS, such that the exact volume
    equals (2 * pi * R_maj) * (pi * a_avg**2)
        R_maj  = major radius (m)
        aspect = aspect ratio
    '''

    return R_maj / aspect

def r_min(R_maj, aspect, elong):
    '''
    Minor radius (m) for the last closed flux surface according to the following
    definition: if the plasma boundary were an axisymmetric torus with an
    elliptical cross-section, r_min would be the semi-major axis of the 
    ellipse for elong < 1 or the semi-minor axis for elong < 1. Elongation
    in this context would be the ratio between the semi-major and semi-minor
    axes.
        R_maj  = major radius (m)
        aspect = aspect ratio
        elong  = elongation
    '''

    return R_maj / aspect / np.sqrt(elong)

def plasma_surface_area(R_maj, aspect, elong):
    '''
    Plasma surface area (m^2)
        R_maj  = major radius (m)
        aspect = aspect ratio
        elong  = elongation
    '''

    # Spreadsheet definitions: not very accurate
    #r = r_min(R_maj, aspect, elong)
    #return (2.*np.pi)**2 * R_maj * r * np.sqrt(0.5 * (1. + elong**2))

    return 8. * np.pi * R_maj**2 / aspect / np.sqrt(elong) \
              * sp.ellipe(1. - elong**2) 

def plasma_volume(R_maj, aspect):
    '''
    Plasma volume (m^3)
        R_maj  = major radius (m)
        aspect = aspect ratio
    '''

    return 2. * np.pi**2 * R_maj * a_avg(R_maj, aspect)**2

#----- Plasma physics quantities -----------------------------------------------

def ne_0(alpha_n, ne_vol_avg):
    '''
    Central electron density (m^-3)
        alpha_n    = density profile coefficient: 
                     ne(rho) = ne_0*(1 - rho^2)^alpha_n
        ne_vol_avg = volume-averaged electron density (m^-3)

    Source: Kovari et al., Fusion Engineering and Design 89, 3054 (2014)
    '''

    return ne_vol_avg * (1 + alpha_n)

def ne_la(alpha_n, ne_vol_avg):
    '''
    Line-averaged electron density (m^-3), in terms of the volume-averaged
    electron density
        alpha_n    = density profile coefficient: 
                     ne(rho) = ne_0*(1 - rho^2)^alpha_n
        ne_vol_avg = volume-averaged electron density (m^-3)

    Source: Kovari et al., Fusion Engineering and Design 89, 3054 (2014)
    '''

    return 0.5 * ne_0(alpha_n, ne_vol_avg) * sp.gamma(0.5) \
               * sp.gamma(alpha_n + 1.) / sp.gamma(alpha_n + 1.5)

def nd_ne(Z_eff, Z_imp):
    '''
    Ratio of deuterium nuclei to electrons
        Z_eff = effective ion charge
        Z_imp = impurity ion charge
    '''

    return (Z_imp - Z_eff) / (Z_imp - 1)

def ni_ne(nd_ne, Z_eff, Z_imp):
    '''
    Ratio of ions to electrons
        nd_ne = ratio of deuterium (and/or tritium) density to electron density
        Z_eff = effective Z
        Z_imp = impurity ion charge
    '''

    return nd_ne + (Z_eff - 1.) / (Z_imp**2 - Z_imp)

def sudo_density(P, B, a, R):
    '''
    Sudo (line-averaged) density limit (m^-3)
        P = heating power (W)
        B = magnetic field (T)
        a = minor radius (m)
        R = major radius (m)
    
    Source: Sudo et al., Nuclear Fusion 30, 11 (1990)
    '''

    return 0.25 * 1.e20 * np.sqrt((P*1.e-6) * B / (a**2 * R))

def tauE_iss04(a,R,P,ne,B,iota):
    '''
    Energy confinement time (seconds) according to the ISS04 scaling law.
    Note that inputs are all taken to be in SI units where applicable.
        a    = minor radius (m)
        R    = major radius (m)
        P    = absorbed heating power (W)
        ne   = line-averaged electron density (m^-3)
        B    = magnetic field (T)
        iota = rotational transform at 2/3 of the minor radius

    Source: Yamada et al., Nuclear fusion 45, 1684 (2005)
    '''

    return 0.134 * a**2.28 * R**0.64 * (P*1.e-6)**-0.61 * (ne*1.e-19)**0.54 \
                 * B**0.84 * iota**0.41

def tauE_iss95(a,R,P,ne,B,iota):
    '''
    Energy confinement time (seconds) according to the ISS95 scaling law.
    Note that inputs are all taken to be in SI units where applicable.
        a    = minor radius (m)
        R    = major radius (m)
        P    = absorbed heating power (W)
        ne   = line-averaged electron density (m^-3)
        B    = magnetic field (T)
        iota = rotational transform at 2/3 of the minor radius

    Source: U. Stroth et al., Nuclear Fusion 36, 1063 (1996)
    '''

    return 0.079 * a**2.21 * R**0.65 * (P*1.e-6)**-0.59 * (ne*1.e-19)**0.51 \
                 * B**0.83 * iota**0.4

def W_plasma(tauE, P):
    '''
    Plasma stored energy (J)
        tauE = energy confinement time (s)
        P    = total heating power (W)
    '''

    return P * tauE

def T_wtd_vol_avg(ne_vol_avg, W_tot, V, ni_ne):
    '''
    Density-weighted volume-averaged temperature (eV)
        ne    = volume-averaged electron density (m^-3)
        W_tot = plasma stored energy (J)
        V     = plasma volume (m^3)
        ni_ne = ion/electron density ratio
    '''

    return (2./3.) * W_tot / (qe * V * ne_vol_avg * (1. + ni_ne))

def T_0(alpha_T, alpha_n, T_wtd_vol_avg):
    '''
    Central temperature (eV)
        alpha_T       = temperature profile coefficient: 
                        T(rho) = T_0*(1 - rho^2)^alpha_T
        alpha_n       = density profile coefficient: 
                        n(rho) = n_0*(1 - rho^2)^alpha_n
        T_wtd_vol_avg = density-weighted volume-averaged temperature 
    
    Source: Kovari et al., Fusion Engineering and Design 89, 3054 (2014)
    '''

    return T_wtd_vol_avg * (1. + alpha_T + alpha_n) / (1. + alpha_n)

def beta_th(W_plasma, V, B):
    '''
    Thermal plasma beta
        W_plasma = plasma stored energy (J)
        V        = plasma volume (m^3)
        B        = magnetic field strength (T)
    '''

    return (2./3.) * (W_plasma / V) / ( B**2 / (2 * mu0) )

def Ip_eff_spreadsheet(R_maj, B_T):
    ''' 
    Effective plasma current, according to the spreadsheet (A)
        R_maj = major radius (m)
        B_T   = toroidal field (T)
    '''

    return 4451858.06365629 * R_maj * B_T / (2.18 * 6.)

def beta_p_spreadsheet(beta, B, r_min, Ip_eff):
    '''
    Poloidal beta as defined in the spreadsheet
        beta   = total beta
        B      = total magnetic field strength (T)
        r_min  = minor radius (R_maj/A, without accounting for elongation)
        Ip_eff = effective plasma current (A)
    '''

    return beta * (0.25 * r_min**2 * B**2) / Ip**2

def boots_frac_ITER(aspect, beta_p, iota_edge, iota_0):
    '''
    Total bootstrap current as a fraction of the total plasma current I_p,
    calculated according to an ITER scaling law
        aspect    = aspect ratio 
        beta_p    = poloidal beta
        iota_edge = edge value of the rotational transform
        iota_0    = rotational transform on axis

    Source: Uckan, ITER Physics Design Guidlines, ITER Documentation Series No.
        10 (1989); as related in Wilson, Nuclear Fusion 32, 257 (1992).
    '''
    
    C_bs = 1.32 - 0.235 * iota_0/iota_edge + 0.0185 * (iota_0/iota_edge)**2
    return C_bs * (aspect**-0.5 * beta_p)**1.3

#----- Nuclear physics quantities ----------------------------------------------

def dd_fusion_rate_spreadsheet(alpha_n, alpha_T, nd0, T0, V):
    '''
    Estimates the total rate (s^-1) for both types of D-D fusion reactions
    (3He+n and T+p) using a cylindrical approximation for the plasma, 
    as implemented in M. Zarnstorff's spreadsheet. The expression for the
    local rate per reactant <sigma*v> agrees with the tables in Bosch and 
    Hale, Nuclear Fusion 1992 to within about 70%. 
        alpha_n = density profile parameter: 
                  n(rho) = n0*(1 - rho^2)^alpha_n
        alpha_T = temperature profile parameter: 
                  T(rho) = T0*(1 - rho^2)^alpha_T
        nd0     = central deuterium density (m^-3)
        T0      = central temperature (eV)
        V       = plasma volume (m^3)
    '''

    nZones = 20     # number of radial zones for discrete integration
    dr  = 1./nZones
    rho = (np.arange(nZones) + 0.5) * dr
    rdr = rho * dr
    nd  = nd0 * (1. - rho**2)**alpha_n
    T   = T0  * (1. - T**2  )**alpha_T * 1.e-3  # convert to keV

    # Expression for <sigma*v> from the NRL Plasma Formulary
    # Factor of 1.e-6 converts to m^3 s^-1
    # Factor of 2 accounts for the two types of reactions
    sigmav = 1.e-6 * 2. * 2.33e-14 * T**(-2./3.) * np.exp(-18.76 * T**(-1./3.))

    return 2. * V * np.sum(sigmav * 0.5 * nd**2 * rdr)

def dt_fusion_rate_spreadsheet(alpha_n, alpha_T, n0, T0, V):
    '''
    Estimates the total rate (s^-1) for D-T fusion using a cylindrical 
    approximation for the plasma and assuming an equal D-T mixture, 
    as implemented in M. Zarnstorff's spreadsheet. The expression for the
    local rate per reactant <sigma*v> agrees with the tables in Bosch and 
    Hale, Nuclear Fusion 1992 to within about 22%. 
        alpha_n = density profile parameter: 
                  n(rho) = n0*(1 - rho^2)^alpha_n
        alpha_T = temperature profile parameter: 
                  T(rho) = T0*(1 - rho^2)^alpha_T
        n0      = central density (m^-3), equal parts deuterium and tritium
        T0      = central temperature (eV)
        V       = plasma volume (m^3)
    '''

    nZones = 20     # number of radial zones for discrete integration
    dr  = 1./nZones
    rho = (np.arange(nZones) + 0.5) * dr
    rdr = rho * dr
    n   = n0  * (1. - rho**2)**alpha_n
    T   = T0  * (1. - rho**2)**alpha_T * 1.e-3  # convert to keV

    # Expression for <sigma*v> from the spreadsheet implementation
    # Factor of 1.e-6 converts to m^3 s^-1
    sigmav = 1.e-6 * \
        np.exp(-21.377692*T**-0.2935 - 25.204054 - 7.1013427e-2*T \
               + 1.9375451e-4*T**2 + 4.9246592e-6*T**3 + 3.9836572e-8*T**4)

    return 2. * V * np.sum(sigmav * 0.25 * n**2 * rdr)

def dt_fusion_rate_nrl(alpha_n, alpha_T, n0, T0, V):
    '''
    Estimates the total rate (s^-1) for D-T fusion using a cylindrical 
    approximation for the plasma and assuming an equal D-T mixture, 
    using the reaction rate expression in the NRL formulary. The expression for 
    the local rate per reactant <sigma*v> agrees with the tables in Bosch and 
    Hale, Nuclear Fusion 1992 to within about 22%. 
        alpha_n = density profile parameter: 
                  n(rho) = n0*(1 - rho^2)^alpha_n
        alpha_T = temperature profile parameter: 
                  T(rho) = T0*(1 - rho^2)^alpha_T
        n0      = central density (m^-3), equal parts deuterium and tritium
        T0      = central temperature (eV)
        V       = plasma volume (m^3)
    '''

    nZones = 20     # number of radial zones for discrete integration
    dr  = 1./nZones
    rho = (np.arange(nZones) + 0.5) * dr
    rdr = rho * dr
    n   = n0  * (1. - rho**2)**alpha_n
    T   = T0  * (1. - rho**2)**alpha_T * 1.e-3  # convert to keV

    # Expression for <sigma*v> from the NRL formulary
    # Factor of 1.e-6 converts to m^3 s^-1
    sigmav = 1.e-6 * 3.68e-12 * T**(-2./3.) * np.exp(-19.94 * T**(-1./3.))

    return 2. * V * np.sum(sigmav * 0.25 * n**2 * rdr)

#----- Transport quantities ----------------------------------------------------

def nu_ie(Te, Ei, ne, Z, mu):
    '''
    Ion-electron collision frequency (s^-1), assuming v_{i,th}**2 << v_{e,th}**2
        Te = electron temperature (eV)
        Ei = ion energy (eV)
        ne = electron density (m^-3)
        Z  = ion charge number
        mu = ion mass relative to the proton mass

    Source: NRL Plasma Formulary
    '''

    # Coulomb logarithm, valid if Ti*me/mi < (10 eV)*Z**2 < Te
    lambda_ie = 24. - np.log(np.sqrt(ne*1.e6) / Te)
    return 3.2e-9 * (ne*1.e6) * Z**2 * lambda_ie / ( mu * np.sqrt(Te) * Ei )

#----- Coil and blanket quantities ---------------------------------------------

def Bt_coil_inboard(Bt, R_maj, aspect, th_blanket, dist_pl_vv):
    '''
    Estimates the toroidal magnetic field at the location of the coils on
    the inboard side.
        Bt            = toroidal field on axis (T)
        R_maj         = major radius (m)
        aspect        = aspect ratio
        blanket_thick = thickness of the blanket (m)
        dist_pl_vv    = distance between plasma boundary and vacuum vessel (m)
    '''

    r_coil = R_maj * (1. - 1./aspect) - dist_pl_vv - th_blanket
    return Bt * R_maj / r_coil

def Bmax_coil(Bt_coil, Bmax_Bt_coil):
    '''
    Provides the maximum field at the coils based on the inboard toroidal
    field at the coils and a configuration-dependent enhancement factor
        Bt_coil      = toroidal field at the coil on the inboard side (T)
        Bmax_Bt_coil = Ratio of the maximum field at the coil to Bt_coil
    '''
    
    return Bt_coil * Bmax_Bt_coil

#----- Power plant quantities --------------------------------------------------

def P_fus(dt_rate):
    '''
    Total power (W) from fusion reactions
        dt_rate = total fusion reaction rate (s^-1)
    '''

    return dt_rate * 17.58e6 * qe

def P_alpha(P_fus):
    '''
    Total power (W) from the alpha particle kinetic energy from DT fusion 
    reactions
        P_fus = total power from fusion reactions
    '''

    return 0.2 * P_fus
    
def P_n(P_fus):
    '''
    Total power (W) from the neutron kinetic energy from DT fusion reactions
        P_fus = total power from fusion reactions
    '''

    return 0.8 * P_fus

def Q(P_fus, P_aux):
    '''
    Physics Q
    '''

    return P_fus / P_aux

def neutron_wall_load(P_neut, surf_area):
    '''
    Neutron power flux density to the wall. Assumes the wall surface area
    is equal to the plasma surface area; hence, the output is an upper bound.
    '''

    return 0.8 * P_neut / surf_area

def P_th(M_n, P_n, P_alpha, P_aux):
    '''
    Total thermal power (W)
        M_n     = neutron energy multiplier
        P_n     = neutron power (W)
        P_alpha = alpha power (W)
        P_aux   = auxiliary power (W)

    Source: Menard et al., Nuclear Fusion 51, 103014 (2011)
    '''

    return M_n * P_n + P_alpha + P_aux

def P_pump(P_th):
    ''' 
    Pumping power (W)
        P_th = thermal power (W)

    Source: Menard et al., Nuclear Fusion 51, 103014 (2011)
    '''

    return 0.03 * P_th
 
def P_sub(P_th):
    ''' 
    Subsystem power (W)
        P_th = thermal power (W)

    Source: Menard et al., Nuclear Fusion 51, 103014 (2011)
    '''

    return 0.04 * P_th

def Q_eng(eta_th, eta_aux, Q, P_fus, P_pump, P_sub, P_coil, M_n):
    '''
    Overall pilot plant engineering efficiency (i.e. the ratio of electrical
    power produced to electrical power consumed).
        eta_th   = thermal conversion efficiency
        eta_aux  = auxiliary power wall-plug efficiency
        Q        = P_fus / P_aux
        P_fus    = total fusion power (W)
        P_pump   = pumping power (W)
        P_sub    = subsystem power (W)
        P_coil   = power lost in normally-conducting coils (W)
        M_n      = neutron energy multiplier

    Source: Menard et al., Nuclear Fusion 51, 103014 (2011)
    Note: P_control is included in P_sub in this version
    '''

    return eta_th * eta_aux * Q * (4.*M_n + 1. + 5./Q + 5.*P_pump/P_fus) / \
           (5. * (1. + eta_aux * Q * (P_pump + P_sub + P_coil)/P_fus))

def Q_eng_spreadsheet(eta_th, eta_aux, P_th, P_aux, P_pump, P_sub, P_coil):
    '''
    Overall pilot plant engineering efficiency (i.e., the ratio of electrical
    power produced to electrical power consumed.
        eta_th   = thermal conversion efficiency
        eta_aux  = auxiliary power wall-plug efficiency
        P_th     = total thermal power (W)
        P_pump   = pumping power (W)
        P_sub    = subsystem power (W)
        P_coil   = power lost in normally-conducting coils (W)

    This is slightly modified from Q_eng (based on the Menard paper) in that
    it does not count P_pump in both the produced power (numerator) and 
    consumed power (demoninator); rather, P_pump is only counted as consumed
    power.
    '''

    return eta_th*P_th / (P_aux/eta_aux + P_pump + P_sub + P_coil)

def P_el_net(eta_th, P_th, Q_eng):
    '''
    Net electric power output (W)
        eta_th = thermal conversion efficiency
        P_th   = total thermal power (W)
        Q_eng  = ratio of power produced to power consumed
    '''

    return eta_th * P_th * (1. - 1./Q_eng)

def grad_tauE_iss04(a,R,P,ne,B,iota,output='vector'):
    '''
    Gradient of the ISS04 confinement time according to the input parameters.
    '''

    grad = np.zeros(6)
    a_fac    = a**2.28
    R_fac    = R**0.64
    P_fac    = (P*1.e-6)**-0.61
    ne_fac   = (ne*1.e-19)**0.54
    B_fac    = B**0.84
    iota_fac = iota**0.41

    grad[0] = 0.134 * 2.28*a**1.28 * R_fac * P_fac * ne_fac * B_fac * iota_fac
    grad[1] = 0.134 * a_fac * 0.64*R**-0.36 * P_fac * ne_fac * B_fac * iota_fac
    grad[2] = 0.134 * a_fac * R_fac * -0.61*1.e-6*(P*1.e-6)**-1.61 \
                    * ne_fac * B_fac * iota_fac
    grad[3] = 0.134 * a_fac * R_fac * P_fac * 0.54*1.e-19*(1.e-19*ne)**-0.46 \
                    * B_fac * iota_fac
    grad[4] = 0.134 * a_fac * R_fac * P_fac * ne_fac * 0.84*B**-0.16 * iota_fac
    grad[5] = 0.134 * a_fac * R_fac * P_fac * ne_fac * B_fac * 0.41*iota**-0.59

    if output=='vector':
        return grad
    elif output=='dict':
        return {'a':grad[0], 'R':grad[1], 'P':grad[2], 'ne':grad[3], \
                'P':grad[4], 'iota':grad[5]}


