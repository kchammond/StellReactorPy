#!/usr/bin/env python

import copy
import numpy as np
import scipy.optimize as op
import ReactorEquations as re

#TODO: handle cases where certain input params (ne_nSudo, etc.) aren't supplied
#TODO: make a feature that allows previous outputs to be input in new run
#TODO: add documentation strings and comments
#DONE: handle the case of the ad-hoc 1.7 MW value of P_sub

class param(object):
    '''
    Stores values and metata for a given reactor parameter.
    '''

    def __init__(self, name, unit, scale, desc='', \
                 value=None, minval=1.e-8, maxval=np.inf, opt=False):

        self.name = name
        self.unit = unit
        self.value = value
        self.scale = scale
        self.desc  = desc
        self.minval = minval
        self.maxval = maxval
        self.opt    = opt

    def set(self, unscaled_value):
        '''
        Used for setting a parameter value from an unscaled value
        '''

        self.value = unscaled_value / self.scale

    def data(self):
        '''
        Used for obtaining the unscaled (typically in SI units) value
        '''

        if self.value == None:
            raise ValueError("No value stored for parameter %s" % (self.name))

        return self.value * self.scale

class constraintBuilder():

    def buildConstraints(constraints, optPar, inPar):

        nConstr = len(constraints)
        constraints = constraints

        minvals = np.zeros(nConstr)
        maxvals = np.zeros(nConstr)
        funcs   = [[]]*nConstr

        for i in range(nConstr):
            minvals[i] = 0.
            if constraintBuilder.constraintKey[constraints[i]]['type'] == 'eq':
                maxvals[i] = 0.
            else:
                maxvals[i] = 1.
            funcs[i] = constraintBuilder.constraintKey[constraints[i]]['fun']

        constr_func = lambda x: \
            constraintBuilder.general_constraint(x, funcs, optPar, inPar)

        return op.NonlinearConstraint(constr_func, minvals, maxvals, \
                                      jac='2-point', hess=op.BFGS())

    def P_heat_consistency(inPar, outPar):
        P_heat   = inPar['P_heat'].value
        P_aux    = inPar['P_aux'].value
        frac_fil = inPar['frac_fil'].value
        P_alpha  = outPar['P_alpha'].value
        return P_heat - P_aux - P_alpha*(1. - frac_fil)

    def density_limit(inPar, outPar):
        ne_nSudo_lim = inPar['ne_nSudo_lim'].value
        n_Sudo   = outPar['n_Sudo'].value
        ne_la    = outPar['ne_la'].value
        return ne_la/n_Sudo/ne_nSudo_lim

    def neut_load_limit(inPar, outPar):
        neut_wall_load = outPar['neut_wall_load'].value
        max_wall_load  = inPar['max_neut_load'].value
        return neut_wall_load/max_wall_load

    def P_el_target(inPar, outPar):
        return outPar['P_el_net'].value - inPar['P_el_targ'].value

    def P_heat_target(inPar, outPar):
        return inPar['P_heat'].value - inPar['P_heat_targ'].value

    def beta_th_limit(inPar, outPar):
        return outPar['beta_th'].value/inPar['beta_th_lim'].value

    def Bt_coil_limit(inPar, outPar):
        return outPar['Bt_coil'].value/inPar['Bt_coil_lim'].value

    def T0_limit(inPar, outPar):
        return outPar['T0'].value/inPar['T0_lim'].value

    constraintKey= { \
        'P_heat_consistency': {'fun': P_heat_consistency, 'type': 'eq'  }, \
        'density_limit':      {'fun': density_limit,      'type': 'ineq'}, \
        'neut_load_limit':    {'fun': neut_load_limit,    'type': 'ineq'}, \
        'P_el_target':        {'fun': P_el_target,        'type': 'eq'  }, \
        'P_heat_target':      {'fun': P_heat_target,      'type': 'eq'  }, \
        'beta_th_limit':      {'fun': beta_th_limit,      'type': 'ineq'}, \
        'Bt_coil_limit':      {'fun': Bt_coil_limit,      'type': 'ineq'}, \
        'T0_limit':           {'fun': T0_limit,           'type': 'ineq'}   }

    def general_constraint(x, funcs, optPar, inPar):

        nFuncs = len(funcs)
        c = np.zeros(nFuncs)

        inPar_x = inPar
        for i in range(len(x)):
            inPar[optPar[i]].value = x[i]

        outPar_x = evalParams(inPar_x)

        for i in range(nFuncs): 
            c[i] = funcs[i](inPar_x, outPar_x)

        return c

class costFuncBuilder():

    def buildCostFunc(costs, weights, optPar, inPar):

        nCosts = len(costs)
        costFuncs = [[]]*nCosts
        for i in range(nCosts):
            costFuncs[i] = costFuncBuilder.costFuncKey[costs[i]]

        costFunc = lambda x: \
            costFuncBuilder.general_cost(x, costFuncs, weights, optPar, inPar)

        return costFunc

    def Q_eng_cost(inPar, outPar):
        return -outPar['Q_eng'].value**2

    def P_el_cost(inPar, outPar):
        return -outPar['P_el_net'].value**2

    costFuncKey = { \
        'Q_eng_cost': Q_eng_cost, \
        'P_el_cost':  P_el_cost     }

    def general_cost(x, costFuncs, weights, optPar, inPar):

        inPar_x = inPar
        for i in range(len(x)):
            inPar_x[optPar[i]].value = x[i]

        outPar_x = evalParams(inPar_x)
        
        nCosts = len(costFuncs)
        costVal = 0.
        for i in range(nCosts):
            costVal += weights[i] * costFuncs[i](inPar_x, outPar_x)

        return costVal

def evalParams(inPar, verbose=False):

    # Input parameters
    R_maj      = inPar['R_maj'].data()
    aspect     = inPar['aspect'].data()
    elong      = inPar['elong'].data()
    Bt         = inPar['Bt'].data()
    iota_23    = inPar['iota_23'].data()
    H          = inPar['H'].data()
    ne_vol_avg = inPar['ne_vol_avg'].data()
    alpha_n    = inPar['alpha_n'].data()
    alpha_T    = inPar['alpha_T'].data()
    Z_eff      = inPar['Z_eff'].data()
    Z_imp      = inPar['Z_imp'].data()
    th_blanket = inPar['th_blanket'].data()
    dist_pl_vv = inPar['dist_pl_vv'].data()
    eta_th     = inPar['eta_th'].data()
    eta_aux    = inPar['eta_aux'].data()
    M_n        = inPar['M_n'].data()
    P_aux      = inPar['P_aux'].data()
    P_heat     = inPar['P_heat'].data()

    pl_vol       = re.plasma_volume(R_maj, aspect)
    pl_surf_area = re.plasma_surface_area(R_maj, aspect, elong)

    Bt_coil = re.Bt_coil_inboard(Bt, R_maj, aspect, th_blanket, dist_pl_vv)
    
    nd_ne = re.nd_ne(Z_eff, Z_imp)
    ni_ne = re.ni_ne(nd_ne, Z_eff, Z_imp)

    ne0 = re.ne_0(alpha_n, ne_vol_avg)
    nd0 = nd_ne * ne0
    ne_la = re.ne_la(alpha_n, ne_vol_avg)

    if inPar['tauE_type'].value.lower() == 'iss04':
        tauE = H*re.tauE_iss04(R_maj/aspect, R_maj, P_heat, ne_la, Bt, iota_23)
    elif inPar['tauE_type'].value.lower() == 'iss95':
        tauE = H*re.tauE_iss95(R_maj/aspect, R_maj, P_heat, ne_la, Bt, iota_23)
    else:
        raise ValueError("Unrecognized value for tauE_type")

    W_plasma = re.W_plasma(tauE, P_heat)
    T_wtd_vol_avg = re.T_wtd_vol_avg(ne_vol_avg, W_plasma, pl_vol, ni_ne)
    T0 = re.T_0(alpha_T, alpha_n, T_wtd_vol_avg)
    
    dt_neut_rate = \
        re.dt_fusion_rate_spreadsheet(alpha_n, alpha_T, nd0, T0, pl_vol)
    #dt_neut_rate = \
    #    re.dt_fusion_rate_nrl(alpha_n, alpha_T, nd0, T0, pl_vol)
    
    P_fus   = re.P_fus(dt_neut_rate)
    P_neut  = re.P_n(P_fus)
    P_alpha = re.P_alpha(P_fus)

    beta_th = re.beta_th(W_plasma, pl_vol, Bt)

    n_Sudo = re.sudo_density(P_alpha+P_aux, Bt, \
                             R_maj/aspect/np.sqrt(elong), R_maj)
    
    P_th    = re.P_th(M_n, P_neut, P_alpha, P_aux)
    P_pump  = re.P_pump(P_th)
    P_coil  = 0.
    if not inPar['P_sub_val'].value:
        P_sub = re.P_sub(P_th)
    else:
        P_sub = inPar['P_sub_val'].data()

    Q = re.Q(P_fus, P_aux)

    Q_eng = re.Q_eng(eta_th, eta_aux, Q, P_fus, P_pump, P_sub, P_coil, M_n)

    P_el_net = re.P_el_net(eta_th, P_th, Q_eng)
    
    neut_wall_load = re.neutron_wall_load(P_neut, pl_surf_area)

    outPar = initOutPar()
    outPar['pl_vol'].set(pl_vol)
    outPar['pl_surf_area'].set(pl_surf_area)
    outPar['nd_ne'].set(nd_ne)
    outPar['ni_ne'].set(ni_ne)
    outPar['ne0'].set(ne0)
    outPar['nd0'].set(nd0)
    outPar['ne_la'].set(ne_la)
    outPar['n_Sudo'].set(n_Sudo)
    outPar['tauE'].set(tauE)
    outPar['W_plasma'].set(W_plasma)
    outPar['T0'].set(T0)
    outPar['T_wtd_vol_avg'].set(T_wtd_vol_avg)
    outPar['dt_neut_rate'].set(dt_neut_rate)
    outPar['Bt_coil'].set(Bt_coil)
    outPar['P_fus'].set(P_fus)
    outPar['beta_th'].set(beta_th)
    outPar['P_neut'].set(P_neut)
    outPar['P_alpha'].set(P_alpha)
    outPar['P_th'].set(P_th)
    outPar['P_pump'].set(P_pump)
    outPar['P_sub'].set(P_sub)
    outPar['P_coil'].set(P_coil)
    outPar['Q'].set(Q)
    outPar['Q_eng'].set(Q_eng)
    outPar['P_el_net'].set(P_el_net)
    outPar['neut_wall_load'].set(neut_wall_load)
 
    return outPar


def initInPar():
    '''
    Creates the dictionary of input parameters for the general evaluation 
    function. 
    '''

    inPar = { \
        'R_maj':         param('R_maj',         'm',          1.   ,   \
            desc='Major radius'                                     ), \
        'aspect':        param('aspect',        '',           1.   ,   \
            desc='Aspect ratio'                                     ), \
        'elong':         param('elong',         '',           1.   ,   \
            desc='Elongation'                                       ), \
        'Bt':            param('Bt',            'T',          1.   ,   \
            desc='Toroidal magnetic field',                         ), \
        'iota_23':       param('iota_23',       '',           1.   ,   \
            desc='Rotational transform at 2/3 of the minor radius'  ), \
        'ne_vol_avg':    param('ne_vol_avg',    '10^20 m^-3', 1.e20,   \
            desc='Volume-averaged electron density'                 ), \
        'alpha_n':       param('alpha_n',       '',           1.   ,   \
            desc='Density profile parameter: n = n0*(1 - rho^2)^alpha_n'), \
        'alpha_T':       param('alpha_T',       '',           1.   ,   \
            desc='Temperature profile parameter: T = T0*(1 - rho^2)^alpha_T'), \
        'Z_eff':         param('Z_eff',         '',           1.   ,   \
            desc='Effective ion charge number'                      ), \
        'Z_imp':         param('Z_imp',         '',           1.   ,   \
            desc='Impurity charge number'                           ), \
        'H':             param('H',             '',           1.   ,   \
            desc='Enhancement factor for the energy confinement time'), \
        'tauE_type':     param('tauE_type',     '',           None ,   \
            desc='Scaling law for energy confinement time (iss95, iss04)'), \
        'frac_fil':      param('frac_fil',      '',           1.   ,   \
            desc='Fast ion loss fraction'                           ), \
        'dist_pl_vv': param('dist_pl_vv',       'm',          1.   ,   \
            desc='Distance between plasma boundary and vac vessel'  ), \
        'th_blanket':    param('th_blanket',    'm',          1.   ,   \
            desc='Blanket thickness',                               ), \
        'eta_th':        param('eta_th',        '',           1.   ,   \
            desc='Power plant thermal conversion efficiency'        ), \
        'eta_aux':       param('eta_aux',       '',           1.   ,   \
            desc='Wall plug efficiency for auxiliary heating power' ),  \
        'M_n':           param('M_n',           '',           1.   ,   \
            desc='Neutron energy multiplier'                        ), \
        'P_aux':         param('P_aux',         'MW',         1.e6 ,   \
            desc='Absorbed auxiliary heating power'                 ), \
        'P_heat':        param('P_heat',        'MW',         1.e6 ,   \
            desc='Total plasma heating power'                       ), \
        'P_sub_val':     param('P_sub_val',     'MW',         1.e6 ,   \
            desc='Subsystem power (calculated as 0.04*P_th if not specified)'),\
        'P_el_targ':     param('P_el_targ',     'MW',         1.e6 ,   \
            desc='Target value for the net electric power'          ), \
        'P_heat_targ':   param('P_heat_targ',   'MW',         1.e6 ,   \
            desc='Target value for the total heating power'         ), \
        'ne_nSudo_lim':  param('ne_nSudo_lim',  '',           1.   ,   \
            desc='Max ratio of line-averaged density to Sudo limit' ), \
        'beta_th_lim':   param('beta_th_lim',   '',           1.   ,   \
            desc='Maximum value for the thermal plasma beta'        ), \
        'Bt_coil_lim':   param('Bt_coil_lim',   'T',          1.   ,   \
            desc='Maximum value for Bt at the coil, inboard side'   ), \
        'T0_lim':        param('T0_lim',        'keV',        1.e3 ,   \
            desc='Maximum value peak/central plasma temperature'    ), \
        'max_neut_load': param('max_neut_load', 'MW/m^2',     1.e6 ,   \
            desc='Maximum permissible neutron wall load'            )  \
       }

    # Assign default values to certain parameters
    inPar['tauE_type'].value = 'iss04'

    return inPar

def initOutPar():

    outPar = { \
        'pl_vol':          param('pl_vol',         'm^3',        1.   ,  \
            desc='Plasma volume'),                                       \
        'pl_surf_area':    param('pl_surf_area',   'm^2',        1.   ,  \
            desc='Plasma surface area'),                                 \
        'ni_ne':           param('ni_ne',          '',           1.   ,  \
            desc='Ratio of ion density to electron density'),            \
        'nd_ne':           param('nd_ne',          '',           1.   ,  \
            desc='Ratio of deuterium density to electron density'),      \
        'ne_la':           param('ne_la',          '10^20 m^-3', 1.e20,  \
            desc='Line-averaged electron density'),                      \
        'ne0':             param('ne0',            '10^20 m^-3', 1.e20,  \
            desc='Electron density on axis'),                            \
        'nd0':             param('nd0',            '10^20 m^-3', 1.e20,  \
            desc='Deuterium density on axis'),                           \
        'n_Sudo':          param('n_Sudo',         '10^20 m^-3', 1.e20,  \
            desc='Sudo density limit'),                                  \
        'tauE':            param('tauE',           's',          1.   ,  \
            desc='Energy confinement time'),                             \
        'W_plasma':        param('W_plasma',       'MJ',         1.e6 ,  \
            desc='Plasma stored energy'),                                \
        'T0':              param('T0',             'keV',        1.e3 ,  \
            desc='Temperature on axis'),                                 \
        'T_wtd_vol_avg':   param('T_wtd_vol_avg',  'keV',        1.e3 ,  \
            desc='Density-weighted volume-averaged temperature'),        \
        'dt_neut_rate':    param('dt_neut_rate',   '10^20 s^-1', 1.e20,  \
            desc='Local DT fusion reaction rate'),                       \
        'beta_th':         param('beta_th',        '',           1.   ,  \
            desc='Thermal beta'),                                        \
        'Bt_coil':         param('Bt_coil',        'T',          1.   ,  \
            desc='Estimated toroidal field at the coils, inboard side'), \
        'P_fus':           param('P_fus',          'MW',         1.e6 ,  \
            desc='Total fusion power'),                                  \
        'P_neut':          param('P_neut',         'MW',         1.e6 ,  \
            desc='Fusion power carried by neutrons'),                    \
        'P_alpha':         param('P_alpha',        'MW',         1.e6 ,  \
            desc='Fusion power carried by alpha particles'),             \
        'P_th':            param('P_th',           'MW',         1.e6 ,  \
            desc='Total thermal power absorbed by reactor walls'),       \
        'P_pump':          param('P_pump',         'MW',         1.e6 ,  \
            desc='Power consumed by pumping systems'),                   \
        'P_sub':           param('P_sub',          'MW',         1.e6 ,  \
            desc='Power consumed by subsystems and control systems'),    \
        'P_coil':          param('P_coil',         'MW',         1.e6 ,  \
            desc='Power consumed by coils'),                             \
        'Q':               param('Q',              '',           1.   ,  \
            desc='Ratio of fusion power ' +                              \
                 'to absorbed auxiliary heating power'),                 \
        'Q_eng':           param('Q_eng',          '',           1.   ,  \
            desc='Ratio of electric power produced ' +                   \
                 'to power consumed by plant'),                          \
        'P_el_net':        param('P_el_net',       'MW',         1.e6 ,  \
            desc='Net electric power output of plant'),                  \
        'neut_wall_load':  param('neut_wall_load', 'MW/m^2',     1.e6 ,  \
            desc='Neutron load on the reactor walls')                    \
       }

    return outPar

def optParams(inVals, constraints=['P_heat_consistency'], optPar=[], \
              optParMin=1.e-8, optParMax=np.inf, costs=[], weights=1., 
              options={}):

    inPar = initInPar()

    # Prepare the input parameters based on user-supplied values
    for name in inVals.keys():
        if type(inVals[name]) == param:
            inPar[name] = inVals[name]
            inPar[name].opt = False 
        else:
            inPar[name].value = inVals[name]
    for name in optPar:
        inPar[name].opt = True

    # If no optimizable parameters are given, perform a simple eval and return
    if len(optPar) == 0:
        outPar = evalParams(inPar)
        return inPar, outPar

    # If optPar is not empty, ensure that it contains P_heat
    if not 'P_heat' in optPar:
        optPar.append('P_heat')
    nOptPar = len(optPar)

    # Prepare bounds for optimizable parameters
    if np.isscalar(optParMin):
        optParMin = np.ones(nOptPar)*optParMin
    if np.isscalar(optParMax):
        optParMax = np.ones(nOptPar)*optParMax
    if len(optParMin) != nOptPar or len(optParMax) != nOptPar:
        raise ValueErr('Array length of optParMin and optParMax must agree ' \
                       + 'with length of optPar for non-scalar optParMin '   \
                       + 'and optParMax')
    bounds = op.Bounds(optParMin, optParMax)

    # Prepare the nonlinear constraint object
    if not 'P_heat_consistency' in constraints:
        constraints.append('P_heat_consistency')
    nConstr = len(constraints)
    nlc = constraintBuilder.buildConstraints(constraints, optPar, inPar)

    # Prepare the cost function
    nCostFunc = len(costs)
    if np.isscalar(weights):
        weights = np.ones(nCostFunc)*weights
    if len(weights) != nCostFunc:
        raise ValueError('Array length of costs and weights must agree ' \
                         + 'for non-scalar weights')
    cf = costFuncBuilder.buildCostFunc(costs, weights, optPar, inPar)

    # Load the initial guesses of the optimizable parameters
    x0 = np.zeros(nOptPar)
    for i in range(nOptPar):
        x0[i] = inPar[optPar[i]].value

    # Run the optimizer
    res = op.minimize(cf, x0, method='trust-constr', constraints=[nlc], \
                      bounds=bounds, jac='2-point', hess=op.BFGS(), \
                      options=options)

    # Update the values of the input parameters
    for i in range(nOptPar):
        inPar[optPar[i]].value = res.x[i]

    outPar = evalParams(inPar)

    return inPar, outPar

def optParamScan(inVals, scanParams, scanVals, \
                 constraints=['P_heat_consistency'], optPar=[], \
                 optParMin=1.e-8, optParMax=np.inf, costs=[], weights=1., 
                 options={}):

    # Determine number of parameters and scan values; check for consistency
    nScanParams = len(scanParams)
    if nScanParams == 1:
        if len(scanVals) != 1:
            scanVals = [scanVals]
        nScan = len(scanVals[0])
    else:
        if len(scanVals) != nScanParams:
            raise ValueError('scanVals must contain one array for each item ' +\
                             'in scanParams')
        else:
            nScan = len(scanVals[0])
            for valArray in scanVals:
                if len(valArray) != nScan:
                    raise ValueError('Each array in scanVals must have the ' + \
                                     'same length')

    # Initialize the input parameters
    inPar = initInPar()

    # Prepare the input parameters based on user-supplied values
    for name in inVals.keys():
        if type(inVals[name]) == param:
            inPar[name] = inVals[name]
            inPar[name].opt = False 
        else:
            inPar[name].value = inVals[name]
    for name in optPar:
        inPar[name].opt = True

    # Lists for input and output parameters at each step of the scan
    inParList = [[]]*nScan
    outParList = [[]]*nScan
    inParList[0] = inPar

    # Perform the scan
    for i in range(nScan):

        for j in range(nScanParams):
            inParList[i][scanParams[j]].value = scanVals[j][i]

        inParList[i], outParList[i] = \
            optParams(inParList[i], constraints=constraints, optPar=optPar, \
                      optParMin=optParMin, optParMax=optParMax, costs=costs, \
                      weights=weights, options=options)

        # Initialize the input parameters for the next scan point
        if i < nScan-1:
            inParList[i+1] = copy.deepcopy(inParList[i])

    return inParList, outParList


