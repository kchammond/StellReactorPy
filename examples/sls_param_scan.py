#!/usr/bin/env/python

'''
sls_param_scan.py

Scans variants of the SLS reactor with different aspect ratios, magnetic
fields, and beta limits. For each parameter combination, finds the major
radius required for 200 MW of net electric power.

Created 2020-09-10
Author: K. C. Hammond <khammond@pppl.gov>
'''

import numpy as np
import pandas as pd
import ReactorSolve as rs
import ReactorIO as rio
import matplotlib.pylab as pl
import os

def interp_inverse(yq, x, y):
    '''
    Evaluates the inverse of a monotonic function y = f(x) through linear 
    interpolation.

        yq: query value of f for which the argument is to be determined
        x:  data points for x
        y:  data points for f(x), assumed to vary monotonically with x
    '''

    indices = np.argsort(y)

    if np.max(y) < yq:
        return np.inf

    elif np.min(y) > yq:
        return -np.inf

    else:
        return np.interp(yq, y[indices], x[indices])


if __name__=='__main__':

    example_name = 'sls_param_scan'

    inVal_sls = dict( R_maj      = 9.4,               \
                      aspect     = 5.5,               \
                      elong      = 3.96,              \
                      Bt         = 6.0,               \
                      iota_23    = 0.57,              \
                      H          = 1.5,               \
                      ne_vol_avg = 6.00,              \
                      alpha_n    = 0.,                \
                      alpha_T    = 1.,                \
                      Z_eff      = 1.1,               \
                      Z_imp      = 6.0,               \
                      th_blanket = 1.0,               \
                      dist_pl_vv = 0.5,               \
                      eta_th     = 0.30,              \
                      eta_aux    = 0.45,              \
                      frac_fil   = 0.05,              \
                      M_n        = 1.1,               \
                      P_aux      = 1.e-6,             \
                      P_heat     = 300.0,             \
                      max_neut_load = 5.,             \
                      P_el_targ  = 1000.0,            \
                      ne_nSudo_lim = 2.,                \
                      beta_th_lim = 0.06,            \
                      P_sub_val  = 1.7,               \
                      tauE_type  = 'iss04'             )

    optPar = ['P_heat', 'P_aux', 'ne_vol_avg']
    const  = ['P_heat_consistency', 'density_limit', 'beta_th_limit']
    costs  = ['Q_eng_cost']

    rmaj_array = inVal_sls['R_maj'] - np.linspace(0., 8., 17)

    betaVals = [0.06, 0.05, 0.04, 0.03]
    BtVals   = [6.00, 7.00, 8.00, 9.00]
    aspectVals = [6.0, 5.5, 5.0, 4.5]
    betaGrid, BtGrid, AGrid, = np.meshgrid(betaVals, BtVals, aspectVals)

    nBeta = len(betaVals)
    nBt   = len(BtVals)
    nAspect = len(aspectVals)
    summaryTable = np.zeros((nBeta*nBt*nAspect, 8))

    dat_dir = 'data/'
    txt_dir = 'text/'
    plt_dir = 'plots/'
    dirNames = [dat_dir, txt_dir, plt_dir]
    if not os.path.isdir(example_name):
        try:
            os.mkdir(example_name)
        except:
            raise RuntimeError('Unable to find or create subdirectory ' \
                               % (example_name))
    for name in dirNames:
        subdir = example_name + '/' + name
        if not os.path.isdir(subdir):
            try:
                os.mkdir(subdir)
            except:
                raise RuntimeError('Unable to find or create directory ' \
                                   % (subdir))
    
    

    dat_prefix = example_name + '/' + dat_dir + example_name + '_'
    txt_prefix = example_name + '/' + txt_dir + example_name + '_'
    plt_prefix = example_name + '/' + plt_dir + example_name + '_'

    print('%10s %10s %10s %10s %10s %10s %10s %10s' \
              % ('aspect', 'beta (%)', 'Bt (T)', 'R_maj (m)', 'P_th (MW)', \
                 'P_fus (MW)', 'V (m^3)', 'Bcoil (T)'))
    print('---------- ---------- ---------- ---------- ---------- ' \
          + '---------- ---------- ----------')

    for k in range(nAspect):
        for j in range(nBeta):
            for i in range(nBt):
    
                A1 = np.floor(AGrid[i,j,k])
                A2 = np.floor(10*(AGrid[i,j,k]-A1))
                Astr = 'A=%dp%d_' % (A1,A2)
                Btstr = 'Bt=%dT_' % (np.round(BtGrid[i,j,k]))
                betastr = 'beta=%02dpct' % (100*betaGrid[i,j,k])
    
                dat_fname = dat_prefix + Astr + Btstr + betastr + '.csv'
                txt_fname = txt_prefix + Astr + Btstr + betastr + '.txt'
                plt_fname = plt_prefix + Astr + Btstr + betastr + '.png'
    
                ylims = {'power':          [-100, 1500], \
                         'density':        [   0,    9], \
                         'Q':              [   0,   11], \
                         'T':              [   0,   35], \
                         'neut_wall_load': [   0,    3]   }
        
                inVal_sls['beta_th_lim'] = betaGrid[i,j,k]
                inVal_sls['Bt'] = BtGrid[i,j,k]
                inVal_sls['aspect'] = AGrid[i,j,k]
        
                inParList, outParList = \
                    rs.optParamScan(inVal_sls, ['R_maj'], [rmaj_array],  \
                                    constraints=const, optPar=optPar, \
                                    costs=costs)
        
                rio.printScanResults(inParList, outParList, \
                                     filename=txt_fname, toScreen=False)
        
                data = rio.scanTable(inParList, outParList)
                data.to_csv(dat_fname)
        
                f1 = rio.makePlots(data, rmaj_array, r'$R_{maj}$ (m)', \
                                   ylims=ylims)
                f1.savefig(plt_fname)
    
                Rmaj_200MW = interp_inverse(200., data.R_maj, data.P_el_net)
                sort_inds = np.argsort(data.R_maj)
                if np.isfinite(Rmaj_200MW):
                    Pth_200MW = \
                        np.interp(Rmaj_200MW, data.R_maj[sort_inds], \
                                  data.P_th[sort_inds])
                    Pfus_200MW = \
                        np.interp(Rmaj_200MW, data.R_maj[sort_inds], \
                                  data.P_fus[sort_inds])
                    V_200MW = \
                        np.interp(Rmaj_200MW, data.R_maj[sort_inds], \
                                  data.pl_vol[sort_inds])
                    Btcoil_200MW = \
                        np.interp(Rmaj_200MW, data.R_maj[sort_inds], \
                              data.Bt_coil[sort_inds])
                else:
                    Pth_200MW = np.nan
                    Pfus_200MW = np.nan
                    V_200MW = np.nan
                    Btcoil_200MW = np.nan

                print(('%10.1f %10.1f %10.1f ' \
                          + '%10.2f %10.2f %10.2f %10.3f %10.2f') \
                      % (aspectVals[k], 100*betaVals[j], BtVals[i], \
                         Rmaj_200MW, Pth_200MW, Pfus_200MW, V_200MW, \
                         Btcoil_200MW))

                summaryTable[k*nBeta*nBt + j*nBt + i,:] = \
                    [aspectVals[k], betaVals[j], BtVals[i], \
                     Rmaj_200MW, Pth_200MW, Pfus_200MW, V_200MW, Btcoil_200MW]
        
    summaryData = \
        pd.DataFrame(data=summaryTable, \
                     columns=['aspect', 'beta', 'Bt', 'Rmaj_200', 'Pth_200', \
                              'Pfus_200MW', 'V_200MW', 'Btcoil_200MW'])
    summaryData.to_csv(dat_prefix + 'summaryData.csv')
