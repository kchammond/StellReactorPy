#!/usr/bin/env/python

import numpy as np
import pandas as pd
import matplotlib.pylab as pl

def printScanResults(inParList, outParList, filename='', toScreen=True):
    '''
    Print all input and output parameters from a scan as a table to the
    standard out.

    Parameters
    ----------
        inParList: list of inPar dictionaries output by scanParams
        outParList: list of outPar dictionaries output by scanParams
        filename: string with the name of a file to save the output to
        toScreen: True if output should be printed to screen 
    '''

    if filename != '':
        toFile = True
        f = open(filename, 'w')
    else:
        toFile = False

    for key in inParList[0].keys():

        printStr = "%16s (%10s): " % (key, inParList[0][key].unit)

        for i in range(len(inParList)):

            if inParList[i][key].value == None:
                valStr = '        -- '
            elif type(inParList[i][key].value) == str:
                valStr = "%10s " % (inParList[i][key].value)
            else:
                valStr = "%10.3g " % (inParList[i][key].value)

            printStr += valStr

        if toScreen:
            print(printStr)

        if toFile:
            f.write(printStr + '\n')

    for key in outParList[0].keys():

        printStr = "%16s (%10s): " % (key, outParList[0][key].unit)

        for i in range(len(outParList)):

            if outParList[i][key].value == None:
                valStr = '        -- '
            elif type(outParList[i][key].value) == str:
                valStr = "%10s " % (outParList[i][key].value)
            else:
                valStr = "%10.3g " % (outParList[i][key].value)

            printStr += valStr

        if toScreen:
            print(printStr)

        if toFile:
            f.write(printStr + '\n')

    if toFile:
        f.close()

def scanTable(inParList, outParList):
    '''
    Organize results from a scan into a Pandas data frame

    Parameters
    ----------
        inParList: list of inPar dictionaries output by scanParams
        outParList: list of outPar dictionaries output by scanParams
    '''

    nScan = len(inParList)
    data = dict()

    for key in inParList[0].keys():
        data[key] = [item[key].value for item in inParList]

    for key in outParList[0].keys():
        data[key] = [item[key].value for item in outParList]

    return pd.DataFrame(data)

def makePlots(data, x_var, x_label, ylims={}):

    figHt = 8
    figWd = 15
    axVgap = 0.05
    axHgap = 0.03
    axVlab = 0.05
    axHlab = 0.03
    axHt = 0.38
    axWd = 0.27
    lw = 2

    pl.rcParams.update({'font.family':      'sans-serif',   \
                        'font.sans-serif':  'Helvetica',    \
                        'mathtext.fontset': 'stixsans',     \
                        'axes.labelsize':   16,             \
                        'xtick.labelsize':  14,             \
                        'ytick.labelsize':  14,             \
                        'legend.fontsize':  14               })

    f1 = pl.figure(figsize=(figWd, figHt))

    axP = f1.add_axes([1*axHlab+1*axHgap+0*axWd, 2*axVlab+2*axVgap+1*axHt, \
                       axWd, axHt])
    axP.plot(x_var, data.P_el_net, label='$P_{net}$', linewidth=lw, \
             color=(0.4, 0.0, 0.7))
    axP.plot(x_var, data.P_aux, label='$P_{aux}$', linewidth=lw, \
             color=(0.8, 0.5, 1.0))
    axP.set_xlabel(x_label)
    axP.set_ylabel("MW")
    if 'power' in ylims:
        axP.set_ylim(ylims['power'])
    axP.grid()
    axP.legend()

    axN = f1.add_axes([2*axHlab+2*axHgap+1*axWd, 2*axVlab+2*axVgap+1*axHt, \
                       axWd, axHt])
    axN.plot(x_var, data.ne_la, label='$\overline{n}_e$', linewidth=lw, \
             color=(0.0, 0.5, 0.0))
    axN.plot(x_var, data.n_Sudo, label='$n_{Sudo}$', linewidth=lw, \
             color=(0.2, 1.0, 0.2), linestyle='--')
    axN.set_xlabel(x_label)
    axN.set_ylabel('$10^{20}$ m$^{-3}$')
    if 'density' in ylims:
        axN.set_ylim(ylims['density'])
    axN.grid()
    axN.legend()

    axQB = f1.add_axes([3*axHlab+3*axHgap+2*axWd, 2*axVlab+2*axVgap+1*axHt, \
                       axWd, axHt])
    axQB.plot(x_var, data.Q_eng, label='$Q_{eng}$', linewidth=lw, 
              color=(0.0, 0.0, 0.6))
    axQB.plot(x_var, 100*data.beta_th, label=r'$\beta_{th}$ (%)', \
              linewidth=lw, color=(0.5, 0.7, 1.0))
    axQB.set_xlabel(x_label)
    axQB.set_ylabel('')
    if 'Q' in ylims:
        axQB.set_ylim(ylims['Q'])
    axQB.grid()
    axQB.legend()

    axT = f1.add_axes([1*axHlab+1*axHgap+0*axWd, 1*axVlab+1*axVgap+0*axHt, \
                       axWd, axHt])
    axT.plot(x_var, data.T0, label='$T_0$', linewidth=lw, \
             color=(0.6, 0.0, 0.0))
    axT.plot(x_var, data.T_wtd_vol_avg, label='$<T>$', linewidth=lw, \
             color=(1.0, 0.4, 0.4))
    axT.set_xlabel(x_label)
    axT.set_ylabel('keV')
    if 'T' in ylims:
        axT.set_ylim(ylims['T'])
    axT.grid()
    axT.legend()

    axL = f1.add_axes([2*axHlab+2*axHgap+1*axWd, 1*axVlab+1*axVgap+0*axHt, \
                       axWd, axHt])
    axL.plot(x_var, data.neut_wall_load, label='Neutron load', \
             linewidth=lw, color=(0.4, 0.4, 0.4))
    axL.set_xlabel(x_label)
    axL.set_ylabel('MW/m$^2$')
    if 'neut_wall_load' in ylims:
        axL.set_ylim(ylims['neut_wall_load'])
    axL.grid()
    axL.legend()

    axPf = f1.add_axes([3*axHlab+3*axHgap+2*axWd, 1*axVlab+1*axVgap+0*axHt, \
                       axWd, axHt])
    rho = np.linspace(0,1,100)
    profile = lambda alpha: (1. - rho**2)**alpha
    if len(data.alpha_T) > 1 and data.alpha_T[1] - data.alpha_T[0] > 1.e-6:
        nAlphas = len(data.alpha_T)
        for i in range(nAlphas):
            tColor = (0.0, 0.0, 0.2 + 0.8*float(i)/float(nAlphas-1))
            axPf.plot(rho, profile(data.alpha_T[i]), \
                      label=r'$T$ profile, $\alpha_T$ = %.2f' % \
                            (data.alpha_T[i]), \
                      linewidth=lw, color=tColor)
    else:
        axPf.plot(rho, profile(data.alpha_T[0]), label='$T$ profile', \
                  linewidth=lw, color=(0.0, 0.0, 0.6))
    axPf.plot(rho, profile(data.alpha_n[0]), label='$n$ profile', \
              linewidth=lw, color=(0.6, 0.0, 0.0))
    axPf.set_xlabel(r'$\rho$')
    axPf.set_ylabel('A.U.')
    axPf.grid()
    axPf.legend()

    return f1


