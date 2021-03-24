# Library of routines for working with ASKAPsoft Self Calibration data, e.g. cont_gains_cal_SB10944_GASKAP_M344-11B_T0-0A.beam00.tab.
# These are mostly focussed around plotting the phase solutions and identifying jumps or failures in these solutions. Note that this module requires CASA support.
# The code is based on work by Chenoa Tremblay and Emil Lenc.

# Author James Dempsey
# Date 18 Oct 2020

import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from casacore.tables import *
import seaborn as sns

class SelfCalSolutions:
      # phase is [time, beam, ant, pol]

    def __init__(self):
        """Initialises parameters for reading a selfcal table
        """
        self.nsol = None
        self.nant = None
        self.nbeam = 36
        self.npol = None
        # selfcal is an array in order [time, beam, ant, pol] of phase angle and amplitude value
        self.selfcal = None
        self.selfcal_times = None
        self.selfcal_flags = None
        self.field = None

    def load(self, base_dir):
        flist = glob.glob(base_dir + "/cont_gains*tab")
        flist.sort()
        filename = flist[0]
        print (filename)
        pos = filename.find("beam")
        if pos == -1:
            raise Exception("Can't find beam information in " + filename)
        wildcard = filename[:pos+4] + "??" + filename[pos+6:]
        flist = glob.glob(wildcard)
        flist.sort()
        first_beam = flist[0]
        tb = table(first_beam, readonly=True, ack=False)
        t_vals = tb.getcol("TIME")
        sc_vals = tb.getcol("GAIN",1,1)
        self.selfcal_times = t_vals[1:]
        self.nsol = t_vals.shape[0] - 1
        gain_shape = sc_vals.shape
        self.npol = gain_shape[3]
        self.nant = gain_shape[2]
        tb.close()
        self.selfcal = np.zeros((self.nsol, 36, self.nant, self.npol), dtype=np.complex)
        self.selfcal_flags = np.zeros((self.nsol, 36, self.nant, self.npol), dtype=np.bool)
        for beam in range(self.nbeam):
            fname = wildcard.replace("??", "%02d" %(beam))
            if os.path.exists(fname) == False:
                continue
            tb = table(fname, readonly=True, ack=False)
            t_vals = tb.getcol("TIME", 1, self.nsol)
            sc_vals = tb.getcol("GAIN", 1, self.nsol)
            flag_vals = tb.getcol("GAIN_VALID", 1, self.nsol)
            for index in range(self.nsol):
                self.selfcal[index, beam] = sc_vals[index, 0, :, :]
                self.selfcal_flags[index, beam] = np.invert(flag_vals[index, 0, :, :])
        self.selfcal[np.where(self.selfcal_flags)] = np.nan
        self.field = os.path.basename(base_dir)
        print("Read %d solutions, %d antennas, %d beams, %d polarisations" %(self.nsol, self.nant, self.nbeam, self.npol))
                
    def plotGains(self, ant, outFile = None):
        fig = plt.figure(figsize=(14, 14))
        amplitudes = np.abs(self.selfcal)
        phases = np.angle(self.selfcal, deg=True)
        times = np.array(range(self.nsol))
        plt.subplot(1, 1, 1)
        if self.nant == 36:
            plt.title("ak%02d" %(ant+1), fontsize=8)
        else:
            plt.title("ant%02d" %(ant), fontsize=8)
        for beam in range(self.nbeam):
            plt.plot(times, phases[:,beam,ant,0], marker=None, label="beam %d" %(beam))
#            plt.plot(times, phases[:,ant,beam,1], marker=None, color="red")
            plt.ylim(-200.0, 200.0)
            #rms = np.sqrt(np.mean(np.square(phases[:,beam,ant,0])))
            #print ("ant ak{:02d} beam {:02d} rms={:.2f}".format(ant+1, beam, rms))

        plt.legend()

        plt.tight_layout()
        if outFile == None:
            plt.show()
        else:
            plt.savefig(outFile)
        plt.close()
                
def _plot_ant_phase(sc, ant, outFile = None):
    fig = plt.figure(figsize=(14, 14))
    amplitudes = np.abs(sc.selfcal)
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))
    ax = plt.subplot(1, 1, 1)
    if sc.nant == 36:
        plt.title("ak%02d" %(ant+1), fontsize=8)
    else:
        plt.title("ant%02d" %(ant), fontsize=8)

    low = np.nanpercentile(phases[:,:,ant,0], 2.5, axis=(1))
    high = np.nanpercentile(phases[:,:,ant,0], 97.5, axis=(1))
    colours = sns.color_palette()

    ax.plot(np.nanmedian(phases[:,:,ant,0], axis=(1)), color=colours[0], label='median')
    ax.fill_between(range(phases.shape[0]), low, high, color=colours[0], alpha= .2, label=r'95\% range')
    ax.plot(np.nanmax(phases[:,:,ant,0], axis=(1)), color=colours[1], ls=':', label='maximum')
    ax.plot(np.nanmin(phases[:,:,ant,0], axis=(1)), color=colours[1], ls=':', label='minimum')

    plt.ylim(-200.0, 200.0)
    plt.legend()

    plt.tight_layout()
    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile)
    plt.close()


def _plot_rms_map(sc, field, outFile = None):
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))
    #rms = np.sqrt(np.nanmean(np.square(phases[:,:,:,:]), axis=0))
    rms = np.std(phases[:,:,:,:], axis=0)
    print (np.nanmin(rms), np.nanmedian(rms), np.nanmax(rms))

    ant_list = ['ak{:02}'.format(i+1) for i in range(36)]
    beam_list = ['b{:02}'.format(i) for i in range(36)]

    sns.set()
    fig, axs = plt.subplots(1, 2, figsize=(20,8))
    pol = ['XX', 'YY']
    for i, ax in enumerate(axs):
        sns.heatmap(rms[:,:,i].transpose(), ax=ax, cmap='GnBu', square=True, xticklabels=beam_list, yticklabels=ant_list, 
                    vmin=0, vmax=40, linewidths=.5, cbar_kws={"shrink": .9, "label": 'Phase Standard Deviation (deg)'})
        ax.set_title('Self-cal phase for %s pol %s' % (field, pol[i]))
        ax.set_xlabel(r'Beam')
    axs[0].set_ylabel(r'Antenna')

    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()


def _plot_summary_phases(sc, field, outFile = None):
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))

    sns.set()
    fig, axs = plt.subplots(1, 2, figsize=(20,8))
    pol = ['XX', 'YY']
    colours = sns.color_palette()

    for i, ax in enumerate(axs):
        low = np.nanpercentile(phases[:,:,:,i], 2.5, axis=(1,2))
        high = np.nanpercentile(phases[:,:,:,i], 97.5, axis=(1,2))
        low_ant = np.nanmin(np.nanpercentile(phases[:,:,:,i], 2.5, axis=(1)), axis=1)
        high_ant = np.nanmax(np.nanpercentile(phases[:,:,:,i], 97.5, axis=(1)), axis=1)
        low_med = np.nanmin(np.nanmedian(phases[:,:,:,i], axis=(1)), axis=1)
        high_med = np.nanmax(np.nanmedian(phases[:,:,:,i], axis=(1)), axis=1)
        print (low.shape, low_ant.shape)

        ax.fill_between(range(phases.shape[0]), low_ant, high_ant, color=colours[2], alpha= .2, label="95 percentile range")
        ax.plot(np.nanmedian(phases[:,:,:,i], axis=(1,2)), color=colours[0], label="median")
        ax.fill_between(range(phases.shape[0]), low_med, high_med, color=colours[0], alpha= .4, label="median range")
        ax.plot(np.nanmax(phases[:,:,:,i], axis=(1,2)), color=colours[1], ls=':', alpha=.6, label="maximum")
        ax.plot(np.nanmin(phases[:,:,:,i], axis=(1,2)), color=colours[1], ls=':', alpha=.6, label="minimum")

        ax.set_title('Self-cal phase for %s pol %s' % (field, pol[i]))
        ax.set_xlabel(r'Time (Integration number)')
        ax.set_ylim(-200.0, 200.0)
        ax.legend()
        
    axs[0].set_ylabel(r'Phase (deg)')

    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()


def _plot_median_phases(sc, field, outFile = None):
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))

    sns.set()
    fig, axs = plt.subplots(1, 2, figsize=(20,8))
    pol = ['XX', 'YY']
    colours = sns.color_palette()

    for i, ax in enumerate(axs):
        means = np.nanmedian(phases[:,:,:,i], axis=1)

        for ant in range(36):
            if ant > 30:
                ax.plot(means[:,ant], label="ak%02d" %(ant+1), lw=2, zorder=2)
            else:
                ax.plot(means[:,ant], color='grey', lw=1, zorder=1)

        ax.set_title('Median self-cal phase for %s pol %s' % (field, pol[i]))
        ax.set_xlabel(r'Time (Integration number)')
    
    axs[0].legend()
    axs[0].set_ylabel(r'Phase (deg)')

    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()


def _plot_ant_phases(sc, field, outFile = None):
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))

    sns.set()
    fig, axs = plt.subplots(1, 2, figsize=(20,8))
    pol = ['XX', 'YY']
    colours = sns.color_palette()

    for i, ax in enumerate(axs):
        means = np.nanmedian(phases[:,:,:,i], axis=1)

        for ant in range(36):
            if ant > 30:
                ax.plot(means[:,ant], label="ak%02d" %(ant+1), lw=2, zorder=2)
            else:
                ax.plot(means[:,ant], color='grey', lw=1, zorder=1)

        ax.set_title('Median self-cal phase for %s pol %s' % (field, pol[i]))
        ax.set_xlabel(r'Time (Integration number)')
    
    axs[0].legend()
    axs[0].set_ylabel(r'Phase (deg)')

    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()


def _plot_all_phases(sc, field, outFile = None):
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))

    sns.set()
    fig, axs = plt.subplots(6, 12, figsize=(40,16))
    pols = ['XX', 'YY']
    colours = sns.color_palette()

    for i, pol in enumerate(pols):
        for ant in range(36):
            ax = axs[ant // 6, i*6+ant%6]
            for beam in range(sc.nbeam):
                ax.plot(times, phases[:,beam,ant,i], marker=None, label="beam %d" %(beam))
            ax.set_ylim(-200.0, 200.0)
            ax.set_title('Phases for ak%02d pol %s' % (ant+1, pol[i]))

        #ax.set_xlabel(r'Time (Integration number)')
    
    #axs[0].legend()
    axs[0,0].set_ylabel(r'Phase (deg)')

    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight', dpi=300)
    plt.close()


def _plot_amp_rms_map(sc, field, outFile = None):
    amplitudes = np.absolute(sc.selfcal)
    times = np.array(range(sc.nsol))
    rms = np.std(amplitudes[:,:,:,:], axis=0)
    print (np.nanmin(rms), np.nanmedian(rms), np.nanmax(rms))

    ant_list = ['ak{:02}'.format(i+1) for i in range(36)]
    beam_list = ['b{:02}'.format(i) for i in range(36)]

    sns.set()
    fig, axs = plt.subplots(1, 2, figsize=(20,8))
    pol = ['XX', 'YY']
    for i, ax in enumerate(axs):
        sns.heatmap(rms[:,:,i].transpose(), ax=ax, cmap='GnBu', square=True, xticklabels=beam_list, yticklabels=ant_list, 
                    vmin=0, vmax=0.1, linewidths=.5, cbar_kws={"shrink": .9, "label": 'Bandpass Standard Deviation (Jy)'})
        ax.set_title('Bandpass stability for %s pol %s' % (field, pol[i]))
        ax.set_xlabel(r'Beam')
    axs[0].set_ylabel(r'Antenna')

    #plt.tight_layout()
    if outFile == None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()


def prepare_self_cal_set(folder):
    """
    Prepare a set of self cal solutions for analysis.

    Parameters
    ----------
    folder: path
        Path to the folder containing the self cal solution files. Normally named after the field/interleave.

    Returns
    -------
    The SelfCalSolutions object for use by other calls.
    """
    sc = SelfCalSolutions()
    sc.load(folder)
    return sc



def plot_self_cal_set(sc, fig_folder):
    """
    Produce plots for a set of self calibration solutions for a field.

    Parameters
    ----------
    sc: SelfCalSolutions
        The loaded self cal solutions object for the field/interleave.
    fig_folder: string
        Path to the folder we should put any plots or reports in.

    Returns
    -------
    The paths to the RMS map plot and the summary plot produced for this field.
    """
    rms_map_plot = fig_folder + '/sc_heatmap_{}.png'.format(sc.field)  
    summary_plot = fig_folder + '/sc_summary_{}.png'.format(sc.field)  
    all_phases_plot = fig_folder + '/sc_phases_{}.png'.format(sc.field)  
    _plot_rms_map(sc, sc.field, rms_map_plot)
    _plot_summary_phases(sc, sc.field, summary_plot)
    _plot_all_phases(sc, sc.field, all_phases_plot)
    return rms_map_plot, summary_plot, all_phases_plot


def calc_phase_stability(sc, phase_rms_max=40):
    """
    Calculate summary statistics of the phase stability as recorded in the self-cal solution.

    Parameters
    ----------
    sc: SelfCalSolutions
        The loaded self cal solutions object for the field/interleave.
    phase_rms_max: double
        The maximum allowed median rms before a beam or antenna is classified as bad.

    Returns
    -------
    The number of bad beams and bad antennas.
    """
    phases = np.angle(sc.selfcal, deg=True)
    times = np.array(range(sc.nsol))
    rms = np.std(phases[:,:,:,:], axis=0)
    # phase is [time, beam, ant, pol]

    bad_beams = []
    bad_ant = []
    for i in range(2): # polarisations XX and YY
        bad_ant.append(np.median(rms[:,:,i], axis=0) >= phase_rms_max)
        bad_beams.append(np.median(rms[:,:,i], axis=1) >= phase_rms_max)
    bad_ant_either = bad_ant[0] | bad_ant[1]
    bad_beam_either = bad_beams[0] | bad_beams[1]
    print('ants', bad_ant_either)
    print('beams', bad_beam_either)
    return np.sum(bad_beam_either), np.sum(bad_ant_either)


def find_field_folder(cube, image, field_name):
    potential_parent_dirs = []
    if cube:
        potential_parent_dirs.append(os.path.dirname(cube))
    if image:
        potential_parent_dirs.append(os.path.dirname(image))

    for parent in potential_parent_dirs:
        field_dir = parent + '/' + field_name
        if os.path.isdir(field_dir):
            return field_dir
    
    return None
