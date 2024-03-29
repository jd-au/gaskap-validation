# Library of routines for working with ASKAPsoft BPCAL data.
# These are mostly focussed around plotting the bandpasses. Note thta this module requires CASA support.

# Author James Dempsey
# Date 14 Aug 2020

import glob
import os
import re
import time

from casacore import tables
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

def get_cal_bandpass(diagnostics_dir):
    cal_dir = os.path.abspath(diagnostics_dir+'/../BPCAL')
    cal_pattern = cal_dir+'/calparameters*bp*smooth'
    paths = glob.glob(cal_pattern)
    if len(paths) == 0:
        raise Exception("Unable to find a cal bandpass matching " + cal_pattern)
    cal_tab = tables.table(paths[0])

    # Extract the scheduling block id from the file name
    print (os.path.basename(paths[0]))
    name_parts = re.split('[_.]', os.path.basename(paths[0]))
    sbid = 0
    for part in name_parts:
        result = re.findall(r'^SB[0-9]+', part)
        if result:
            print (result)
            sbid = int(result[0][2:])
    
    return cal_tab.BANDPASS[0], sbid

def get_median_bandpasses(bandpass, summary_axis):
    """
    Retrieve the median x and y bandpasses for each beam or antenna.
    """
    start = time.time()
    # Split the bandpass array
    x_bandpasses = np.abs(bandpass[:,:,::2])
    y_bandpasses = np.abs(bandpass[:,:,1::2])

    # Extract the median bandpasses for the summary axis
    x_medians = np.median(x_bandpasses, axis=summary_axis)
    y_medians = np.median(y_bandpasses, axis=summary_axis)

    end = time.time()
    print('Bandpass extracted in {:.02f} s'.format((end - start)))
    return x_medians, y_medians


def get_bandpass_ranges(bandpass, summary_axis, x_medians, y_medians):
    """
    Retrieve the stats for mean bandpasses by either beam or antenna.
    Stats produced are:
    * the median and standard deviation of the mean bandpasses
    * the median and standard deviation of the ranges of the mean bandpasses.
    We use the mean bandpass here to avoid the cost of sorting the full bandpass array.
    """
    
    # Get xx and yy range stats for the mean bandpasses
    x_bp_mins = np.min(x_medians, axis=1)
    x_bp_maxs = np.max(x_medians, axis=1)
    x_bp_ranges = x_bp_maxs - x_bp_mins
    x_bp_rng_med = np.median(x_bp_ranges)
    x_bp_rng_std = np.std(x_bp_ranges)

    y_bp_mins = np.min(y_medians, axis=1)
    y_bp_maxs = np.max(y_medians, axis=1)
    y_bp_ranges = y_bp_maxs - y_bp_mins
    y_bp_rng_med = np.median(y_bp_ranges)
    y_bp_rng_std = np.std(y_bp_ranges)

    # Get xx and yy mean bandpass stats
    x_mean_bps = np.median(x_medians, axis=1)
    x_bp_median = np.median(x_mean_bps)
    x_bp_std = np.std(x_mean_bps)

    y_mean_bps = np.median(y_medians, axis=1)
    y_bp_median = np.median(y_mean_bps)
    y_bp_std = np.std(y_mean_bps)

    return (x_bp_median, y_bp_median), (x_bp_std, y_bp_std), (x_bp_rng_med, y_bp_rng_med), (x_bp_rng_std, y_bp_rng_std)


def get_outlier_bandpasses(bandpass, summary_axis, x_means, y_means):
    """
    From the 36 samples along an axis, identify the biggest outliers of the samples. 
    A bandpass is assessed both on its median and  outlier can be due either its mean in
    """
            
    bp_median, bp_std, bp_rng_med, bp_rng_std = get_bandpass_ranges(bandpass, summary_axis, x_means, y_means)
    
    x_med = np.median(x_means, axis=1)
    y_med = np.median(y_means, axis=1)
    
    x_range = np.max(x_means, axis=1) - np.min(x_means, axis=1)
    y_range = np.max(y_means, axis=1) - np.min(y_means, axis=1)

    x_dev = np.abs(x_med-bp_median[0])/bp_std[0]
    y_dev = np.abs(y_med-bp_median[1])/bp_std[1]
    x_range_dev = np.abs((x_range)-bp_rng_med[0])/bp_rng_std[0]
    y_range_dev = np.abs((y_range)-bp_rng_med[1])/bp_rng_std[1]

    #print (x_dev, y_dev, x_range_dev, y_range_dev)
    order_desc = np.argsort((np.max((x_dev, y_dev, x_range_dev, y_range_dev), axis=0)))[::-1]
    return order_desc, (x_dev, y_dev, x_range_dev, y_range_dev)


def plot_bandpass_summary(bandpass, summary_axis, sbid, fig_folder, bandpass_type):
    sns.set()
    sns.set_context("paper")

    fig, axs = plt.subplots(1, 2, figsize=(14,8), sharey=True)

    tgt_name = 'beam' if summary_axis == 1 else 'antenna'
    x_medians, y_medians = get_median_bandpasses(bandpass, summary_axis)
    bp_median, bp_std, bp_rng_med, bp_rng_std = get_bandpass_ranges(bandpass, summary_axis, x_medians, y_medians)
    order, dev_array = get_outlier_bandpasses(bandpass, summary_axis, x_medians, y_medians)
    beam_dev = np.max(dev_array, axis=0)

    for beam_num in range(36):
        ax0 = axs[0]
        ax1 = axs[1]

        #print ("beam {:d} order {} dev {:.2f}".format(beam_num, np.where(order == beam_num)[0], beam_dev[beam_num]))
        if tgt_name == 'beam':
            label = "{} {:d}".format(tgt_name, beam_num)
        else:
            label = "ak{:02d}".format(beam_num+1)
        if np.where(order == beam_num)[0] < 10 and beam_dev[beam_num] > 1:
            ax0.plot(x_medians[beam_num], label=label, lw=2, zorder=2)
            ax1.plot(y_medians[beam_num], label=label, lw=2, zorder=2)
        else: 
            ax0.plot(x_medians[beam_num], color='grey', lw=1, zorder=1)
            ax1.plot(y_medians[beam_num], color='grey', lw=1, zorder=1)

    ax0.legend()
    ax0.set_title("Median XX bandpass for each " + tgt_name)
    ax1.set_title("Median YY bandpass for each " + tgt_name)
    fig.suptitle("Bandpass for {} SBID {}".format(bandpass_type, sbid))

    axs[0].set_ylabel("Amplitude (Jy)")
    for i in range(2):
        axs[i].set_xlabel("Channel")

    fname = fig_folder + '/bandpass_sb{}_by_{}.png'.format(sbid, tgt_name)
    fig.savefig(fname, bbox_inches='tight')
    #fig.savefig(fname[:-3]+"pdf", bbox_inches='tight')
    return fname

def plot_bandpass_by_antenna(bandpass, sbid, fig_folder, bandpass_type='Calibration'):
    return plot_bandpass_summary(bandpass, 0, sbid, fig_folder, bandpass_type)

def plot_bandpass_by_beam(bandpass, sbid, fig_folder, bandpass_type='Calibration'):
    return plot_bandpass_summary(bandpass, 1, sbid, fig_folder, bandpass_type)

