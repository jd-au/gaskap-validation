# Library of routines for working with ASKAPsoft pipeline diagnostic reports.
# These are mostly focussed around flagging of baselines per integration per beam in the measurement sets.
# Much of this code is adapted from the WALLABY Spectral Line validation scripts at https://github.com/askap-qc/validation

# Author James Dempsey
# Date 1 Jul 2020

import glob
import os
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def find_subdir(cube, image, name):
    potential_parent_dirs = []
    if cube:
        potential_parent_dirs.append(os.path.dirname(cube))
    if image:
        potential_parent_dirs.append(os.path.dirname(image))

    for parent in potential_parent_dirs:
        diag_dir = parent + '/' + name
        if os.path.isdir(diag_dir):
            return diag_dir
    
    return None


def find_diagnostics_dir(cube, image):
    """
    Identify the location of the diagnostics dir based on other specified file locations.

    Parameters
    ----------
    cube: path
        Path to the spectral line cube, if any
    image: path
        Path to the continuum image, if any

    Returns
    -------
    The path to the diagnostics directory, or None if it cannot be found.
    """
    return find_subdir(cube, image, 'diagnostics')


def find_flagging_summary_dir(diagnostics):
    """
    Identify the location of the directory containing the flagging sumamries.

    Parameters
    ----------
    diagnostics: path
        Path to the diagnostics directory

    Returns
    -------
    The path to the flagging summary directory, or None if it cannot be found.
    """
    if not (diagnostics and os.path.exists(diagnostics)):
        return None
    
    potential_flag_dir = [diagnostics+'/Flagging_Summaries', diagnostics]
    for flag_dir in potential_flag_dir:
        if os.path.exists(flag_dir):
            summaries = glob.glob(flag_dir+'/*_SL.ms.flagSummary')
            if len(summaries) > 0:
                return flag_dir
    
    return None

def find_metadata_file(diagnostics_dir, file_pattern, verbose=False):
    """
    Identify the location of the metadata dir based on other specified file locations.

    Parameters
    ----------
    cube: path
        Path to the spectral line cube, if any
    image: path
        Path to the continuum image, if any

    Returns
    -------
    The path to the metadata directory, or None if it cannot be found.
    """
    if not diagnostics_dir:
        return None
    metadata_dir = os.path.dirname(diagnostics_dir) + '/metadata'
    if not os.path.isdir(metadata_dir):
        if verbose:
            print("Unable to find metadata dir at", metadata_dir)
        return None
    metafiles = sorted(glob.glob(metadata_dir+'/'+file_pattern))
    if len(metafiles) == 0:
        if verbose:
            print("Unable to find metdata file wiht pattern {} in {}".format(file_pattern, metadata_dir))
        return None
    metadata_file = metafiles[0]
    return metadata_file

def find_cal_sbid(diagnostics_dir, verbose=False):
    metadata_dir = os.path.dirname(diagnostics_dir) + '/metadata'
    if not os.path.isdir(metadata_dir):
        if verbose:
            print("Unable to find metadata dir at", metadata_dir)
        return 'N/A'
    
    # The name of the metadata/mslist-cal... file seems to have changed and now is a timestamp and not an sbid
    return 'N/A'


def get_freq_details(diagnostics_dir, verbose=False):
    """
    Retrieve frequency details from the metadata ms list. This assumes the same single specral window was 
    observed for all beams.

    Parameters
    ----------
    diagnostics_dir: path
        Path to the diagnostics dir, the metadata dir is assumed to b a sibling of this directory.
    verbose: boolean
        True if extra output is needed.

    Returns
    -------
    The channel width, centre frequency and number of channels
    """
    metafile_science = find_metadata_file(diagnostics_dir, 'mslist-scienceData*txt', verbose=False)
    if not metafile_science:
        return None, None, None

    with open(metafile_science, 'r') as mslist_file:
        lines = mslist_file.readlines()

    in_spw_block = False
    for line in lines:
        if in_spw_block:
            parts = line.split()
            chan_width = float(parts[10])*1000. # convert kHz to Hz
            cfreq = parts[12] #MHz
            nchan = parts[7]
            break
        else:
            in_spw_block = line.find('Frame') >= 0

    return chan_width, cfreq, nchan


def get_metadata(diagnostics_dir, verbose=False):
    """
    Getting basic information on observed field (one field only). 
    """
    metafile = find_metadata_file(diagnostics_dir, 'mslist-2*txt', verbose=False)

    with open(metafile, 'r') as mslist_file:
        lines = mslist_file.readlines()

    nBlocks = 6  # these are the number of correlator cards (PILOT survey value)
    
    obs_date = 'Observed from'
    code = 'Code'
    duration = 'Total elapsed time'
    antenna = 'antennas'
    frame = 'Frame'
    
    for i in range(len(lines)):
        line = lines[i]
        if line.find(antenna) >=0:
            toks = line.split()
            n_ant = toks[5][-2:]
        if line.find(obs_date) >=0:
            toks = line.split()
            start_obs_date = toks[6]
            end_obs_date = toks[8]
        if line.find(duration) >=0:
            toks = line.split()
            tobs = float(toks[10]) # in second
        if line.find(code) >= 0:
            next_line = lines[i+1]
            toks = next_line.split()
            field = toks[5]
            ra = toks[6][:-5]
            dec = toks[7][:-4]
        if line.find(frame) >= 0:
            next_line = lines[i+1]
            toks = next_line.split()
            total_obs_bw = float(toks[10])*nBlocks/1000.0 # kHz to MHz 
            
    return n_ant, start_obs_date, end_obs_date, tobs, field, ra, dec, total_obs_bw


def _get_flagging_key_values(flagging_file):
    """
    Getting Flagging Key Values. 
    """

    with open(flagging_file, 'r') as f:
        lines = f.readlines()[:6]
    
    N_Rec = 'nRec'  # Total number of spectra feeds into the synthesis image. This is not always constant so grab the value beam-by-beam.
    N_Chan = 'nChan'  # Total number of channel

    # Search for keywords in the file
    
    for i in range(len(lines)):
        line = lines[i]
        if line.find(N_Rec) >=0:
            tokens = line.split()
            n_Rec = float(tokens[2])
        if line.find(N_Chan) >=0:
            tokens = line.split()
            n_Chan = float(tokens[2])

    exp_count = n_Rec*35 #counting antenna number from zero based on the recorded data
    
    return n_Rec, n_Chan, exp_count


def _get_flagging(flagging_file, flag_ant_file, num_integ, n_chan):
    """
    Getting flagging statistics and finding out beam-by-beam antenna based (completely) flagging. 
    """

    # Inner: 1-6
    # Mid: 7-30
    # Outer: 31 - 36

    # Finding out which antenna has been flagged completely.
    all_ant1, all_ant2, all_flag = [], [], []
    num_inner, flag_inner, num_outer, flag_outer = 0, 0, 0, 0
    integ_ant1, integ_ant2, integ_flag = [], [], []
    integ_num_inner, integ_flag_inner, integ_num_outer, integ_flag_outer = 0, 0, 0, 0
    num_integ_flagged = 0
    with open(flagging_file, 'r') as f:
        for line in f:
            if "#" not in line:  # grep -v "#"
                if "Flagged" not in line:   # grep -v "Flagged"
                    tokens = line.split()
                    ant1 = int(tokens[3])
                    ant2 = int(tokens[4])
                    flag = float(tokens[6])
                    if (ant1 < ant2) and (flag == 100): # extract non-correlated antenna pairs with 100 percent flagging
                        integ_ant1.append(ant1)
                        integ_ant2.append(ant2)
                        integ_flag.append(flag)
                    if (ant1 < 6) and (ant2 < 6) and (ant1 != ant2):
                        integ_num_inner += 1
                        integ_flag_inner += flag
                    if (ant1 >= 30) and (ant2 >= 30) and (ant1 != ant2):
                        integ_num_outer += 1
                        integ_flag_outer += flag
            elif "# Integration Number:" in line:
                tokens = line.split()
                integ_num = int(tokens[3])
                flag = float(tokens[5])
                if flag == 100:
                    num_integ_flagged += 1
                    # totally flagged so don't count individual flagging
                else:
                    all_ant1.extend(integ_ant1)
                    all_ant2.extend(integ_ant2)
                    all_flag.extend(integ_flag)
                    num_inner += integ_num_inner
                    flag_inner += integ_flag_inner
                    num_outer += integ_num_outer
                    flag_outer += integ_flag_outer
                # Reset the integration details ready for the enxt integration (if any)
                integ_ant1, integ_ant2, integ_flag = [], [], []
                integ_num_inner, integ_flag_inner, integ_num_outer, integ_flag_outer = 0, 0, 0, 0


    last_line = line
    exp_count = (num_integ - num_integ_flagged) * 35 # Number of unflagged integrations times number of non-autocorrelation baselines

    # Analyse the flagging data
    ant1, ant2, flag = np.asarray(all_ant1), np.asarray(all_ant2), np.asarray(all_flag)

    ant_names = []
    for x in range(0,36):
        count1 = np.count_nonzero(ant1 == x)
        count2 = np.count_nonzero(ant2 == x)
        total_count = count1 + count2
        if total_count == exp_count:
            ant_num = x+1
            ant_name = 'ak{:02d}'.format(ant_num)
            ant_names.append(ant_name)

    total_flagged_ant = len(ant_names)

    with open(flag_ant_file,'a') as ffile:
        ffile.write(flagging_file[-24:-18])
        if total_flagged_ant > 0:
            ffile.write('\n')
            for item in ant_names:
                ffile.write(item)
                ffile.write('\n')
        else:
            ffile.write('\n none \n')
        ffile.write('\n')
    
    flag_pct_inner = 0 if num_inner == 0 else flag_inner / num_inner
    flag_pct_outer = 0 if num_outer == 0 else flag_outer / num_outer
    flag_pct_integ = 0 if num_integ == 0 else 100* num_integ_flagged / num_integ

    # Getting data flagged percentage from the last line of the summary
    str_line = last_line
    if isinstance(str_line, bytes):
        str_line = last_line.decode('utf-8')
    tokens = str_line.split()
    total_flagged_pct = float(tokens[-2]) #data+autocorrelation
    total_uv = float(tokens[7])
    autocorr_flagged_pct = (36 * num_integ * n_chan / total_uv)*100.0
    data_flagged_pct = round(total_flagged_pct - autocorr_flagged_pct, 3)

    return data_flagged_pct, total_flagged_ant, flag_ant_file, ant_names, flag_pct_inner, flag_pct_outer, flag_pct_integ


def get_flagging_stats(diagnostics_dir, fig_folder, verbose=False):
    """
    Retrieve a summary of the flagging for this observation.

    Parameters
    ----------
    diagnostics_dir: path
        Path to the diagnostics dir, the metadata dir is assumed to b a sibling of this directory.
    verbose: boolean
        True if extra output is needed.

    Returns
    -------
    The list of antennas that are flagged in all beams, the percent of sort baseline integrations flagged, 
    the percent of long baseline integrations flagged.
    """
    flagging_dir = find_flagging_summary_dir(diagnostics_dir)
    if flagging_dir is None:
        if verbose:
            print("Unable to find Flagging Summary files.")
            return None

    flagging_files = sorted(glob.glob(flagging_dir+'/*_SL.ms.flagSummary'))

    flag_stat_beams, n_flag_ant_beams, flag_inner_beams, flag_outer_beams, flag_integ = [], [], [], [], []

    flag_ant_file = fig_folder +'/flagged_antenna.txt'

    if os.path.exists(flag_ant_file):
        os.remove(flag_ant_file)
    
    if verbose:
        print ("Num flagging files:", len(flagging_files))

    ant_flagged_in_all = None
    for ffile in flagging_files:
        n_rec, n_chan, exp_count = _get_flagging_key_values(ffile)
        flag_stat, n_flag_ant, flag_ant_file, flagged_ant, pct_flag_inner, pct_flag_outer, pct_integ_flagged = _get_flagging(ffile, flag_ant_file, n_rec, n_chan)
        flag_stat_beams.append(flag_stat)
        n_flag_ant_beams.append(n_flag_ant)
        flag_integ.append(pct_integ_flagged)
        if pct_integ_flagged < 100:
            flag_inner_beams.append(pct_flag_inner)
            flag_outer_beams.append(pct_flag_outer)
        if ant_flagged_in_all == None:
            ant_flagged_in_all = flagged_ant
        else:
            for ant in list(ant_flagged_in_all):
                if ant not in flagged_ant:
                    ant_flagged_in_all.remove(ant)

    pct_inner_base_flagged = round(np.mean(flag_inner_beams),1)
    pct_outer_base_flagged = round(np.mean(flag_outer_beams),1)
    pct_integ_flagged = round(np.mean(flag_integ),1)
    print ("Flagged integrations", flag_integ)

    return flag_stat_beams, n_flag_ant_beams, ant_flagged_in_all, pct_inner_base_flagged, pct_outer_base_flagged, pct_integ_flagged


def calc_beam_exp_rms(flag_stat_beams, theoretical_rms_mjy):
    """
    Calculating the theoretical RMS of individual beam by taking into account the flagged percentage. 
    Assuming same weighting for the non-flagged data. 
    """

    stats = np.asarray(flag_stat_beams)
    beam_exp_rms = 1/np.sqrt(1.0 - stats/100.0)*theoretical_rms_mjy
    
    return beam_exp_rms

def beam_positions(closepack=False):
    """
    Defining 36 beam positions for plotting.
    """
    
    x_pos, y_pos = [], []

    x=0
    for j in range(0,6,1):
        x += 0.1
        y=0
        for k in range(0,6,2):
            y += 0.2
            x_pos.append(x+(0.05 if closepack else 0))
            y_pos.append(y)
            y += 0.2
            x_pos.append(x)
            y_pos.append(y)

    return x_pos, y_pos


def get_beam_numbers_closepack():
    n = []
    for i in range(6,0,-1):
        for j in range(6):
            n.append(j*6+i-1)
    return n


def make_thumbnail(image, fig_folder, rel_to_folder, size_x=70, size_y=70, name=None):
    """
    Making thumbnail image.
    """
    thumb_img = name if name else 'thumb_'+ str(size_x) + '_'+ os.path.basename(image)
    size = size_x, size_y
    im = Image.open(image)
    im.thumbnail(size)
    file_name = fig_folder+ '/' + thumb_img
    im.save(file_name)
    rel_file_name = os.path.relpath(file_name, rel_to_folder)
    return file_name, rel_file_name


def plot_flag_stat(flag_stat, beam_nums, fig_dir, closepack=False):
    """
    Plotting and visualising flagging statistics of 36 beams. 
    """
    
    title = 'Flagged Fraction'
    plot_name = 'FlagStat.png'
    saved_fig = fig_dir+'/'+plot_name

    params = {'axes.labelsize': 10,
              'axes.titlesize': 10,
              'font.size':10}

    mpl.rcParams.update(params)

    num_interleaves = len(flag_stat) // 36
    fig = plt.figure(figsize=(18, 12))
    axes = []
    if len(flag_stat) > 36:
        for i in range(1,num_interleaves+1):
            ax = fig.add_subplot(2,2,i)
            axes.append(ax)
    else:
        ax = fig.add_subplot()
        axes.append(ax)

    beam_xpos, beam_ypos = beam_positions(closepack=closepack)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    
    for i in range(num_interleaves):
        print ("Interleave", i)
        ax = axes[i]
        for j in range(36):
            bnum = beam_nums[j]
            b_idx = i*36+bnum
            ax.scatter([beam_xpos[j]], [beam_ypos[j]], s=1800, c=[flag_stat[b_idx]], cmap='tab20b', edgecolors='black',vmin=0, vmax=100)
            ax.text(beam_xpos[j], beam_ypos[j], bnum)

        ax.set_xlim(0,0.7)
        ax.tick_params(axis='both',which='both', bottom=False,top=False,right=False,left=False,labelbottom=False, labelleft=False)
        ax.set_title("Flagged Fraction for Interleave " + chr(ord('A')+i))

        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap='tab20b', norm=norm), ax=ax)
        cb.set_label('Percentage Flagged')
        cb.ax.tick_params(labelsize=10)

    plt.savefig(saved_fig, bbox_inches='tight')
    plt.close()
    print (saved_fig, plot_name)

    return saved_fig


def plot_flag_ant(n_flag_ant_beams, beam_nums, fig_dir, closepack=False):
    """
    Plotting and visualising number of flagged (completely) antennas beam-by-beam. 
    """

    title = 'No. of 100% flagged antenna'
    plot_name = 'FlagAnt.png'
    saved_fig = fig_dir+'/'+plot_name
    
    from_list = mpl.colors.LinearSegmentedColormap.from_list

    params = {'axes.labelsize': 10,
              'axes.titlesize': 10,
              'font.size':10}

    mpl.rcParams.update(params)

    num_interleaves = len(n_flag_ant_beams) // 36
    fig = plt.figure(figsize=(18, 12))
    axes = []
    if len(n_flag_ant_beams) > 36:
        for i in range(1,num_interleaves+1):
            ax = fig.add_subplot(2,2,i)
            axes.append(ax)
    else:
        ax = fig.add_subplot()
        axes.append(ax)

    beam_xpos, beam_ypos = beam_positions(closepack=closepack)
    maxflag = np.max(n_flag_ant_beams)+1 # Account for starting at 0
    cmap = plt.get_cmap('summer', maxflag)
    norm = mpl.colors.Normalize(vmin=0, vmax=maxflag)

    for i in range(num_interleaves):
        print ("Interleave", i)
        ax = axes[i]
        for j in range(36):
            bnum = beam_nums[j]
            b_idx = i*36+bnum
            ax.scatter([beam_xpos[j]], [beam_ypos[j]], s=1500, c=[n_flag_ant_beams[b_idx]], cmap=cmap, edgecolors='black',vmin=0, vmax=maxflag)
            ax.text(beam_xpos[j], beam_ypos[j], bnum)

        ax.set_xlim(0,0.7)
        ax.tick_params(axis='both',which='both', bottom=False,top=False,right=False,left=False,labelbottom=False, labelleft=False)
        ax.set_title("No. of 100% flagged antenna for Interleave " + chr(ord('A')+i))

        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
        labels =np.arange(0,maxflag)
        loc = labels + .5
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=10)

    plt.savefig(saved_fig, bbox_inches='tight')
    plt.close()
    print (saved_fig, plot_name)

    return saved_fig


def plot_beam_exp_rms(beam_exp_rms, beam_nums, fig_folder, closepack=False):
    """
    Plotting and visualising expected RMS of all beams.
    """

    title = 'Expected RMS'
    plot_name = 'Exp_RMS.png'
    saved_fig = fig_folder+'/'+plot_name

    params = {'axes.labelsize': 10,
              'axes.titlesize': 10,
              'font.size':10}

    mpl.rcParams.update(params)

    num_interleaves = len(beam_exp_rms) // 36
    fig = plt.figure(figsize=(18, 12))
    axes = []
    if len(beam_exp_rms) > 36:
        for i in range(1,num_interleaves+1):
            ax = fig.add_subplot(2,2,i)
            axes.append(ax)
    else:
        ax = fig.add_subplot()
        axes.append(ax)

    beam_xpos, beam_ypos = beam_positions(closepack=closepack)
    vmin = round(np.min(beam_exp_rms), 3)
    vmax = round(np.minimum(vmin*2,np.max(beam_exp_rms)), 3)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(num_interleaves):
        print ("Interleave", i)
        ax = axes[i]
        for j in range(36):
            bnum = beam_nums[j]
            b_idx = i*36+bnum
            ax.scatter([beam_xpos[j]], [beam_ypos[j]], s=1500, c=[beam_exp_rms[b_idx]], cmap='GnBu', edgecolors='black', vmin=vmin, vmax=vmax)
            ax.text(beam_xpos[j], beam_ypos[j], bnum)

        ax.set_xlim(0,0.7)
        ax.tick_params(axis='both',which='both', bottom=False,top=False,right=False,left=False,labelbottom=False, labelleft=False)
        ax.set_title("Expected RMS for Interleave " + chr(ord('A')+i))

        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap='GnBu', norm=norm), ax=ax)
        cb.set_label('mJy / beam')
        cb.ax.tick_params(labelsize=10)

    plt.savefig(saved_fig, bbox_inches='tight')
    plt.close()
    print (saved_fig, plot_name)

    return saved_fig
