#!/usr/bin/env python -u

# Validation script for GASKAP HI data
#

# Author James Dempsey
# Date 23 Nov 2019


from __future__ import print_function, division

import argparse
import csv
import datetime
import glob
import math
import os
import re
from string import Template
import shutil
import time
import warnings

import matplotlib
matplotlib.use('agg')

import aplpy
from astropy.constants import k_B
from astropy.coordinates import solar_system_ephemeris, EarthLocation, Angle, SkyCoord, AltAz, FK5, get_body
from astropy.io import ascii, fits
from astropy.io.votable import parse, from_table, writeto
from astropy.io.votable.tree import Info
from astropy.table import Table, Column
from astropy.time import Time, TimezoneInfo
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from radio_beam import Beam
from spectral_cube import SpectralCube
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns

from validation import Bandpass, Diagnostics, SelfCal, Spectra
from validation_reporter import ValidationReport, ReportSection, ReportItem, ValidationMetric, output_html_report, output_metrics_xml


vel_steps = [-324, -280, -234, -189, -143, -100, -60, -15, 30, 73, 119, 165, 200, 236, 273, 311, 357, 399]
#emission_vel_range=[] # (165,200)*u.km/u.s
emission_vel_range=(119,165)*u.km/u.s 
non_emission_val_range=(-100,-60)*u.km/u.s 
figures_folder = 'figures'

METRIC_BAD = 3
METRIC_UNCERTAIN = 2
METRIC_GOOD = 1

def parseargs():
    """
    Parse the command line arguments
    :return: An args map with the parsed arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Produce a validation report for GASKAP HI observations. Either a cube or an image (or both) must be supplied to be validated.")

    parser.add_argument("-c", "--cube", required=False, help="The HI spectral line cube to be checked.")
    parser.add_argument("-i", "--image", required=False, help="The continuum image to be checked.")
    parser.add_argument("-s", "--source_cat", required=False, help="The selavy source catalogue used for source identification.")
    parser.add_argument("-b", "--beam_list", required=False, help="The csv file describing the positions of each beam (in radians).")
    parser.add_argument("-d", "--duration", required=False, help="The duration of the observation in hours.", type=float, default=12.0)

    parser.add_argument("-o", "--output", help="The folder in which to save the validation report and associated figures.", default='report')

    parser.add_argument("-e", "--emvel", required=False, help="The low velocity bound of the velocity region where emission is expected.")
    parser.add_argument("-n", "--nonemvel", required=False, 
                        help="The low velocity bound of the velocity region where emission is not expected.", default='-100')

    parser.add_argument("-N", "--noise", required=False, help="Use this fits image of the local rms. Default is to run BANE", default=None)
    parser.add_argument("-r", "--redo", help="Rerun all steps, even if intermediate files are present.", default=False,
                        action='store_true')
    parser.add_argument("--num_spectra", required=False, help="Number of sample spectra to create", type=int, default=15)

    args = parser.parse_args()
    return args


def get_str(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def plot_histogram(file_prefix, xlabel, title):
    data = fits.getdata(file_prefix+'.fits')
    flat = data.flatten()
    flat = flat[~np.isnan(flat)]
    v =plt.hist(flat, bins=200, bottom=1, log=True, histtype='step')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(file_prefix+'_hist.png', bbox_inches='tight')
    plt.savefig(file_prefix+'_hist_sml.png', dpi=16, bbox_inches='tight')
    plt.close()

    
def plot_map(file_prefix, title, cmap='magma', stretch='linear', pmax=99.75, colorbar_label=None):
    fig = plt.figure(figsize=(5, 4.5))

    gc = aplpy.FITSFigure(file_prefix+'.fits', figure=fig)
    gc.show_colorscale(cmap=cmap, stretch=stretch, pmax=pmax)
    gc.add_colorbar()
    if colorbar_label:
        gc.colorbar.set_axis_label_text(colorbar_label)
    gc.add_grid()
    gc.set_title(title)
    gc.savefig(filename=file_prefix+'.png', dpi=200)
    gc.savefig(filename=file_prefix+'.pdf', dpi=100)
    gc.savefig(filename=file_prefix+'_sml.png', dpi=16 )
    gc.close()


def plot_difference_map(hdu, file_prefix, title, vmin=None, vmax=None):
    # Initiate a figure and axis object with WCS projection information
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection=wcs)

    no_nan_data = np.nan_to_num(hdu.data)
    if vmin is None and vmax is None:
        vmin=np.percentile(no_nan_data, 0.25)
        vmax=np.percentile(no_nan_data, 99.75)

    im = ax.imshow(hdu.data, cmap='RdBu_r',vmin=vmin,vmax=vmax, origin='lower')
    #ax.invert_yaxis() 

    ax.set_xlabel("Right Ascension (degrees)", fontsize=16)
    ax.set_ylabel("Declination (degrees)", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(color = 'gray', ls = 'dotted', lw = 2)
    cbar = plt.colorbar(im, pad=.07)

    plt.savefig(file_prefix+'.png', bbox_inches='tight')
    plt.savefig(file_prefix+'_sml.png', dpi=10, bbox_inches='tight')

    plt.close()


def output_plot(mp, title, imagename):
    mp.write('\n<h2>{}</h2>\n<br/>'.format(title))
    mp.write('\n<a href="{}"><img width="800px" src="{}"></a>'.format(imagename, imagename))
    mp.write('\n<br/>\n')


def output_map_page(filename, file_prefix, title):
    with open(filename, 'w') as mp:
        mp.write('<html>\n<head><title>{}</title>\n</head>'.format(title))
        mp.write('\n<body>\n<h1>{}</h1>'.format(title))

        output_plot(mp, 'Large Scale Emission Map', file_prefix + '_bkg.png')
        output_plot(mp, 'Noise Map', file_prefix + '_rms.png')
        output_plot(mp, 'Moment 0 Map', file_prefix + '.png')
        mp.write('\n</body>\n</html>\n')


def convert_slab_to_jy(slab, header):
    my_beam = Beam.from_fits_header(header)
    restfreq = 	1.420405752E+09*u.Hz
    if 'RESTFREQ' in header.keys():
        restfreq = header['RESTFREQ']*u.Hz
    elif 'RESTFRQ' in header.keys():
        restfreq = header['RESTFRQ']*u.Hz

    if slab.unmasked_data[0,0,0].unit != u.Jy and slab.unmasked_data[0,0,0].unit != u.Jy/u.beam:
        print ("Converting slab from {} to Jy".format(slab.unmasked_data[0,0,0].unit) )
        print (slab)
        slab.allow_huge_operations=True
        slab = slab.to(u.Jy, equivalencies=u.brightness_temperature(my_beam, restfreq))
        print (slab)
    return slab


def convert_data_to_jy(data, header, verbose=False):
    my_beam = Beam.from_fits_header(header)
    restfreq = 	1.420405752E+09*u.Hz
    if 'RESTFREQ' in header.keys():
        restfreq = header['RESTFREQ']*u.Hz
    elif 'RESTFRQ' in header.keys():
        restfreq = header['RESTFRQ']*u.Hz

    if data[0].unit != u.Jy:
        if verbose:
            print ("Converting data from {} to Jy".format(data[0].unit) )
        data = data.to(u.Jy, equivalencies=u.brightness_temperature(my_beam, restfreq))
    return data

def get_vel_limit(vel_cube):
    velocities = np.sort(vel_cube.spectral_axis)
    return velocities[0], velocities[-1]

def extract_slab(filename, vel_start, vel_end):
    cube = SpectralCube.read(filename)
    vel_cube = cube.with_spectral_unit(u.m/u.s, velocity_convention='radio')
    cube_vel_min, cube_vel_max = get_vel_limit(vel_cube)
    if vel_start > cube_vel_max or vel_end < cube_vel_min:
        return None

    slab = vel_cube.spectral_slab(vel_start, vel_end)
    header = fits.getheader(filename)
    slab = convert_slab_to_jy(slab, header)
    return slab


def extract_channel_slab(filename, chan_start, chan_end):
    cube = SpectralCube.read(filename)
    vel_cube = cube.with_spectral_unit(u.m/u.s, velocity_convention='radio')
    slab = vel_cube[chan_start:chan_end,:, :].with_spectral_unit(u.km/u.s)

    header = fits.getheader(filename)
    return slab

def build_fname(example_name, suffix):
    basename = os.path.basename(example_name)
    prefix = os.path.splitext(basename)[0]
    fname = prefix + suffix
    return fname

def get_figures_folder(dest_folder):
    return dest_folder + '/' + figures_folder + '/'

def get_bane_background(infile, outfile_prefix, plot_title_suffix, ncores=8, redo=False, plot=True):
    background_prefix = outfile_prefix+'_bkg'
    background_file = background_prefix + '.fits'
    if redo or not os.path.exists(background_file):
        cmd = "BANE --cores={0} --out={1} {2}".format(ncores, outfile_prefix, infile)
        print (cmd)
        os.system(cmd)
    
    if plot:
        plot_map(background_prefix, "Large scale emission in " + plot_title_suffix)
        plot_histogram(background_prefix, 'Emission (Jy beam^{-1} km s^{-1})', "Emission for " + plot_title_suffix)
        plot_map(outfile_prefix+'_rms', "Noise in "+ plot_title_suffix)
    
    return background_file


def assess_metric(metric, threshold1, threshold2, low_good=False):
    if metric < threshold1:
        return METRIC_GOOD if low_good else METRIC_BAD
    elif metric < threshold2:
        return METRIC_UNCERTAIN
    else:
        return METRIC_BAD if low_good else METRIC_GOOD


def get_spectral_units(ctype, cunit, hdr):
    spectral_conversion = 1
    if not cunit in hdr:
        if ctype.startswith('VEL') or ctype.startswith('VRAD'):
            spectral_unit = 'm/s'
        else:
            spectral_unit = 'Hz'
    else:
        spectral_unit = hdr[cunit]
    if spectral_unit == 'Hz':
        spectral_conversion = 1e6
        spectral_unit = 'MHz'
    elif spectral_unit == 'kHz':
        spectral_conversion = 1e3
        spectral_unit = 'MHz'
    elif spectral_unit == 'm/s':
        spectral_conversion = 1e3
        spectral_unit = 'km/s'
    return spectral_unit, spectral_conversion


def calc_velocity_res(hdr):
    spec_sys = hdr['SPECSYS']
    axis = '3' if hdr['CTYPE3'] != 'STOKES' else '4'
    spec_type = hdr['CTYPE'+axis]

    spectral_unit, spectral_conversion = get_spectral_units(spec_type, 'CUNIT'+axis, hdr)
    if 'CUNIT'+axis in hdr.keys():
        spec_unit = hdr['CUNIT'+axis]
    #elif spec_type == 'VRAD' or spec_type == 'VEL':
    #    spec_unit = 'm/s'
    else:
        spec_unit = None
    spec_delt = hdr['CDELT'+axis]
    print ('CDELT={}, CUNIT={}, spec_unit={}, conversion={}'.format(spec_delt, spec_unit, spectral_unit, spectral_conversion))

    spec_res_km_s = np.abs(spec_delt) / spectral_conversion
    if spectral_unit == 'MHz':
        spec_res_km_s = spec_res_km_s/5e-4*0.1 # 0.5 kHz = 0.1 km/s
    #elif spec_unit == 'Hz':
    #        spec_res_km_s = spec_res_km_s/500*0.1 # 0.5 kHz = 0.1 km/s
    #elif spec_unit == 'kHz':
    #    spec_res_km_s = spec_res_km_s/0.5*0.1 # 0.5 kHz = 0.1 km/s
    return spec_res_km_s


def get_observing_location():
    # Coordinates for Inyarrimanha Ilgari Bundara, the CSIRO Murchison Radio-astronomy Observatory
    latitude = Angle("-26:41:46.0", unit=u.deg)
    longitude = Angle("116:38:13.0", unit=u.deg)
    return EarthLocation(lat=latitude, lon=longitude)


def plot_solar_elevation(fig_folder, obs_time_utc, duration, field_pos, sbid):
    obs_end_utc = obs_time_utc + duration
    tmjd = np.arange(obs_time_utc.mjd, obs_end_utc.mjd, 1./24.0/60.0/6.0)
    times = Time(tmjd, format='mjd', scale='utc')

    observing_location = get_observing_location()

    solar_system_ephemeris.set('builtin')
    sun_sc = get_body("sun", times, location=observing_location)
    #sun_sc_median = get_body("sun", Time(np.median(tmjd), format='mjd', scale='utc'), location=observing_location)

    sun_altaz = sun_sc.transform_to(AltAz(obstime=times, location=observing_location))
    sun_altdeg = sun_altaz.alt.deg
    field_sun_sep_deg = sun_sc.separation(field_pos).degree

    sns.set()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot, = ax1.plot((tmjd-tmjd[0])*24.0, sun_altdeg, marker='None', color="black")
    #print("Solar position = %s" %(sun_sc_median.to_string(style='hmsdms')))
    ax1.set_title("Solar elevation for SBID %s" %(sbid))
    ax1.set_xlabel("Time since start of observation (h)")
    ax1.set_ylabel("Elevation (deg)")
    ax1.axhline(0, color='g')
    ax1.axhline(-6, color='blue')
    ax1.axhline(-12, color='darkblue')
    ax1.axhline(-18, color='indigo')
    fig_path = fig_folder + 'solar_el.png'
    fig.savefig(fig_path)
    plt.close()

    # Calculate the daylight percent of the observation
    sun_up_filter = sun_altdeg >= 0
    steps_sun_up = np.sum(sun_up_filter)
    pct_daylight = steps_sun_up/len(sun_altdeg)*100


    return fig_path, pct_daylight, field_sun_sep_deg


def plot_target_elevation(fig_folder, obs_time_utc, duration, field_pos, sbid):
    obs_end_utc = obs_time_utc + duration
    tmjd = np.arange(obs_time_utc.mjd, obs_end_utc.mjd, 1./24.0/60.0/6.0)
    times = Time(tmjd, format='mjd', scale='utc')

    observing_location = get_observing_location()

    field_altaz = field_pos.transform_to(AltAz(obstime=times, location=observing_location))
    field_altdeg = field_altaz.alt.deg

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot, = ax1.plot((tmjd-tmjd[0])*24.0, field_altdeg, marker='None', color="black")
    ax1.set_title("Target elevation for SBID %s" %(sbid))
    ax1.set_xlabel("Time since start of observation (h)")
    ax1.set_ylabel("Elevation (deg)")
    ax1.axhline(15, color='g')
    fig_path = fig_folder + 'target_el.png'
    fig.savefig(fig_path)
    plt.close()

    return fig_path


def report_observation(image, dest_folder, reporter, input_duration, sched_info, obs_metadata):
    print('\nReporting observation based on ' + image)

    hdr = fits.getheader(image)
    w = WCS(hdr).celestial

    sbid = hdr['SBID'] if 'SBID' in hdr else sched_info.sbid
    project = hdr['PROJECT'] if 'PROJECT' in hdr else ''
    proj_link = None
    if project.startswith('AS'):
        proj_link = "https://confluence.csiro.au/display/askapsst/{0}+Data".format(project)

    obs_date = hdr['DATE-OBS']  if 'DATE-OBS' in hdr else 'Unknown'
    duration = float(hdr['DURATION'])/3600 if 'DURATION' in hdr else input_duration

    naxis1 = int(hdr['NAXIS1'])
    naxis2 = int(hdr['NAXIS2'])
    pixcrd = np.array([[naxis1/2, naxis2/2]])
    centre = w.all_pix2world(pixcrd,1)
    centre_coord = SkyCoord(ra=centre[0][0], dec=centre[0][1], unit="deg,deg")
    centre = centre_coord.to_string(style='hmsdms',sep=':', precision=1)

    # spectral axis
    spectral_unit = 'None'
    spectral_range = ''
    spec_title = 'Spectral Range'
    for i in range(3,int(hdr['NAXIS'])+1):
        ctype = hdr['CTYPE'+str(i)]
        if (ctype.startswith('VEL') or ctype.startswith('VRAD') or ctype.startswith('FREQ')):
            key = 'CUNIT'+str(i)
            spectral_unit, spectral_conversion = get_spectral_units(ctype, key, hdr)
            
            step = float(hdr['CDELT'+str(i)])
            #print ('step {} rval {} rpix {} naxis {}'.format(step, hdr['CRVAL'+str(i)], hdr['CRPIX'+str(i)], hdr['NAXIS'+str(i)]))
            spec_start = (float(hdr['CRVAL'+str(i)]) - (step*(float(hdr['CRPIX'+str(i)])-1)))/spectral_conversion
            if int(hdr['NAXIS'+str(i)]) > 1:
                spec_end = spec_start + (step * (int(hdr['NAXIS'+str(i)]-1)))/spectral_conversion
                if step > 0:
                    spectral_range = '{:0.3f} - {:0.3f}'.format(spec_start, spec_end)
                else:
                    spectral_range = '{:0.3f} - {:0.3f}'.format(spec_end, spec_start)
                spec_title = 'Spectral Range'
            else:
                centre_freq = (float(hdr['CRVAL'+str(i)]) - (step*(float(hdr['CRPIX'+str(i)])-1)))/spectral_conversion 
                spectral_range = '{:0.3f}'.format(centre_freq)
                spec_title = 'Centre Freq'

    # Field info
    if obs_metadata:
        field_names = ''
        field_centres = ''
        for i,field in enumerate(obs_metadata.fields):
            if i > 0:
                field_names += '<br/>'
                field_centres += '<br/>'
            field_names += field.name
            field_centres += field.ra + ' ' + field.dec
    else:
        field_names = sched_info.field_name
        field_centres = centre
    
    footprint = sched_info.footprint
    if footprint and sched_info.pitch:
        footprint = "{}_{}".format(footprint, sched_info.pitch)        

    section = ReportSection('Observation')
    section.add_item('SBID', value=sbid)
    section.add_item('Project', value=project, link=proj_link)
    section.add_item('Date (UTC)', value=obs_date)
    section.add_item('Duration<br/>(hours)', value='{:.2f}'.format(duration))
    section.add_item('Field(s)', value=field_names)
    section.add_item('Field Centre(s)', value=field_centres)
    section.add_item('Correlator<br/>Mode', value=sched_info.corr_mode)
    section.add_item('Footprint', value=footprint)
    section.add_item('{}<br/>({})'.format(spec_title, spectral_unit), value=spectral_range)

    if obs_date != 'Unknown':
        murchison_zone = TimezoneInfo(utc_offset=8*u.hour)
        obs_time_utc = Time(obs_date, format='isot', scale='utc')
        local_start = obs_time_utc.to_datetime(timezone=murchison_zone)
        local_end = (obs_time_utc + duration*u.hour).to_datetime(timezone=murchison_zone)
        local_time = '{:%Y-%m-%d %H:%M:%S%z}<br/>{:%Y-%m-%d %H:%M:%S%z}'.format(local_start, local_end)

        fig_folder= get_figures_folder(dest_folder)
        field_pos = SkyCoord(obs_metadata.fields[0].ra, obs_metadata.fields[0].dec, frame=FK5, unit=(u.hourangle, u.deg)) if obs_metadata else centre_coord
        solr_el_path, pct_daylight, field_sun_sep_deg = plot_solar_elevation(fig_folder, obs_time_utc, duration*u.hour, field_pos, sbid)

        tgt_el_path = plot_target_elevation(fig_folder, obs_time_utc, duration*u.hour, field_pos, sbid)

        section.start_new_row()
        section.add_item('Local Time', value=local_time)
        #section.add_item('Solar Elevation', link=solr_el_path_rel, image=solr_el_path_rel) # Second should be a thumbnail
        add_opt_image_section('Solar Elevation', solr_el_path, fig_folder, dest_folder, section)
        section.add_item('Daytime Proportion (%)', value="{:.1f}".format(pct_daylight))
        section.add_item('Solar Separation (deg)', value="{:.3f} to {:.3f}".format(field_sun_sep_deg[0], field_sun_sep_deg[1]))
        add_opt_image_section('Target Elevation', tgt_el_path, fig_folder, dest_folder, section)


    reporter.add_section(section)

    reporter.project = project
    return sbid


def report_cube_stats(cube, reporter):
    print ('\nReporting cube stats')

    hdr = fits.getheader(cube)
    w = WCS(hdr).celestial
    
    # Cube information
    askapSoftVer = 'N/A'
    askapPipelineVer = 'N/A'
    history = hdr['history'] if 'history' in hdr else []
    askapSoftVerPrefix = 'Produced with ASKAPsoft version '
    askapPipelinePrefix = 'Processed with ASKAP pipeline version '
    for row in history:
        if row.startswith(askapSoftVerPrefix):
            askapSoftVer = row[len(askapSoftVerPrefix):]
        elif row.startswith(askapPipelinePrefix):
            askapPipelineVer = row[len(askapPipelinePrefix):]
    beam = 'N/A'
    if 'BMAJ' in hdr:
        beam_maj = hdr['BMAJ'] * 60 * 60
        beam_min = hdr['BMIN'] * 60 * 60
        beam = '{:.1f} x {:.1f}'.format(beam_maj, beam_min)

    units = 'N/A'
    if 'BUNIT' in hdr:
        units = hdr['BUNIT']

    dims = []
    for i in range(1,int(hdr['NAXIS'])+1):
        dims.append(str(hdr['NAXIS'+str(i)]))
    dimensions = ' x '.join(dims)

    # self.area,self.solid_ang = get_pixel_area(fits, nans=True, ra_axis=self.ra_axis, dec_axis=self.dec_axis, w=w)

    cube_name = os.path.basename(cube)
    section = ReportSection('Image Cube', cube_name)
    section.add_item('ASKAPsoft<br/>version', value=askapSoftVer)
    section.add_item('Pipeline<br/>version', value=askapPipelineVer)
    section.add_item('Synthesised Beam<br/>(arcsec)', value=beam)
    section.add_item('Sky Area<br/>(deg2)', value='')
    section.add_item('Dimensions', value=dimensions)
    section.add_item('Units', value=units)
    reporter.add_section(section)
    return


def check_for_emission(cube, vel_start, vel_end, reporter, dest_folder, ncores=8, redo=False):
    print ('\nChecking for presence of emission in {:.0f} < v < {:.0f}'.format(vel_start, vel_end))

    # Extract a moment 0 map
    slab = extract_slab(cube, vel_start, vel_end)
    if slab is None:
        print ("** No data for the emission range - skipping check **")
        return

    num_channels = slab.shape[0]
    hdr = fits.getheader(cube)
    spec_res_km_s = calc_velocity_res(hdr)

    mom0 = slab.moment0()
    prefix = build_fname(cube, '_mom0')
    folder = get_figures_folder(dest_folder)
    mom0_fname = folder + prefix + '.fits'
    mom0.write(mom0_fname, overwrite=True)

    hi_data = fits.open(mom0_fname)
    plot_title_suffix = "emission region in " + os.path.basename(cube)
    plot_difference_map(hi_data[0], folder+prefix, "Moment 0 map of " + plot_title_suffix)

    # Produce the background plots
    bkg_data = get_bane_background(mom0_fname, folder+prefix, plot_title_suffix, ncores=ncores, redo=redo)
    map_page = folder + '/emission.html'
    rel_map_page = get_figures_folder('.') + '/emission.html'
    output_map_page(map_page, prefix, 'Emission Plots for ' + os.path.basename(cube))

    hi_data = fits.open(folder + prefix+'_bkg.fits')
    max_em = np.nanmax(hi_data[0].data)
    max_em_per_kms = max_em / (spec_res_km_s * num_channels)

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Presence of Emission', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Channels', value='{}'.format(num_channels))
    section.add_item('Large Scale<br/>Emission Map', link=rel_map_page, image='figures/'+prefix+'_bkg_sml.png')
    section.add_item('Emission Histogram', link='figures/'+prefix+'_bkg_hist.png', image='figures/'+prefix+'_bkg_hist_sml.png')
    section.add_item('Max Emission<br/>(Jy beam<sup>-1</sup>)', value='{:.3f}'.format(max_em_per_kms))
    reporter.add_section(section)

    metric = ValidationMetric('Presence of Emission', 
        'Maximum large scale emission intensity in the velocity range where emission is expected.',
        int(max_em_per_kms), assess_metric(max_em_per_kms, 12, 20))
    reporter.add_metric(metric)
    return


def check_for_non_emission(cube, vel_start, vel_end, reporter, dest_folder, ncores=8, redo=False):
    print ('\nChecking for absence of emission in {:.0f} < v < {:.0f}'.format(vel_start, vel_end))

    # Extract a moment 0 map
    slab = extract_slab(cube, vel_start, vel_end)
    if slab is None:
        print ("** No data for the non-emission range - skipping check **")
        return None

    num_channels = slab.shape[0]
    hdr = fits.getheader(cube)
    spec_res_km_s = calc_velocity_res(hdr)

    mom0 = slab.moment0()
    prefix = build_fname(cube, '_mom0_off')
    folder = get_figures_folder(dest_folder)
    mom0_fname = folder + prefix + '.fits'
    mom0.write(mom0_fname, overwrite=True)

    hi_data = fits.open(mom0_fname)
    plot_title_suffix = "non-emission region in " + os.path.basename(cube)
    plot_difference_map(hi_data[0], folder+prefix, "Moment 0 map of " + plot_title_suffix)

    # Produce the background plots
    bkg_data = get_bane_background(mom0_fname, folder+prefix, plot_title_suffix, ncores=ncores, redo=redo)
    map_page = folder + '/off_emission.html'
    rel_map_page = get_figures_folder('.') + '/off_emission.html'
    output_map_page(map_page, prefix, 'Off-line Emission Plots for ' + os.path.basename(cube))

    hi_data = fits.open(folder+prefix+'_bkg.fits')
    max_em = np.nanmax(hi_data[0].data)
    max_em_per_kms = max_em / (spec_res_km_s * num_channels)

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Absence of Off-line Emission', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Channels', value='{}'.format(num_channels))
    section.add_item('Large Scale<br/>Emission Map', link=rel_map_page, image='figures/'+prefix+'_bkg_sml.png')
    section.add_item('Emission Histogram', link='figures/'+prefix+'_bkg_hist.png', image='figures/'+prefix+'_bkg_hist_sml.png')
    section.add_item('Max Emission<br/>(Jy beam<sup>-1</sup>)', value='{:.3f}'.format(max_em_per_kms))
    reporter.add_section(section)

    metric = ValidationMetric('Absence of Off-line Emission', 
        'Maximum large scale emission intensity in the velocity range where emission is not expected.',
        int(max_em_per_kms), assess_metric(max_em_per_kms, 5, 12, low_good=True))
    reporter.add_metric(metric)
    return slab


def calc_theoretical_rms(chan_width, t_obs= 12*60*60, n_ant=36):
    """
    Calculating the theoretical rms noise for ASKAP. Assuming natural weighting and not taking into account fraction of flagged data. 
    Based on ASKAP SEFD measurement in SB 9944.

    Parameters
    ----------
    chan_width : int
        channel width in Hz
    t_obs : int
        duration of the observation in seconds
    n_ant : int
        Number of antennae
        
    Returns
    -------
    rms : int
        Theoretical RMS in mJy
    """
    #cor_eff = 0.8    # correlator efficiency - WALLABY
    cor_eff = 1.0    # correlator efficiency - assumed to be included in the SEFD
    n_pol = 2.0      # Number of polarisation, npol = 2 for images in Stokes I, Q, U, or V
    #sefd = 1700*u.Jy # As measured in SB 9944
    sefd = 1800*u.Jy # Hotan et al 2021
    
    rms_jy = sefd/(cor_eff*math.sqrt(n_pol*n_ant*(n_ant-1)*chan_width*t_obs))

    return rms_jy.to(u.mJy).value



def measure_spectral_line_noise(slab, cube, vel_start, vel_end, reporter, dest_folder, duration, redo=False):
    print ('\nMeasuring the spectral line noise levels across {:.0f} < v < {:.0f}'.format(vel_start, vel_end))

    if slab is None:
        print ("** No data for the non-emission range - skipping check **")
        return

    # Extract header details
    hdr = fits.getheader(cube)
    spec_sys = hdr['SPECSYS']
    axis_num = '3' if hdr['CTYPE3'] != 'STOKES' else '4'
    spec_type = hdr['CTYPE'+axis_num]

    axis = spec_sys + ' ' + spec_type
    spec_res_km_s = calc_velocity_res(hdr)

    # Scale the noise to mJy / 5 kHz channel
    std_data = np.nanstd(slab.unmasked_data[:], axis=0)
    noise_5kHz = std_data / np.sqrt(1 / spec_res_km_s)
    noise_5kHz = noise_5kHz*1000 # Jy => mJy

    # Extract the spectral line noise map
    mom0_prefix = build_fname(cube, '_mom0_off')
    folder = get_figures_folder(dest_folder)
    mom0_fname = folder + mom0_prefix + '.fits'
    prefix = build_fname(cube, '_spectral_noise')
    noise_fname = folder + prefix  + '.fits'
    fits.writeto(noise_fname, noise_5kHz.value, fits.getheader(mom0_fname), overwrite=True)

    # Produce the noise plots
    cube_name = os.path.basename(cube)
    plot_map(folder+prefix, "Spectral axis noise map for " + cube_name, cmap='mako', stretch='arcsinh', 
        colorbar_label=r'Noise level per 5 kHz channel (mJy beam$^{-1}$)')
    plot_histogram(folder+prefix, r'Noise level per 5 kHz channel (mJy beam$^{-1}$)', 'Spectral axis noise for ' + cube_name)
    median_noise_5kHz = np.nanmedian(noise_5kHz.value[noise_5kHz.value!=0.0])

    theoretical_gaskap_noise = calc_theoretical_rms(5000, t_obs=duration*60*60) # mJy per 5 kHz for the observation duration
    print ("Theoretical noise {:.3f} mJy/beam".format(theoretical_gaskap_noise))
    median_ratio = median_noise_5kHz / theoretical_gaskap_noise

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Spectral Line Noise', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Spectral Axis', value=axis)
    section.add_item('Spectral Resolution<br/>(kms)', value='{}'.format(round(spec_res_km_s,2)))
    section.add_item('Spectral Axis<br/>Noise Map', link='figures/'+prefix+'.png', image='figures/'+prefix+'_sml.png')
    section.add_item('Spectral Axis<br/>Noise Histogram', link='figures/'+prefix+'_hist.png', image='figures/'+prefix+'_hist_sml.png')
    section.add_item('Spectral Axis Noise<br/>(mJy per 5 kHz)', value='{:.3f}'.format(median_noise_5kHz))
    section.add_item('Spectral Axis Noise<br/>(vs theoretical for {:.2f} hr)'.format(duration), value='{:.3f}'.format(median_ratio))
    reporter.add_section(section)

    metric = ValidationMetric('Spectral Noise', 
        '1-sigma spectral noise comparison to theoretical per 5 kHz channel for {:.2f} hr observation.'.format(duration),
        round(median_ratio,3), assess_metric(median_ratio, 
        np.sqrt(2), np.sqrt(2)*2, low_good=True))
    reporter.add_metric(metric)

    return


def get_pixel_area(fits_file,flux=0,nans=False,ra_axis=0,dec_axis=1,w=None):

    """For a given image, get the area and solid angle of all non-nan pixels or all pixels below a certain flux (doesn't count pixels=0).
    The RA and DEC axes follow the WCS convention (i.e. starting from 0).

    Arguments:
    ----------
    fits : astropy.io.fits
        The primary axis of a fits image.

    Keyword arguments:
    ------------------
    flux : float
        The flux in Jy, below which pixels will be selected.
    nans : bool
        Derive the area and solid angle of all non-nan pixels.
    ra_axis : int
        The index of the RA axis (starting from 0).
    dec_axis : int
        The index of the DEC axis (starting from 0).
    w : astropy.wcs.WCS
        A wcs object to use for reading the pixel sizes.

    Returns:
    --------
    area : float
        The area in square degrees.
    solid_ang : float
        The solid angle in steradians.

    See Also:
    ---------
    astropy.io.fits
    astropy.wcs.WCS"""
 
    if w is None:
        w = WCS(fits_file.header)

    #count the pixels and derive area and solid angle of all these pixels
    if nans:
        count = fits_file.data[(~np.isnan(fits_file.data)) & (fits_file.data != 0)].shape[0]
    else:
        count = fits_file.data[(fits_file.data < flux) & (fits_file.data != 0)].shape[0]

    area = (count*np.abs(w.wcs.cdelt[ra_axis])*np.abs(w.wcs.cdelt[dec_axis]))
    solid_ang = area*(np.pi/180)**2
    return area,solid_ang


def report_image_stats(image, noise_file, reporter, dest_folder, diagnostics_dir, ncores=8, redo=False):
    print ('\nReporting image stats')

    fits_file = fits.open(image)
    hdr = fits_file[0].header
    w = WCS(hdr).celestial
    

    fig_folder= get_figures_folder(dest_folder)

    # Image information
    askapSoftVer = 'N/A'
    askapPipelineVer = 'N/A'
    history = hdr['history']
    askapSoftVerPrefix = 'Produced with ASKAPsoft version '
    askapPipelinePrefix = 'Processed with ASKAP pipeline version '
    for row in history:
        if row.startswith(askapSoftVerPrefix):
            askapSoftVer = row[len(askapSoftVerPrefix):]
        elif row.startswith(askapPipelinePrefix):
            askapPipelineVer = row[len(askapPipelinePrefix):]
    beam = 'N/A'
    if 'BMAJ' in hdr:
        beam_maj = hdr['BMAJ'] * 60 * 60
        beam_min = hdr['BMIN'] * 60 * 60
        beam = '{:.1f} x {:.1f}'.format(beam_maj, beam_min)

    # Analyse image data
    area,solid_ang = get_pixel_area(fits_file[0], nans=False)
    # if not noise_file:
    #     prefix = build_fname(image, '')
    #     folder = get_figures_folder(dest_folder)
    #     noise_file = get_bane_background(image, folder+prefix, redo=redo, plot=False)
    # rms_map = fits.open(noise_file)[0]
    img_data = fits_file[0].data
    img_peak = np.max(img_data[~np.isnan(img_data)])
    # rms_bounds = rms_map.data > 0
    # img_rms = int(np.median(rms_map.data[rms_bounds])*1e6) #uJy
    # img_peak_bounds = np.max(img_data[rms_bounds])
    # img_peak_pos = np.where(img_data == img_peak_bounds)
    # img_peak_rms = rms_map.data[img_peak_pos][0]
    # dynamic_range = img_peak_bounds/img_peak_rms
    #img_flux = np.sum(img_data[~np.isnan(img_data)]) / (1.133*((beam_maj * beam_min) / (img.raPS * img.decPS))) #divide by beam area

    # Copy pipleine plots
    field_src_plot = copy_existing_image(diagnostics_dir+'/image.i.SB*.cont.restored_sources.png', fig_folder) if diagnostics_dir else None

    image_name = os.path.basename(image)
    section = ReportSection('Image', image_name)
    section.add_item('ASKAPsoft<br/>version', value=askapSoftVer)
    section.add_item('Pipeline<br/>version', value=askapPipelineVer)
    section.add_item('Synthesised Beam<br/>(arcsec)', value=beam)
    add_opt_image_section('Source Map', field_src_plot, fig_folder, dest_folder, section)

    # section.add_item('Median r.m.s.<br/>(uJy)', value='{:.2f}'.format(img_rms))
    # section.add_item('Image peak<br/>(Jy)', value='{:.2f}'.format(img_peak_bounds))
    # section.add_item('Dynamic Range', value='{:.2f}'.format(dynamic_range))
    section.add_item('Sky Area<br/>(deg2)', value='{:.2f}'.format(area))
    reporter.add_section(section)
    return


def set_velocity_range(emvelstr, nonemvelstr):
    emvel = int(emvelstr)
    if not emvel in vel_steps:
        raise ValueError('Velocity {} is not one of the supported GASS velocity steps e.g. 165, 200.'.format(emvel))
    nonemvel = int(nonemvelstr)
    if not nonemvel in vel_steps:
        raise ValueError('Velocity {} is not one of the supported GASS velocity steps e.g. 165, 200.'.format(nonemvel))

    idx = vel_steps.index(emvel)
    if idx +1 >= len(vel_steps):
        raise ValueError('Velocity {} is not one of the supported GASS velocity steps e.g. 165, 200.'.format(emvel))

    # emission_vel_range=(vel_steps[idx],vel_steps[idx+1])*u.km/u.s
    emission_vel_range[0]=vel_steps[idx]*u.km/u.s
    emission_vel_range[1]=vel_steps[idx+1]*u.km/u.s
    print ('\nSet emission velocity range to {:.0f} < v < {:.0f}'.format(emission_vel_range[0], emission_vel_range[1]))

    idx = vel_steps.index(nonemvel)
    if idx +1 >= len(vel_steps):
        raise ValueError('Velocity {} is not one of the supported GASS velocity steps e.g. 165, 200.'.format(emvel))

    # emission_vel_range=(vel_steps[idx],vel_steps[idx+1])*u.km/u.s
    non_emission_val_range[0]=vel_steps[idx]*u.km/u.s
    non_emission_val_range[1]=vel_steps[idx+1]*u.km/u.s
    print ('\nSet non emission velocity range to {:.0f} < v < {:.0f}'.format(non_emission_val_range[0], non_emission_val_range[1]))


def identify_periodicity(spectrum):
    """
    Check if there are periodic features in a spectrum. This tests if there are patterns which are 
    present in the spectrum seperated by a specific number of channels (or lag). i.e. if the same 
    pattern repeats every so many channels. Only features with at least 3-sigma significance are 
    returned.

    Arguments:
    ----------
    spectrum : array-like
        The numerical spectrum.

    Returns:
    --------
    repeats: array
        The lag intervals that have 3-sigma or greater periodic features
    sigma: array
        The significance of each repeat value, in sigma.
    """
    # Use a partial auto-correlation function to identify repeated patterns
    pacf = stattools.pacf(spectrum, nlags=min(50, len(spectrum)//5))
    sd = np.std(pacf[1:])
    significance= pacf/sd
    indexes = (significance>3).nonzero()[0]
    repeats = indexes[indexes>3]
    return repeats, significance[repeats]


def plot_all_spectra(spectra, names, velocities, em_unit, vel_unit, figures_folder, prefix):
    fig = None
    if len(spectra) > 20:
        fig = plt.figure(figsize=(18, 72))
    else:
        fig = plt.figure(figsize=(18, 12))
    num_rows = math.ceil(len(spectra)/3)
    for idx, spectrum in enumerate(spectra):
        label = get_str(names[idx])

        ax = fig.add_subplot(num_rows, 3, idx+1)
        ax.plot(velocities, spectrum, linewidth=1)
        ax.set_title(label)
        ax.grid()
        if idx > 2*num_rows:
            ax.set_xlabel("$v_{LSRK}$ " + '({})'.format(vel_unit))
        if idx % 3 == 0:
            ax.set_ylabel(em_unit)

    fig.tight_layout() 
    fig.savefig(figures_folder+'/'+prefix+'-spectra-individual.pdf')


def plot_overlaid_spectra(spectra, names, velocities, em_unit, vel_unit, figures_folder, cube_name, prefix):
    fig = plt.figure(figsize=(18, 12))
    axes = []
    if len(spectra) > 36:
        for i in range(1,4):
            ax = fig.add_subplot(3,1,i)
            axes.append(ax)
    else:
        ax = fig.add_subplot()
        axes.append(ax)
    for i, spec in enumerate(spectra):
        label = get_str(names[i])
        idx = 0
        if len(axes) > 1:
            interleave = label[-4]
            idx = ord(interleave) - ord('A')
        ax = axes[idx]
        ax.plot(velocities, spec, label=label)
    for idx, ax in enumerate(axes):
        ax.set_xlabel("$v_{LSRK}$ " + '({})'.format(vel_unit))
        ax.set_ylabel(em_unit)
        ax.legend()
        ax.grid()
        if len(axes) > 1:
            ax.set_title('Spectra for all beams in interleave {}'.format(chr(ord('A')+idx)))
        else:
            ax.set_title('Spectra for {} brightest sources in {}'.format(len(spectra), cube_name))
    plt.savefig(figures_folder+'/'+prefix+'-spectra.png')
    plt.savefig(figures_folder+'/'+prefix+'-spectra_sml.png', dpi=16)


def output_spectra_page(filename, prefix, title):
    with open(filename, 'w') as mp:
        mp.write('<html>\n<head><title>{}</title>\n</head>'.format(title))
        mp.write('\n<body>\n<h1>{}</h1>'.format(title))

        output_plot(mp, 'All Spectra', prefix + '-spectra.png')
        output_plot(mp, 'Individual Spectra', prefix + '-spectra-individual.pdf')
        mp.write('\n</body>\n</html>\n')


def plot_periodic_spectrum(spectrum, fig, name):
    ax = fig.add_subplot(211)
    ax.plot(spectrum)
    ax.set_title('Spectrum for ' + name)
    ax.grid()

    ax = fig.add_subplot(212)
    plot_pacf(spectrum, lags=50, ax=ax)

    fig.tight_layout()


def output_periodic_spectra_page(filename, prefix, title, periodic, detections):
    with open(filename, 'w') as mp:
        mp.write('<html>\n<head><title>{}</title>\n</head>'.format(title))
        mp.write('\n<body>\n<h1>{}</h1>'.format(title))

        for idx, src_name in enumerate(periodic):
            output_plot(mp, src_name, prefix + '{}_periodicity.png'.format(src_name))
            mp.write('<p>{}</p>'.format(detections[idx]))
        mp.write('\n</body>\n</html>\n')


def save_spectum(name, velocities, fluxes, ra, dec, spectra_folder):
    spec_table = Table(
        [velocities, fluxes],
        names=['Velocity', 'Emission'],
        meta={'ID': name, 'RA' : ra, 'Dec': dec})
    votable = from_table(spec_table)
    votable.infos.append(Info('RA', 'RA', ra))
    votable.infos.append(Info('Dec', 'Dec', dec))

    writeto(votable, '{}/{}.vot'.format(spectra_folder, name))


def extract_spectra(cube, source_cat, dest_folder, reporter, num_spectra, beam_list, slab_size=40):
    print('\nExtracting spectra for the {} brightest sources in {} and beams listed in {}'.format(
        num_spectra, source_cat, beam_list))

    # Prepare the output folders
    spectra_folder = dest_folder + '/spectra'
    if not os.path.exists(spectra_folder):
        os.makedirs(spectra_folder)
    figures_folder = dest_folder + '/figures'

    # Read the source list and identify the brightest sources
    bright_srcs = []
    bright_src_pos = []
    if source_cat:
        votable = parse(source_cat, pedantic=False)
        sources = votable.get_first_table()
        srcs_tab = sources.to_table()
        for key in ('component_name', 'col_component_name'):
            if key in srcs_tab.keys():
                comp_name_key = key
                break
        bright_idx = np.argsort(sources.array['flux_peak'])[-num_spectra:]
        bright_srcs = sources.array[bright_idx]
        bright_srcs.sort(order=comp_name_key)
        for idx, src in enumerate(bright_srcs):
            pos = SkyCoord(ra=src['ra_deg_cont']*u.deg, dec=src['dec_deg_cont']*u.deg)
            bright_src_pos.append(pos)

    # Read the beams
    beams = []
    if beam_list:
        beams = ascii.read(beam_list)
        beams.add_column(Column(name='pos', data=np.empty((len(beams)), dtype=object)))
        beams.add_column(Column(name='name', data=np.empty((len(beams)), dtype=object)))
        for beam in beams:
            name = '{}-{:02d}'.format(beam['col1'], beam['col2'])
            pos = SkyCoord(ra=beam['col3']*u.rad, dec=beam['col4']*u.rad)
            beam['name'] = name
            beam['pos'] = pos

    # Read the cube
    spec_cube = SpectralCube.read(cube)
    vel_cube = spec_cube.with_spectral_unit(u.m/u.s, velocity_convention='radio')
    wcs = vel_cube.wcs.celestial
    spec_len =vel_cube.shape[0]
    header = fits.getheader(cube)

    # Identify the target pixels for each spectrum
    pix_pos_bright = []
    pix_pos_beam = []
    for idx, source in enumerate(bright_srcs):
        pos = pos = bright_src_pos[idx]
        pixel = pos.to_pixel(wcs=wcs)
        rnd = np.round(pixel)
        pix_pos_bright.append((int(rnd[0]), int(rnd[1])))
    for source in beams:
        pos = source['pos']
        pixel = pos.to_pixel(wcs=wcs)
        rnd = np.round(pixel)
        pix_pos_beam.append((int(rnd[0]), int(rnd[1])))


    # Extract the spectra
    start = time.time()
    print("  ## Started spectra extract at {} ##".format(
          (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))))
    prev = start
    spectra_bright = []
    for p in pix_pos_bright:
        spectra_bright.append(np.zeros(spec_len))
    spectra_beam = []
    for p in pix_pos_beam:
        spectra_beam.append(np.zeros(spec_len))
    
    # Extract using slabs
    unit = None
    prev = time.time()
    for i in range(0,spec_len,slab_size):
        max_idx = min(i+slab_size, spec_len)
        slab = extract_channel_slab(cube, i, max_idx)
        checkpoint = time.time()
        print (slab)
        unit = slab.unit

        for j, pos in enumerate(pix_pos_bright):
            if 0 <= pos[1] < slab.shape[1] and 0 <= pos[0] < slab.shape[2]:
                data = slab[:,pos[1], pos[0]]
                #data = convert_data_to_jy(data, header)
                spectra_bright[j][i:max_idx] = data.value
            else:
                print ("Skipping src {} as y:{} x:{} is outside cube".format(j, pos[1], pos[0]))
        for j, pos in enumerate(pix_pos_beam):
            data = slab[:,pos[1], pos[0]]
            spectra_beam[j][i:max_idx] = data.value
        
        print ("Scanning slab of channels {} to {}, took {:.2f} s".format(i, max_idx-1, checkpoint-prev))
        prev = checkpoint

    end = time.time()
    print("  ## Finished spectra extract at {}, took {:.2f} s ##".format(
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)), end-start))

    # Save the spectra
    names = bright_srcs['component_name']
    for idx, spec in enumerate(spectra_bright):
        name = get_str(names[idx])
        pos = bright_src_pos[idx]
        save_spectum(name, vel_cube.spectral_axis.to(u.km/u.s), spec*vel_cube.unit, pos.ra.deg, pos.dec.deg, spectra_folder)
    for idx, spec in enumerate(spectra_beam):
        name = beams[idx]['name']
        pos = beams[idx]['pos']
        save_spectum(name, vel_cube.spectral_axis.to(u.km/u.s), spec*vel_cube.unit, pos.ra.deg, pos.dec.deg, spectra_folder)
        
    # Plot the spectra
    em_unit = str(vel_cube.unit)
    velocities = vel_cube.spectral_axis.to(u.km/u.s)
    plot_overlaid_spectra(spectra_bright, names, velocities, em_unit, 'km/s', figures_folder, os.path.basename(cube), 'bright')
    plot_all_spectra(spectra_bright, names, velocities, em_unit, 'km/s', figures_folder, 'bright')
    bright_spectra_file = figures_folder+'/bright_spectra.html'
    output_spectra_page(bright_spectra_file, './bright', "Spectra for 15 Brightest Sources")
    if beam_list:
        beam_names = beams['name']
        spec_res_hz = Spectra.get_spec_resolution(header)
        print ('Spec res (hz) {}'.format(spec_res_hz))
        theoretical_noise = calc_theoretical_rms(spec_res_hz)
        print ('Theoretical noise (mJy) {}'.format(theoretical_noise))
        plot_overlaid_spectra(spectra_beam, beam_names, velocities, em_unit, 'km/s', figures_folder, os.path.basename(cube), 'beam')
        Spectra.plot_beam_locs(cube, beams, theoretical_noise, figures_folder+'/beam_comparison', spectra_folder)
        plot_all_spectra(spectra_beam, beam_names, velocities, em_unit, 'km/s', figures_folder, 'beam')
        beam_spectra_file = figures_folder+'/beam_spectra.html'
        output_spectra_page(beam_spectra_file, './beam', "Spectra for centre of each beam")
    
    # Check for periodicity in the spectra
    num_bright_periodic = 0
    bright_periodic = []
    detections = []
    for idx, spec in enumerate(spectra_bright):
        if spec.any():
            repeats, sig = identify_periodicity(spec)
            if len(repeats)>0:
                num_bright_periodic += 1
                name = get_str(names[idx])
                bright_periodic.append(name)
                fig = plt.figure(figsize=(8, 6))
                plot_periodic_spectrum(spec, fig, name)
                fig.savefig(figures_folder+'/{}_periodicity.png'.format(name))
                detections.append("Detected periodicity with lag {} of significance {}".format(repeats, sig))
                print ("Spectrum for {} has periodicity with lag {} of signficance {}".format(name, repeats, sig))
    bright_periodic_str = 'None' if len(bright_periodic) == 0 else '<br/>'.join(bright_periodic)
    output_periodic_spectra_page(figures_folder+'/periodic_spectra.html', './', "Spectra with Periodic Features", bright_periodic, detections)

    # Output the report
    cube_name = os.path.basename(cube)
    section = ReportSection('Spectra', cube_name)
    section.add_item('Bright Source Spectra', link='figures/bright_spectra.html', image='figures/bright-spectra_sml.png')
    section.add_item('Spectra wth periodic features', link='figures/periodic_spectra.html', value=bright_periodic_str)
    if beam_list:
        section.add_item('Beam Centre Spectra', link='figures/beam_spectra.html', image='figures/beam-spectra_sml.png')
        section.add_item('Beam Noise Levels', link='figures/beam_comparison.png', image='figures/beam_comparison_sml.png')
    reporter.add_section(section)

    metric = ValidationMetric('Spectra periodicity', 
        'Number of spectra with repeated patterns with more than 3-sigma significance ',
        num_bright_periodic, assess_metric(num_bright_periodic, 
        1, 5, low_good=True))
    reporter.add_metric(metric)


def copy_existing_image(image_pattern, fig_folder):
    paths = glob.glob(image_pattern)
    if len(paths) == 0:
        return None

    # Copy the file with default permnisisons and metadata
    new_name = fig_folder + "/" + os.path.basename(paths[0])
    shutil.copyfile(paths[0], new_name)
    return new_name

def add_opt_image_section(title, image_path, fig_folder, dest_folder, section, thumb_size_x=140, thumb_size_y=140):
    if image_path == None:
        section.add_item(title, value='N/A')
        return
    
    img_thumb, img_thumb_rel = Diagnostics.make_thumbnail(image_path, fig_folder, dest_folder, size_x=thumb_size_x, 
        size_y=thumb_size_y)
    image_path_rel = os.path.relpath(image_path, dest_folder)
    section.add_item(title, link=image_path_rel, image=img_thumb_rel)

def add_opt_mult_image_section(title, image_paths, fig_folder, dest_folder, section, thumb_size_x=140, thumb_size_y=140):
    if image_paths is None:
        section.add_item(title, value='N/A')
        return

    rel_paths = []
    rel_thumbs = []
    found = False
    for image_path in image_paths:
        if image_path:
            found = True
            img_thumb, img_thumb_rel = Diagnostics.make_thumbnail(image_path, fig_folder, dest_folder, 
                size_x=thumb_size_x, size_y=thumb_size_y)
            image_path_rel = os.path.relpath(image_path, dest_folder)
            rel_thumbs.append(img_thumb_rel)
            rel_paths.append(image_path_rel)
    if found:
        section.add_item(title, link=rel_paths, image=rel_thumbs)
    else:
        section.add_item(title, value='N/A')


def report_calibration(diagnostics_dir, dest_folder, reporter):
    print('\nReporting calibration from ' + diagnostics_dir)

    fig_folder= get_figures_folder(dest_folder)

    bandpass, cal_sbid = Bandpass.get_cal_bandpass(diagnostics_dir)

    # Plot bandpasses
    bp_by_ant_fig = Bandpass.plot_bandpass_by_antenna(bandpass, cal_sbid, fig_folder, 'Calibration')
    #bp_by_ant_thumb, bp_by_ant_thumb_rel = Diagnostics.make_thumbnail(bp_by_ant_fig, fig_folder, dest_folder)
    #bp_by_ant_fig_rel = os.path.relpath(bp_by_ant_fig, dest_folder)

    bp_by_beam_fig = Bandpass.plot_bandpass_by_beam(bandpass, cal_sbid, fig_folder, 'Calibration')
    bp_by_beam_thumb, bp_by_beam_thumb_rel = Diagnostics.make_thumbnail(bp_by_beam_fig, fig_folder, dest_folder)
    bp_by_beam_fig_rel = os.path.relpath(bp_by_beam_fig, dest_folder)

    # Include the pipeline diagnostics
    amp_diag_img = copy_existing_image(diagnostics_dir+'/amplitudesDiagnostics_'+str(cal_sbid)+'.png', fig_folder)
    phase_diag_img = copy_existing_image(diagnostics_dir+'/phasesDiagnostics_'+str(cal_sbid)+'.png', fig_folder)
    cal_param_pdf = copy_existing_image(diagnostics_dir+'/calparameters_*_bp_SB'+str(cal_sbid)+'.smooth.pdf', fig_folder)
    cal_param_pdf_rel = os.path.relpath(cal_param_pdf, dest_folder) if cal_param_pdf else None

    # Output the report
    section = ReportSection('Calibration', '')
    section.add_item('Cal SBID', cal_sbid)
    add_opt_image_section('Bandpass by Antenna', bp_by_ant_fig, fig_folder, dest_folder, section)
    add_opt_image_section('Bandpass by Beam', bp_by_beam_fig, fig_folder, dest_folder, section)
    add_opt_image_section('Amplitude Diagnostics', amp_diag_img, fig_folder, dest_folder, section)
    add_opt_image_section('Phase Diagnostics', phase_diag_img, fig_folder, dest_folder, section)
    if cal_param_pdf_rel:
        section.add_item('Parameters', value="pdf", link=cal_param_pdf_rel)
    reporter.add_section(section)


def report_diagnostics(diagnostics_dir, sbid, dest_folder, reporter, sched_info, obs_metadata, short_len=500, long_len=2000):
    print('\nReporting diagnostics')

    fig_folder= get_figures_folder(dest_folder)
    is_closepack = sched_info.footprint == None or sched_info.footprint.startswith('closepack')

    # Extract metadata
    chan_width, cfreq, nchan = Diagnostics.get_freq_details(diagnostics_dir)
    chan_width_kHz = round(chan_width/1000., 3) # convert Hz to kHz

    theoretical_rms_mjy = np.zeros(len(obs_metadata.fields))
    total_rows = sum([field.num_rows for field in obs_metadata.fields])
    for idx, field in enumerate(obs_metadata.fields):
        field_tobs = obs_metadata.tobs * field.num_rows / total_rows
        theoretical_rms_mjy[idx] = calc_theoretical_rms(chan_width, t_obs=field_tobs)

    # Extract flagging details
    flag_stat_beams, n_flag_ant_beams, ant_flagged_in_all, pct_integ_flagged, baseline_flag_pct, pct_each_integ_flagged, bad_chan_pct_count = Diagnostics.get_flagging_stats(
        diagnostics_dir, fig_folder)
    print("Antenna flagged in all:", ant_flagged_in_all)
    flagged_ant_desc = ", ".join(ant_flagged_in_all) if len(ant_flagged_in_all) > 0 else 'None'
    pct_short_base_flagged, pct_medium_base_flagged, pct_long_base_flagged = Diagnostics.calc_flag_percent(
        baseline_flag_pct, short_len=short_len, long_len=long_len)
    pct_chan_unflagged = Diagnostics.calc_pct_channels_unflagged(bad_chan_pct_count)

    # Extract beam RMS
    beam_exp_rms = Diagnostics.calc_beam_exp_rms(flag_stat_beams, theoretical_rms_mjy)
    rms_min = np.min(beam_exp_rms)
    rms_max = np.max(beam_exp_rms)
    rms_range_pct = round((rms_max-rms_min)/rms_min*100,1)

    # Plot beam stats
    beam_nums = Diagnostics.get_beam_numbers_closepack()

    flagged_vis_fig = Diagnostics.plot_flag_stat(flag_stat_beams, beam_nums, sbid, fig_folder, closepack=is_closepack)
    flagged_ant_fig = Diagnostics.plot_flag_ant(n_flag_ant_beams, beam_nums, sbid, fig_folder, closepack=is_closepack)
    beam_exp_rms_fig = Diagnostics.plot_beam_exp_rms(beam_exp_rms, beam_nums, sbid, fig_folder, closepack=is_closepack)

    baseline_fig = Diagnostics.plot_baselines(baseline_flag_pct, fig_folder, sbid, short_len=short_len, long_len=long_len)
    flag_ant_file_rel = os.path.relpath(fig_folder+'/flagged_antenna.txt', dest_folder)

    integ_flag_fig = Diagnostics.plot_integrations(pct_each_integ_flagged, sbid, fig_folder)
    flag_pct_dist_fig = Diagnostics.plot_flagging_distribution(bad_chan_pct_count, sbid, fig_folder)
    

    # Output the report
    section = ReportSection('Diagnostics', '')
    section.add_item('Completely Flagged Antennas', flagged_ant_desc, link=flag_ant_file_rel)
    section.add_item('Integrations Completely<br/>Flagged (%)', pct_integ_flagged)
    add_opt_image_section('Flagging over Time', integ_flag_fig, fig_folder, dest_folder, section) #, thumb_size_x=70, thumb_size_y=70)
    add_opt_image_section('Channel Flagging', flag_pct_dist_fig, fig_folder, dest_folder, section) #, thumb_size_x=70, thumb_size_y=70)
    section.add_item('Short Baselines<br/>Flagged (%)', pct_short_base_flagged)
    section.add_item('Medium Baselines<br/>Flagged (%)', pct_medium_base_flagged)
    section.add_item('Long Baselines<br/>Flagged (%)', pct_long_base_flagged)
    add_opt_image_section('Baselines', baseline_fig, fig_folder, dest_folder, section) #, thumb_size_x=70, thumb_size_y=70)
    section.start_new_row()
    section.add_item('Channel Width (kHz)', chan_width_kHz)
    add_opt_image_section('Flagged Visibilities', flagged_vis_fig, fig_folder, dest_folder, section) #, thumb_size_x=140, thumb_size_y=70)
    add_opt_image_section('Flagged Antennas', flagged_ant_fig, fig_folder, dest_folder, section) #, thumb_size_x=70, thumb_size_y=70)
    add_opt_image_section('Expected RMS per channel', beam_exp_rms_fig, fig_folder, dest_folder, section) #, thumb_size_x=70, thumb_size_y=70)
    reporter.add_section(section)

    metric = ValidationMetric('Flagged Short Baselines', 
        'Percent of short baselines ({}m or less) flagged across all integrations and all beams'.format(short_len),
        pct_short_base_flagged, assess_metric(pct_short_base_flagged, 
        20, 40, low_good=True))
    reporter.add_metric(metric)
    metric = ValidationMetric('Flagged Long Baselines', 
        'Percent of long baselines ({}m or more) flagged across all integrations and all beams'.format(long_len),
        pct_long_base_flagged, assess_metric(pct_long_base_flagged, 
        30, 45, low_good=True))
    reporter.add_metric(metric)
    metric = ValidationMetric('Unflagged Integrations', 
        'Percent of integrations with less than 5% of channels flagged',
        pct_chan_unflagged, assess_metric(pct_chan_unflagged, 
        70, 50))
    reporter.add_metric(metric)
    metric = ValidationMetric('Expected RMS Difference', 
        'The percentage change of expected RMS across the field.',
        rms_range_pct, assess_metric(rms_range_pct, 
        10, 30, low_good=True))
    reporter.add_metric(metric)


def report_self_cal(cube, image, obs_metadata, dest_folder, reporter):
    print('\nReporting self calibration')

    fig_folder= get_figures_folder(dest_folder)
    field_plots = []
    field_names = ""
    total_bad_beams = 0
    max_bad_ant = 0

    for i,field in enumerate(obs_metadata.fields):
        if i > 0:
            field_names += '<br/>'
        field_names += field.name
        folder = Diagnostics.find_subdir(cube, image, field.name)
        if folder:
            sc = SelfCal.prepare_self_cal_set(folder)
            plots = SelfCal.plot_self_cal_set(sc, fig_folder)
            field_plots.append(plots)
            num_bad_beams, num_bad_ant = SelfCal.calc_phase_stability(sc)
            print("In field {} found {} bad beams and {} bad antennas".format(field.name, num_bad_beams, num_bad_ant))
            total_bad_beams += num_bad_beams
            max_bad_ant = max(max_bad_ant, num_bad_ant)
        else:
            field_plots.append([None, None, None])
    plot_array = np.asarray(field_plots)
    print ("Overall found {} bad beams and {} bad antennas.".format(total_bad_beams, max_bad_ant))

    # Output the report
    section = ReportSection('Self Calibration', '')
    section.add_item('Field(s)', value=field_names)
    add_opt_mult_image_section('Phase Stability', plot_array[:,0], fig_folder, dest_folder, section)
    add_opt_mult_image_section('Phase Summary', plot_array[:,1], fig_folder, dest_folder, section)
    add_opt_mult_image_section('All Phases', plot_array[:,2], fig_folder, dest_folder, section)
    reporter.add_section(section)

    metric = ValidationMetric('Bad Phase Antenna', 
        'Number of unflagged antenna with bad phase solutions',
        max_bad_ant, assess_metric(max_bad_ant, 
        1, 3, low_good=True))
    reporter.add_metric(metric)


def log_config(args, dest_folder, figures_folder):
    print ('Configuration:')
    print (' {: >20} : {}'.format('cube', args.cube))
    print (' {: >20} : {}'.format('image', args.image))
    print (' {: >20} : {}'.format('source_cat', args.source_cat))
    print (' {: >20} : {}'.format('beam_list', args.beam_list))
    print (' {: >20} : {}'.format('duration', args.duration))
    print (' {: >20} : {}'.format('emvel', args.emvel))
    print (' {: >20} : {}'.format('nonemvel', args.nonemvel))
    print (' {: >20} : {}'.format('noise', args.noise))
    print (' {: >20} : {}'.format('redo', args.redo))
    print (' {: >20} : {}'.format('num_spectra', args.num_spectra))
    print (' {: >20} : {}'.format('dest_folder', dest_folder))
    print (' {: >20} : {}'.format('figures_folder', figures_folder))
    print ('')


def main():
    # Parse command line options
    args = parseargs()

    start = time.time()
    print("#### Started validation at {} ####".format(
          (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))))

    #ignore astropy warnings 
    warnings.simplefilter('ignore', AstropyWarning)   

    dest_folder = args.output
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    figures_folder = dest_folder + '/figures'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
    log_config(args, dest_folder, figures_folder)

    if args.cube and (not os.path.exists(args.cube) or not os.path.isfile(args.cube)):
        raise ValueError('Cube {} could not be found or is not a file.'.format(args.cube))
    if args.image and (not os.path.exists(args.image) or not os.path.isfile(args.image)):
        raise ValueError('Image {} could not be found or is not a file.'.format(args.image))
    if not args.cube and not args.image:
        raise ValueError('You must supply either an image or a cube to validate.')

    if args.source_cat and (not os.path.exists(args.source_cat) or not os.path.isfile(args.source_cat)):
        raise ValueError('Source catalogue {} could not be found or is not a file.'.format(args.source_cat))

    if args.emvel:
        set_velocity_range(args.emvel, args.nonemvel)

    if args.cube:
        print ('\nChecking quality level of GASKAP HI cube:', args.cube)
        obs_img = args.cube
        metrics_subtitle = 'GASKAP HI Validation Metrics'
    else: 
        print ('\nChecking quality level of ASKAP image:', args.image)
        obs_img = args.image
        metrics_subtitle = 'ASKAP Observation Diagnostics Metrics'
    cube_name = os.path.basename(obs_img)
    reporter = ValidationReport('GASKAP Validation Report: {}'.format(cube_name), metrics_subtitle=metrics_subtitle)

    sched_info = Diagnostics.get_sched_info(obs_img)
    diagnostics_dir = Diagnostics.find_diagnostics_dir(args.cube, args.image)
    obs_metadata = Diagnostics.get_metadata(diagnostics_dir) if diagnostics_dir else None
    sbid = report_observation(obs_img, dest_folder, reporter, args.duration, sched_info, obs_metadata)

    if args.cube:
        report_cube_stats(args.cube, reporter)

        check_for_emission(args.cube, emission_vel_range[0], emission_vel_range[1], reporter, dest_folder, redo=args.redo)
        slab = check_for_non_emission(args.cube, non_emission_val_range[0], non_emission_val_range[1], reporter, dest_folder, redo=args.redo)
        measure_spectral_line_noise(slab, args.cube, non_emission_val_range[0], non_emission_val_range[1], reporter, dest_folder, args.duration, redo=args.redo)
        if args.source_cat or args.beam_list:
            extract_spectra(args.cube, args.source_cat, dest_folder, reporter, args.num_spectra, args.beam_list)

    if args.image:
        report_image_stats(args.image, args.noise, reporter, dest_folder, diagnostics_dir, redo=args.redo)

    if diagnostics_dir:
        report_calibration(diagnostics_dir, dest_folder, reporter)
        report_diagnostics(diagnostics_dir, sbid, dest_folder, reporter, sched_info, obs_metadata)
    if obs_metadata:
        report_self_cal(args.cube, args.image, obs_metadata, dest_folder, reporter)

    print ('\nProducing report to', dest_folder)
    output_html_report(reporter, dest_folder)
    output_metrics_xml(reporter, dest_folder)

    end = time.time()
    print("#### Completed validation at {} ####".format(
          (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)))))
    print('\nChecks completed in {:.02f} s'.format((end - start)))
    return 0

    
if __name__ == '__main__':
    exit(main())
