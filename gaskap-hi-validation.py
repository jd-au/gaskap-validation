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
import time

import matplotlib
matplotlib.use('agg')

import aplpy
from astropy.constants import k_B
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.io.votable import parse, from_table, writeto
from astropy.table import Table, Column
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from radio_beam import Beam
from spectral_cube import SpectralCube
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_pacf

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


def plot_histogram(file_prefix, xlabel):
    data = fits.getdata(file_prefix+'.fits')
    flat = data.flatten()
    flat = flat[~np.isnan(flat)]
    v =plt.hist(flat, bins=200, bottom=1, log=True, histtype='step')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.savefig(file_prefix+'_hist.png')
    plt.savefig(file_prefix+'_hist_sml.png', dpi=16)
    plt.close()

    
def plot_map(file_prefix, cmap='magma', stretch='linear'):
    gc = aplpy.FITSFigure(file_prefix+'.fits')
    gc.show_colorscale(cmap=cmap, stretch=stretch)
    gc.add_colorbar()
    gc.add_grid()
    gc.savefig(filename=file_prefix+'.png')
    gc.savefig(filename=file_prefix+'_sml.png', dpi=10 )
    gc.close()


def plot_difference_map(hdu, file_prefix, vmax=None):
    # Initiate a figure and axis object with WCS projection information
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection=wcs)

    im = ax.imshow(hdu.data, cmap='RdBu_r',vmax=vmax)
    ax.invert_yaxis() 

    ax.set_xlabel("Right Ascension (degrees)", fontsize=16)
    ax.set_ylabel("Declination (degrees)", fontsize=16)
    ax.grid(color = 'gray', ls = 'dotted', lw = 2)
    cbar = plt.colorbar(im, pad=.07)

    plt.savefig(file_prefix+'.png')
    plt.savefig(file_prefix+'_sml.png', dpi=10 )

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

    if slab.unmasked_data[0,0,0].unit != u.Jy:
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

def get_bane_background(infile, outfile_prefix, ncores=8, redo=False, plot=True):
    background_prefix = outfile_prefix+'_bkg'
    background_file = background_prefix + '.fits'
    if redo or not os.path.exists(background_file):
        cmd = "BANE --cores={0} --out={1} {2}".format(ncores, outfile_prefix, infile)
        print (cmd)
        os.system(cmd)
    
    if plot:
        plot_map(background_prefix)
        plot_histogram(background_prefix, 'Emission (Jy beam^{-1} km s^{-1})')
        plot_map(outfile_prefix+'_rms')
    
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


def report_observation(image, reporter):
    hdr = fits.getheader(image)
    w = WCS(hdr).celestial

    sbid = hdr['SBID'] if 'SBID' in hdr else ''
    project = hdr['PROJECT'] if 'PROJECT' in hdr else ''
    proj_link = None
    if project.startswith('AS'):
        proj_link = "https://confluence.csiro.au/display/askapsst/{0}+Data".format(project)

    date = hdr['DATE-OBS']
    duration = float(hdr['DURATION'])/3600 if 'DURATION' in hdr else 0

    naxis1 = int(hdr['NAXIS1'])
    naxis2 = int(hdr['NAXIS2'])
    pixcrd = np.array([[naxis1/2, naxis2/2]])
    centre = w.all_pix2world(pixcrd,1)
    centre = SkyCoord(ra=centre[0][0], dec=centre[0][1], unit="deg,deg").to_string(style='hmsdms',sep=':')

    # spectral axis
    spectral_unit = 'None'
    spectral_range = ''
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

    section = ReportSection('Observation')
    section.add_item('SBID', value=sbid)
    section.add_item('Project', value=project, link=proj_link)
    section.add_item('Date', value=date)
    section.add_item('Duration<br/>(hours)', value='{:.2f}'.format(duration))
    section.add_item('Field Centre', value=centre)
    section.add_item('{}<br/>({})'.format(spec_title, spectral_unit), value=spectral_range)
    reporter.add_section(section)


def report_cube_stats(cube, reporter):
    print ('\nReporting cube stats')

    hdr = fits.getheader(cube)
    w = WCS(hdr).celestial
    
    # Cube information
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
    mom0 = slab.moment0()
    prefix = build_fname(cube, '_mom0')
    folder = get_figures_folder(dest_folder)
    mom0_fname = folder + prefix + '.fits'
    mom0.write(mom0_fname, overwrite=True)

    hi_data = fits.open(mom0_fname)
    plot_difference_map(hi_data[0], folder+prefix)

    # Produce the background plots
    bkg_data = get_bane_background(mom0_fname, folder+prefix, ncores=ncores, redo=redo)
    map_page = folder + '/emission.html'
    rel_map_page = get_figures_folder('.') + '/emission.html'
    output_map_page(map_page, prefix, 'Emission Plots')

    hi_data = fits.open(folder + prefix+'_bkg.fits')
    max_em = np.nanmax(hi_data[0].data)

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Presence of Emission', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Channels', value='{}'.format(num_channels))
    section.add_item('Large Scale<br/>Emission Map', link=rel_map_page, image='figures/'+prefix+'_bkg_sml.png')
    section.add_item('Emission Histogram', link='figures/'+prefix+'_bkg_hist.png', image='figures/'+prefix+'_bkg_hist_sml.png')
    section.add_item('Max Emission<br/>(Jy km s<sup>-1</sup> beam<sup>-1</sup>)', value='{:.3f}'.format(max_em))
    reporter.add_section(section)

    metric = ValidationMetric('Presence of Emission', 
        'Maximum large scale emission intensity in the velocity range where emission is expected.',
        max_em, assess_metric(max_em, 800, 1000))
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
    mom0 = slab.moment0()
    prefix = build_fname(cube, '_mom0_off')
    folder = get_figures_folder(dest_folder)
    mom0_fname = folder + prefix + '.fits'
    mom0.write(mom0_fname, overwrite=True)

    hi_data = fits.open(mom0_fname)
    plot_difference_map(hi_data[0], folder+prefix)

    # Produce the background plots
    bkg_data = get_bane_background(mom0_fname, folder+prefix, ncores=ncores, redo=redo)
    map_page = folder + '/off_emission.html'
    rel_map_page = get_figures_folder('.') + '/off_emission.html'
    output_map_page(map_page, prefix, 'Off-line Emission Plots')

    hi_data = fits.open(folder+prefix+'_bkg.fits')
    max_em = np.nanmax(hi_data[0].data)

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Absence of Off-line Emission', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Channels', value='{}'.format(num_channels))
    section.add_item('Large Scale<br/>Emission Map', link=rel_map_page, image='figures/'+prefix+'_bkg_sml.png')
    section.add_item('Emission Histogram', link='figures/'+prefix+'_bkg_hist.png', image='figures/'+prefix+'_bkg_hist_sml.png')
    section.add_item('Max Emission<br/>(Jy km s<sup>-1</sup> beam<sup>-1</sup>)', value='{:.3f}'.format(max_em))
    reporter.add_section(section)

    metric = ValidationMetric('Absence of Off-line Emission', 
        'Maximum large scale emission intensity in the velocity range where emission is not expected.',
        max_em, assess_metric(max_em, 200, 500, low_good=True))
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
    cor_eff = 0.8    # correlator efficiency - WALLABY
    n_pol = 2.0      # Number of polarisation, npol = 2 for images in Stokes I, Q, U, or V
    sefd = 1700*u.Jy # As measured in SB 9944
    
    rms_jy = sefd/(cor_eff*math.sqrt(n_pol*n_ant*(n_ant-1)*chan_width*t_obs))

    return rms_jy.to(u.mJy).value



def measure_spectral_line_noise(slab, cube, vel_start, vel_end, reporter, dest_folder, redo=False):
    print ('\nMeasuring the spectral line noise levels across {:.0f} < v < {:.0f}'.format(vel_start, vel_end))

    if slab is None:
        print ("** No data for the non-emission range - skipping check **")
        return

    # Extract the spectral line noise map
    std_data = np.nanstd(slab.unmasked_data[:], axis=0)
    mom0_prefix = build_fname(cube, '_mom0_off')
    folder = get_figures_folder(dest_folder)
    mom0_fname = folder + mom0_prefix + '.fits'
    prefix = build_fname(cube, '_spectral_noise')
    noise_fname = folder + prefix  + '.fits'
    fits.writeto(noise_fname, std_data.value, fits.getheader(mom0_fname), overwrite=True)

    # Produce the noise plots
    plot_map(folder+prefix, cmap='plasma', stretch='arcsinh')
    plot_histogram(folder+prefix, 'Noise level per channel (Jy beam^{-1})')
    median_noise = np.nanmedian(std_data.value[std_data.value!=0.0])

    # Extract header details
    hdr = fits.getheader(cube)
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

    axis = spec_sys + ' ' + spec_type
    spec_res_km_s = np.abs(spec_delt) / spectral_conversion
    if spectral_unit == 'MHz':
        spec_res_km_s = spec_res_km_s/5e-4*0.1 # 0.5 kHz = 0.1 km/s
    #elif spec_unit == 'Hz':
    #        spec_res_km_s = spec_res_km_s/500*0.1 # 0.5 kHz = 0.1 km/s
    #elif spec_unit == 'kHz':
    #    spec_res_km_s = spec_res_km_s/0.5*0.1 # 0.5 kHz = 0.1 km/s

    median_noise_5kHz = median_noise / np.sqrt(1 / spec_res_km_s)
    median_noise_5kHz *= 1000 # Jy => mJy

    theoretical_gaskap_noise = calc_theoretical_rms(5000) # mJy per 5 kHz for 12 hr observation
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
    section.add_item('Spectral Axis Noise<br/>(vs theoretical)', value='{:.3f}'.format(median_ratio))
    reporter.add_section(section)

    metric = ValidationMetric('Spectral Noise', 
        '1-sigma spectral noise comparison to theoretical per 5 kHz channel.',
        median_ratio, assess_metric(median_ratio, 
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


def report_image_stats(image, noise_file, reporter, dest_folder, ncores=8, redo=False):
    print ('\nReporting image stats')

    fits_file = fits.open(image)
    hdr = fits_file[0].header
    w = WCS(hdr).celestial
    
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

    image_name = os.path.basename(image)
    section = ReportSection('Image', image_name)
    section.add_item('ASKAPsoft<br/>version', value=askapSoftVer)
    section.add_item('Pipeline<br/>version', value=askapPipelineVer)
    section.add_item('Synthesised Beam<br/>(arcsec)', value=beam)
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
        raise ValueError('Velocity {} is not one of the supported GASS velocity steps e.g. 165, 200.'.format(emvel))

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
        bright_idx = np.argsort(sources.array['flux_peak'])[-num_spectra:]
        bright_srcs = sources.array[bright_idx]
        bright_srcs.sort(order='component_name')
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
    for i in range(0,spec_len,slab_size):
        checkpoint = time.time()
        max_idx = min(i+slab_size, spec_len)
        slab = extract_channel_slab(cube, i, max_idx)
        print (slab)
        unit = slab.unit

        for j, pos in enumerate(pix_pos_bright):
            data = slab[:,pos[0], pos[1]]
            #data = convert_data_to_jy(data, header)
            spectra_bright[j][i:max_idx] = data.value
        for j, pos in enumerate(pix_pos_beam):
            data = slab[:,pos[0], pos[1]]
            spectra_beam[j][i:max_idx] = data.value
        
        print ("Scanning slab of channels {} to {}, took {:.2f} s".format(i, max_idx-1, checkpoint-prev))
        prev = checkpoint

    end = time.time()
    print("  ## Finished spectra extract at {}, took {:.2f} s ##".format(
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)), end-start))
        
    # Plot the spectra
    names = bright_srcs['component_name']
    em_unit = str(vel_cube.unit)
    velocities = vel_cube.spectral_axis.to(u.km/u.s)
    plot_overlaid_spectra(spectra_bright, names, velocities, em_unit, 'km/s', figures_folder, os.path.basename(cube), 'bright')
    plot_all_spectra(spectra_bright, names, velocities, em_unit, 'km/s', figures_folder, 'bright')
    bright_spectra_file = figures_folder+'/bright_spectra.html'
    output_spectra_page(bright_spectra_file, './bright', "Spectra for 15 Brightest Sources")
    plot_overlaid_spectra(spectra_beam, beams['name'], velocities, em_unit, 'km/s', figures_folder, os.path.basename(cube), 'beam')
    plot_all_spectra(spectra_beam, beams['name'], velocities, em_unit, 'km/s', figures_folder, 'beam')
    beam_spectra_file = figures_folder+'/beam_spectra.html'
    output_spectra_page(beam_spectra_file, './beam', "Spectra for centre of each beam")

    # Save the spectra
    for idx, spec in enumerate(spectra_bright):
        name = get_str(names[idx])
        pos = bright_src_pos[idx]
        save_spectum(name, vel_cube.spectral_axis.to(u.km/u.s), spec*vel_cube.unit, pos.ra.deg, pos.dec.deg, spectra_folder)
    for idx, spec in enumerate(spectra_beam):
        name = beams[idx]['name']
        pos = beams[idx]['pos']
        save_spectum(name, vel_cube.spectral_axis.to(u.km/u.s), spec*vel_cube.unit, pos.ra.deg, pos.dec.deg, spectra_folder)
    
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
    section.add_item('Beam Centre Spectra', link='figures/beam_spectra.html', image='figures/beam-spectra_sml.png')
    reporter.add_section(section)

    metric = ValidationMetric('Spectra periodicity', 
        'Number of spectra with repeated patterns with more than 3-sigma significance ',
        num_bright_periodic, assess_metric(num_bright_periodic, 
        1, 5, low_good=True))
    reporter.add_metric(metric)


def main():
    start = time.time()
    print("#### Started validation at {} ####".format(
          (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))))

    # Parse command line options
    args = parseargs()
    dest_folder = args.output
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    figures_folder = dest_folder + '/figures'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

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
    else: 
        print ('\nChecking quality level of GASKAP image:', args.image)
        obs_img = args.image
    cube_name = os.path.basename(obs_img)
    reporter = ValidationReport('GASKAP Validation Report: {}'.format(cube_name))
    report_observation(obs_img, reporter)

    if args.cube:
        report_cube_stats(args.cube, reporter)

        check_for_emission(args.cube, emission_vel_range[0], emission_vel_range[1], reporter, dest_folder, redo=args.redo)
        slab = check_for_non_emission(args.cube, non_emission_val_range[0], non_emission_val_range[1], reporter, dest_folder, redo=args.redo)
        measure_spectral_line_noise(slab, args.cube, non_emission_val_range[0], non_emission_val_range[1], reporter, dest_folder, redo=args.redo)
        if args.source_cat or args.beam_list:
            extract_spectra(args.cube, args.source_cat, dest_folder, reporter, args.num_spectra, args.beam_list)

    if args.image:
        report_image_stats(args.image, args.noise, reporter, dest_folder, redo=args.redo)

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
