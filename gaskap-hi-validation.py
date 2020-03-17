#!/usr/bin/env python -u

# Validation script for GASKAP HI data
#

# Author James Dempsey
# Date 23 Nov 219


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
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from radio_beam import Beam
from spectral_cube import SpectralCube

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
    parser.add_argument("-o", "--output", help="The folder in which to save the validation report and associated figures.", default='report')
    parser.add_argument("-e", "--emvel", required=False, help="The low velocity bound of the velocity region where emission is expected.")
    parser.add_argument("-n", "--nonemvel", required=False, 
                        help="The low velocity bound of the velocity region where emission is not expected.", default='-100')

    parser.add_argument("-N", "--noise", required=False, help="Use this fits image of the local rms. Default is to run BANE", default=None)
    parser.add_argument("-r", "--redo", help="Rerun all steps, even if intermediate files are present.", default=False,
                        action='store_true')

    args = parser.parse_args()
    return args

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
    gc.show_colorbar()
    gc.show_grid()
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


def extract_slab(filename, vel_start, vel_end):
    cube = SpectralCube.read(filename)
    vel_cube = cube.with_spectral_unit(u.m/u.s, velocity_convention='radio')
    slab = vel_cube.spectral_slab(vel_start, vel_end)

    header = fits.getheader(filename)
    my_beam = Beam.from_fits_header(header)
    restfreq = 	1.420405752E+09*u.Hz
    if 'RESTFREQ' in header.keys():
        restfreq = header['RESTFREQ']*u.Hz
    elif 'RESTFRQ' in header.keys():
        restfreq = header['RESTFRQ']*u.Hz


    if slab.unmasked_data[0,0,0].unit != u.Jy:
        slab.allow_huge_operations=True
        slab = slab.to(u.Jy, equivalencies=u.brightness_temperature(my_beam, restfreq))
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
            spectral_unit = hdr['CUNIT'+str(i)]
            spectral_conversion = 1
            if spectral_unit == 'Hz':
                spectral_conversion = 1e6
                spectral_unit = 'MHz'
            elif spectral_unit == 'kHz':
                spectral_conversion = 1e3
                spectral_unit = 'MHz'
            
            step = float(hdr['CDELT'+str(i)])
            #print ('step {} rval {} rpix {} naxis {}'.format(step, hdr['CRVAL'+str(i)], hdr['CRPIX'+str(i)], hdr['NAXIS'+str(i)]))
            spec_start = (float(hdr['CRVAL'+str(i)]) - (step*(float(hdr['CRPIX'+str(i)])-1)))/spectral_conversion
            if int(hdr['NAXIS'+str(i)]) > 1:
                spec_end = spec_start + (step * (int(hdr['NAXIS'+str(i)]-1)))/spectral_conversion
                spectral_range = '{:0.3f} - {:0.3f}'.format(spec_start, spec_end)
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

    # self.area,self.solid_ang = get_pixel_area(fits, nans=True, ra_axis=self.ra_axis, dec_axis=self.dec_axis, w=w)

    cube_name = os.path.basename(cube)
    section = ReportSection('Image Cube', cube_name)
    section.add_item('ASKAPsoft<br/>version', value=askapSoftVer)
    section.add_item('Pipeline<br/>version', value=askapPipelineVer)
    section.add_item('Synthesised Beam<br/>(arcsec)', value=beam)
    section.add_item('Sky Area<br/>(deg2)', value='')
    reporter.add_section(section)
    return


def check_for_emission(cube, vel_start, vel_end, reporter, dest_folder, ncores=8, redo=False):
    print ('\nChecking for presence of emission in {:.0f} < v < {:.0f}'.format(vel_start, vel_end))

    # Extract a moment 0 map
    slab = extract_slab(cube, vel_start, vel_end)
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
    Based on WALLABY validation.

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
    t_sys = 50*u.K   # WALLABY
    ant_diam = 12 *u.m   # ASKAP
    aper_eff = 0.8  # aperture efficiency - WALLABY
    cor_eff = 0.8    # correlator efficiency - WALLABY
    n_pol = 2.0      # Number of polarisation, npol = 2 for images in Stokes I, Q, U, or V
    
    ant_area = math.pi*(ant_diam/2)**2.
    ant_eff = ant_area * aper_eff
    sefd = (2. * k_B * t_sys/ant_eff).to(u.Jy)
    rms_jy = sefd/(cor_eff*math.sqrt(n_pol*n_ant*(n_ant-1)*chan_width*t_obs))

    return rms_jy.to(u.mJy).value



def measure_spectral_line_noise(slab, cube, vel_start, vel_end, reporter, dest_folder, redo=False):
    print ('\nMeasuring the spectral line noise levels across {:.0f} < v < {:.0f}'.format(vel_start, vel_end))
    # Exract the spectral line noise map
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
    median_noise = np.nanmedian(std_data.value)

    # Extract header details
    hdr = fits.getheader(cube)
    spec_sys = hdr['SPECSYS']
    axis = '3' if hdr['CTYPE3'] != 'STOKES' else '4'
    spec_type = hdr['CTYPE'+axis]
    spec_unit = hdr['CUNIT'+axis] if 'CUNIT'+axis in hdr.keys() else None
    spec_delt = hdr['CDELT'+axis]
    print ('CDELT={}, CUNIT={}'.format(spec_delt, spec_unit))

    axis = spec_sys + ' ' + spec_type
    spec_res_km_s = np.abs(spec_delt)
    if spec_unit == 'm/s':
        spec_res_km_s /= 1000
    elif spec_unit == 'Hz':
            spec_res_km_s = spec_res_km_s/500*0.1 # 0.5 kHz = 0.1 km/s
    elif spec_unit == 'kHz':
        spec_res_km_s = spec_res_km_s/0.5*0.1 # 0.5 kHz = 0.1 km/s

    median_noise_5kHz = median_noise / np.sqrt(1 / spec_res_km_s)
    median_noise_5kHz *= 1000 # Jy => mJy

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Spectral Line Noise', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Spectral Axis', value=axis)
    section.add_item('Spectral Resolution<br/>(kms)', value='{}'.format(round(spec_res_km_s,2)))
    section.add_item('Spectral Axis<br/>Noise Map', link='figures/'+prefix+'.png', image='figures/'+prefix+'_sml.png')
    section.add_item('Spectral Axis<br/>Noise Histogram', link='figures/'+prefix+'_hist.png', image='figures/'+prefix+'_hist_sml.png')
    section.add_item('Spectral Axis Noise<br/>(mJy per 5 kHz)', value='{:.3f}'.format(median_noise_5kHz))
    reporter.add_section(section)

    theoretical_gaskap_noise = calc_theoretical_rms(5000) # mJy per 5 kHz for 12 hr observation
    metric = ValidationMetric('Spectral Noise', 
        '1-sigma spectral noise level per 5 kHz channel.',
        median_noise_5kHz, assess_metric(median_noise_5kHz, 
        theoretical_gaskap_noise* np.sqrt(2), theoretical_gaskap_noise* np.sqrt(2)*2, low_good=True))
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
    


def main():
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

    if args.emvel:
        set_velocity_range(args.emvel, args.nonemvel)

    start = time.time()

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

    if args.image:
        report_image_stats(args.image, args.noise, reporter, dest_folder, redo=args.redo)

    print ('\nProducing report to', dest_folder)
    output_html_report(reporter, dest_folder)
    output_metrics_xml(reporter, dest_folder)

    end = time.time()
    print('\nChecks completed in {:.02f} s'.format((end - start)))
    return 0

    
if __name__ == '__main__':
    exit(main())
