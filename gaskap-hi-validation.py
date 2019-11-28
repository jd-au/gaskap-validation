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
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from radio_beam import Beam
from spectral_cube import SpectralCube

from validation_reporter import ValidationReport, ReportSection, ReportItem, ValidationMetric, output_html_report, output_metrics_xml

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
    parser = argparse.ArgumentParser(
        description="Produce a validation report for GASKAP HI observatons")

    parser.add_argument("-c", "--cube", required=True, help="The HI spectral line cube to be checked.")
    parser.add_argument("-o", "--output", help="The folder in which to save the validation report and associated figures.", default='report')

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


def plot_difference_map(hdu, file_prefix, vmax):
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
    prefix = os.path.splitext(example_name)[0]
    fname = prefix + suffix
    return fname

def get_figures_folder(dest_folder):
    return dest_folder + '/' + figures_folder + '/'

def get_bane_background(infile, outfile_prefix, ncores=8, redo=False):
    background_prefix = outfile_prefix+'_bkg'
    if redo or not os.path.exists(background_prefix + '.fits'):
        cmd = "BANE --cores={0} --out={1} {2}".format(ncores, outfile_prefix, infile)
        print (cmd)
        os.system(cmd)
    
    plot_map(background_prefix)
    plot_histogram(background_prefix, 'Emission (Jy beam^{-1} km s^{-1})')


def assess_metric(metric, threshold1, threshold2, low_good=False):
    if metric < threshold1:
        return METRIC_GOOD if low_good else METRIC_BAD
    elif metric < threshold2:
        return METRIC_UNCERTAIN
    else:
        return METRIC_BAD if low_good else METRIC_GOOD


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
    plot_difference_map(hi_data[0], folder+prefix, vmax=3e3)

    # Produce the background plots
    bkg_data = get_bane_background(mom0_fname, folder+prefix, ncores=ncores, redo=redo)
    hi_data = fits.open(folder + prefix+'_bkg.fits')
    max_em = np.nanmax(hi_data[0].data)

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Presence of emission', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Large Scale<br/>Emission Map', link='figures/'+prefix+'_bkg.png', image='figures/'+prefix+'_bkg_sml.png')
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
    plot_difference_map(hi_data[0], folder+prefix, vmax=3e3)

    # Produce the background plots
    bkg_data = get_bane_background(mom0_fname, folder+prefix, ncores=ncores, redo=redo)
    hi_data = fits.open(folder+prefix+'_bkg.fits')
    max_em = np.nanmax(hi_data[0].data)

    # assess
    cube_name = os.path.basename(cube)
    section = ReportSection('Absence of Off-line Emission', cube_name)
    section.add_item('Velocity Range<br/>(km/s LSR)', value='{:.0f} to {:.0f}'.format(vel_start.value, vel_end.value))
    section.add_item('Large Scale<br/>Emission Map', link='figures/'+prefix+'_bkg.png', image='figures/'+prefix+'_bkg_sml.png')
    section.add_item('Emission Histogram', link='figures/'+prefix+'_bkg_hist.png', image='figures/'+prefix+'_bkg_hist_sml.png')
    section.add_item('Max Emission<br/>(Jy km s<sup>-1</sup> beam<sup>-1</sup>)', value='{:.3f}'.format(max_em))
    reporter.add_section(section)

    metric = ValidationMetric('Absence of Off-line Emission', 
        'Maximum large scale emission intensity in the velocity range where emission is not expected.',
        max_em, assess_metric(max_em, 200, 500, low_good=True))
    reporter.add_metric(metric)
    return slab


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

    theoretical_gaskap_noise = 2.0 # mJy per 5 kHz for 12.5 hr observation (Dickey et al 2013 table 3)
    metric = ValidationMetric('Spectral Noise', 
        '1-sigma spectral noise level per 5 kHz channel.',
        median_noise_5kHz, assess_metric(median_noise_5kHz, 
        theoretical_gaskap_noise* np.sqrt(2), theoretical_gaskap_noise* np.sqrt(2)*2, low_good=True))
    reporter.add_metric(metric)

    return


def main():
    # Parse command line options
    args = parseargs()
    dest_folder = args.output
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    figures_folder = dest_folder + '/figures'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    if not os.path.exists(args.cube) or not os.path.isfile(args.cube):
        raise ValueError('Cube {} could not be found or is not a file.'.format(args.cube))

    start = time.time()

    print ('\nChecking quality level of GASKAP HI cube:', args.cube)

    cube_name = os.path.basename(args.cube)
    reporter = ValidationReport('GASKAP Validation Report: {}'.format(cube_name))

    check_for_emission(args.cube, emission_vel_range[0], emission_vel_range[1], reporter, dest_folder, redo=args.redo)

    slab = check_for_non_emission(args.cube, non_emission_val_range[0], non_emission_val_range[1], reporter, dest_folder, redo=args.redo)

    measure_spectral_line_noise(slab, args.cube, non_emission_val_range[0], non_emission_val_range[1], reporter, dest_folder, redo=args.redo)

    print ('\nProducing report to', dest_folder)
    output_html_report(reporter, dest_folder)
    output_metrics_xml(reporter, dest_folder)

    end = time.time()
    print('\nChecks completed in {:.02f} s'.format((end - start)))
    return 0

    
if __name__ == '__main__':
    exit(main())
