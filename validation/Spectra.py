import os

from astropy.io import ascii, fits
from astropy.io.votable import parse, from_table, writeto
from astropy.io.votable.tree import Info
from astropy.table import Table, Column
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def output_spectra_catalogue(filename, names, ras, decs, noises, noise_ratios):
    table = Table(meta={'name': filename, 'id': 'spectra'})
    table.add_column(Column(name='name', data=names))
    table.add_column(Column(name='RA', data=ras))
    table.add_column(Column(name='Dec', data=decs))
    table.add_column(Column(name='noise', data=noises, unit='Jy', description='Standard deviation of the flux in offline velocities'))
    table.add_column(Column(name='noise_ratio', data=noise_ratios))

    votable = from_table(table)
    writeto(votable, filename)


def get_beam_color(beam, cmap, min_ratio, max_ratio, theoretical_noise, spectra_folder, name):
    vel_range = (30,73)
    sbid = 8906
    filename = '{}/{}.vot'.format(spectra_folder, beam['name'])
    if os.path.exists(filename):
        beam_vot = parse(filename, pedantic=False)
        table = beam_vot.get_first_table()
        arr = table.array
        mask = (arr['Velocity'] <= vel_range[1]) & (arr['Velocity'] >= vel_range[0])
        sample = arr['Emission'][mask]
        noise = np.std(sample)*1000 # Convert to mJy
        ratio = noise/theoretical_noise
        idx = max(min((ratio-min_ratio)/(max_ratio-min_ratio),0.999),0)
        if ratio < 0.01:
            # no noise - likely no spectrum
            return 'grey', noise, ratio
        color = cmap(idx)
        print (name, ratio, idx, color)
        return color, noise, ratio
    print("Unable to find spectrum '{}'".format(filename))
    return 'white', 0, 0

def plot_beam_locs(cube, beams, theoretical_noise, name_prefix, spectra_folder):
    hdr = fits.getheader(cube)
    full_wcs = WCS(hdr)
    wcs = full_wcs.celestial
    #fig = plt.figure(figsize=(8, 24))
    fig = plt.figure(figsize=(16, 16))
    names = []
    ras = []
    decs = []
    noises = []
    noise_ratios = []

    cmap = mpl.cm.get_cmap('plasma', 64)
    min_ratio = 1
    max_ratio = 3
    norm = mpl.colors.Normalize(vmin=max_ratio, vmax=min_ratio)


    #im = ax.imshow(hdu.data, cmap='RdBu_r',vmax=vmax)
    #ax.invert_yaxis() 

    for idx, interleave in enumerate('ABC'):
        ax = fig.add_subplot(2,2,idx+1, projection=wcs)

        for beam in beams:
            field = beam['col1']
            #print (field, field[-1])
            if field[-1] == interleave:
                #print (beam['col2'], beam['name'], beam['pos'].ra.deg, beam['pos'].dec.deg )
                loc = beam['pos']
                names.append(beam['name'])
                ras.append(loc.fk5.ra.value)
                decs.append(loc.fk5.dec.value)

                beam_num = '{:02}'.format(beam['col2'])
                color, noise, noise_ratio = get_beam_color(beam, cmap, min_ratio, max_ratio, theoretical_noise, spectra_folder, beam['name'])
                ax.scatter([loc.fk5.ra.value], [loc.fk5.dec.value], transform=ax.get_transform('world'),
                    s=3000, #alpha=0.8,
                    edgecolor='gray', facecolor=color, lw=2, linestyle='-')
                ax.text(loc.fk5.ra.value, loc.fk5.dec.value-0.1, beam_num, transform=ax.get_transform('world'),
                    color='white', fontsize=18,
                    horizontalalignment='center', verticalalignment='bottom')
                noises.append(noise)
                noise_ratios.append(noise_ratio)


        ax.set_title("Interleave " + interleave)
        ax.set_xlabel("Right Ascension (degrees)", fontsize=16)
        ax.set_ylabel("Declination (degrees)", fontsize=16)
        ax.grid(color = 'gray', ls = 'dotted', lw = 2)
        
        #cbar = plt.colorbar(im, pad=.07)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    label='Noise level / theoretical noise', pad=.07)

    fig.savefig(name_prefix+'.png')
    fig.savefig(name_prefix+'_sml.png', dpi=8)
    output_spectra_catalogue(name_prefix+'.vot', names, ras, decs, noises, noise_ratios)


def get_spectral_units(ctype, cunit_key, hdr):
    spectral_conversion = 1
    if not cunit_key in hdr:
        if ctype.startswith('VEL') or ctype.startswith('VRAD'):
            spectral_unit = 'm/s'
        else:
            spectral_unit = 'Hz'
    else:
        spectral_unit = hdr[cunit_key]
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

def get_spectral_axis(header):
    for i in range(3,int(header['NAXIS'])+1):
        ctype = header['CTYPE'+str(i)]
        if (ctype.startswith('VEL') or ctype.startswith('VRAD') or ctype.startswith('FREQ')):
            return i
    return 0

def get_spec_resolution(header):
    spec_axis = get_spectral_axis(header)
    if not spec_axis:
        return None
    
    ctype = header['CTYPE'+str(spec_axis)]
    spectral_unit, spectral_conversion = get_spectral_units(ctype, 'CUNIT'+str(spec_axis), header)
    cdelt = header['CDELT'+str(spec_axis)]
    spec_res = cdelt / spectral_conversion
    
    spec_res_in_hz = 5000
    if spectral_unit == 'MHz':
        spec_res_in_hz = spec_res *1e6
    elif spectral_unit == 'km/s':
        # 0.5 kHz = 0.1 km/s
        spec_res_in_hz = spec_res *5 * 1000

    return abs(spec_res_in_hz)
