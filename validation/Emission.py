# Library of routines for assessing emission in a slab extracted from an image cube.

# Author James Dempsey
# Date 31 Jul 2026

import os
from pathlib import Path

import astropy.units as u
from astropy.table import Table


def _read_velocity_ranges():
    """
    Read the emission velocity ranges from the em_velocities.csv file.
    """
    lib_src_path = Path(os.path.realpath(__file__)).parent
    ref_dir_path = lib_src_path.parent / 'reference'
    em_velocities = Table.read(os.path.join(ref_dir_path, 'em_velocities.csv'), format='ascii.csv', 
                               names=('glat_min', 'glat_max', 'glon_min', 'glon_max', 'em_min_vel'))
    return em_velocities


def find_em_velocity_range(glat: float, glon: float) -> u.Quantity:
    """
    Find the expected emission velocity range for a given Galactic latitude and longitude.
    """
    em_velocities = _read_velocity_ranges()
    for row in em_velocities:
        if (row['glat_min'] <= glat < row['glat_max']) and (row['glon_min'] <= glon < row['glon_max']):
            return row['em_min_vel'] * u.km / u.s
    return 0 * u.km / u.s


def find_non_em_velocity_range(min_cube_vel: u.Quantity, max_cube_vel: u.Quantity, em_min_vel: u.Quantity, 
                               vel_width=90*u.km/u.s) -> u.Quantity:
    """
    Find the velocity range of a non emission velocity window that is outside the emission velocity range.
    """
    # find the end furthest from the velocity window
    if abs(min_cube_vel - em_min_vel) > abs(max_cube_vel - (em_min_vel + vel_width)):
        # the min_cube_vel is further away from the emission window, so use that as the start of the non-emission window
        non_em_min_vel = min_cube_vel + 20 * u.km/u.s
    else:
        # the max_cube_vel is further away from the emission window, so use that as the start of the non-emission window
        non_em_min_vel = max_cube_vel -  20 * u.km/u.s - vel_width
    return non_em_min_vel