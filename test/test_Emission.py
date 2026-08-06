# Test cases for the Emission module.

# Author James Dempsey
# Date 31 Jul 2026

import inspect
import unittest

from validation import Emission

import astropy.units as u

class TestEmission(unittest.TestCase):

    def test_find_em_velocity_range_returns_expected_type(self):
        glat = 0.0
        glon = 0.0
        result = Emission.find_em_velocity_range(glat, glon)
        self.assertIsInstance(result, u.Quantity)

    def test_find_em_velocity_range_returns_default_value(self):
        glat = 30.0
        glon = 0.0
        result = Emission.find_em_velocity_range(glat, glon)
        self.assertEqual(result, 0 * u.km / u.s)

    def test_find_em_velocity_range_returns_specific_value(self):
        glat = 5.0
        glon = 290.0
        result = Emission.find_em_velocity_range(glat, glon)
        self.assertEqual(result, -45 * u.km / u.s)

    def test_find_em_velocity_range_returns_specific_value_min_glon(self):
        glat = 5.0
        glon = 280.0
        result = Emission.find_em_velocity_range(glat, glon)
        self.assertEqual(result, -45 * u.km / u.s)

    def test_find_em_velocity_range_returns_specific_value_min_glat(self):
        glat = -25.0
        glon = 229.0
        result = Emission.find_em_velocity_range(glat, glon)
        self.assertEqual(result, -20 * u.km / u.s)

    def test_find_non_em_velocity_range_returns_lower_bound(self):
        minvel = -200 * u.km / u.s
        maxvel = 200 * u.km / u.s
        em_min_vel = 0 * u.km / u.s
        result = Emission.find_non_em_velocity_range(minvel, maxvel, em_min_vel)
        self.assertEqual(result, -180 * u.km / u.s)

    def test_find_non_em_velocity_range_returns_upper_bound(self):
        minvel = -200 * u.km / u.s
        maxvel = 300 * u.km / u.s
        em_min_vel = -45 * u.km / u.s
        result = Emission.find_non_em_velocity_range(minvel, maxvel, em_min_vel)
        self.assertEqual(result, 190 * u.km / u.s)

    def test_find_non_em_velocity_range_returns_upper_bound_custom_vel_width(self):
        minvel = -200 * u.km / u.s
        maxvel = 300 * u.km / u.s
        em_min_vel = -45 * u.km / u.s
        result = Emission.find_non_em_velocity_range(minvel, maxvel, em_min_vel, vel_width=100*u.km/u.s)
        self.assertEqual(result, 180 * u.km / u.s)

    def test_find_non_em_velocity_range_returns_upper_bound_metre_cube(self):
        minvel = -200*1000 * u.m / u.s
        maxvel = 300*1000 * u.m / u.s
        em_min_vel = -45 * u.km / u.s
        result = Emission.find_non_em_velocity_range(minvel, maxvel, em_min_vel)
        self.assertEqual(result, 190 * u.km / u.s)

if __name__ == "__main__":
    unittest.main()
