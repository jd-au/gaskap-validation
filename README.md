# gaskap-validation
A validation suite for the GASKAP survey https://gaskap.anu.edu.au focused on metrics for high resolution spectral line data of the diffuse Interstellar Medium (ISM) of the Milky Way and the Magellanic system.

These tests are based on the work of https://github.com/Jordatious/ASKAP-continuum-validation/ with adaptation to the expected MW ISM environment with many resolved objects. Additional tests are also present for the large scale emission and for the spectral line data.

## Usage

    usage: gaskap-hi-validation.py [-h] [-c CUBE] [-i IMAGE] [-o OUTPUT]
                                [-e EMVEL] [-n NONEMVEL] [-N NOISE] [-r]

    Produce a validation report for GASKAP HI observations. Either a cube or an
    image (or both) must be supplied to be validated.

    optional arguments:
    -h, --help            show this help message and exit
    -c CUBE, --cube CUBE  The HI spectral line cube to be checked. (default:
                            None)
    -i IMAGE, --image IMAGE
                            The continuum image to be checked. (default: None)
    -o OUTPUT, --output OUTPUT
                            The folder in which to save the validation report and
                            associated figures. (default: report)
    -e EMVEL, --emvel EMVEL
                            The low velocity bound of the velocity region where
                            emission is expected. (default: None)
    -n NONEMVEL, --nonemvel NONEMVEL
                            The low velocity bound of the velocity region where
                            emission is not expected. (default: -100)
    -N NOISE, --noise NOISE
                            Use this fits image of the local rms. Default is to
                            run BANE (default: None)
    -r, --redo            Rerun all steps, even if intermediate files are
                            present. (default: False)

## Requirements

* python3
* Aegean
* Astropy
* Numpy
* RadioBeam
* SpectralCube

## Tests

### Spectral line tests

The following tests are run on GASKAP spectral line FITS cubes. The cubes can have either an LSRK velocity axis or a frequency axis.

#### Presence of Emission

We check that there is large scale emission present in velocity ranges where the GASS (McClure-Griffiths et al., 2009; Kalberla & Haud, 2015) survey detected emission for the field. 

We extract a ~40 km s<sup>-1</sup> slab from the cube at the velocity range where the strongest emission was detected in GASS. We produce a moment 0 map from this slab and pass this to BANE (Hancock, Trott, & Hurley-Walker, 2018) to produce a map of the large scale emission in the region. The large scale emission image excludes point sources and local noise. We then assess the maximum emission level (in Jy km s<sup>-1</sup> for the 40 km s s<sup>-1</sup> slab) as follows:

| Good        | Uncertain           | Bad  |
| -------------: |-------------:| -----:|
| > 1000  | 800 - 1000 | < 800 |

#### Absence of Off-line Emission

We check that large scale emission is not present in velocity ranges where the GASS (McClure-Griffiths et al., 2009; Kalberla & Haud, 2015) survey did not detect emission for the field. 

We extract a ~40 km s<sup>-1</sup> slab from the cube at a velocity range where little or no emission was detected in GASS. We produce a moment 0 map from this slab and pass this to BANE (Hancock, Trott, & Hurley-Walker, 2018) to produce a map of the large scale emission in the region. The large scale emission image excludes point sources and local noise. We then assess the maximum emission level (in Jy km s<sup>-1</sup> for the 40 km s s<sup>-1</sup> slab) as follows:

| Good        | Uncertain           | Bad  |
| -------------: |-------------:| -----:|
| < 200  | 200 - 500 | > 500 |

#### Spectral Noise

Using the spectral slab from the off-line emission test, we measure the standard deviation of each pixel to assess the noise in the spectral cube.

Taking the ~40 km s<sup>-1</sup> slab produced in the "Absence of Off-line Emission" test, we measure the standard deviation for the spectrum for each pixel. We scale this from the cube's native spectral resolution to a 5 kHz (~1 km s<sup-1></sup>) resolution. We then assess the median of this spectral noise level (in mJy) against the expected theoretical noise (sigma<sub>F</sub>) in 5 kHz channels for a 12 hour observation with ASKAP of 2.880 mJy as follows:

| Good        | Uncertain         | Bad  |
| -------------: |-------------:| -----:|
| < sqrt(2) sigma<sub>F</sub>  | sqrt(2) sigma<sub>F</sub> - < 2 sqrt(2) sigma<sub>F</sub> | > 2 sqrt(2) sigma<sub>F</sub> |