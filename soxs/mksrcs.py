#!/home/srandall/soft/anaconda3/bin/python

# Draw point sources from a logN-logS distribution and add them to an
# input image file.  Flux units are 10^-14 erg/cm^2/s.  Output image is
# in cts/s.
# NOTE: numerical integration and root finding could be done analytically
# to speed up the code at the expense of generality for the form of dNdS.
# Should eventually add an option to choose.
# Author: Scott Randall (srandall@cfa.harvard.edu)

import sys
import argparse
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as optimize
import random
from astropy.utils.console import ProgressBar
from soxs import write_photon_list, instrument_simulator
from soxs.constants import keV_per_erg, erg_per_keV
from soxs.spectra import get_wabs_absorb
from soxs.utils import mylog

class Bgsrc:
    def __init__(self, src_type, flux, z, ind):
        self.src_type = src_type
        self.flux = flux
        self.z = z
        self.ind = ind
        self.scale_factor = 1.0/(1.+z)

# dN/dS, takes S in 1e-14 erg/cm^2/s
# returns 10^14 deg^-2  (erg/cm^2/s)^-1
def dNdS(S, src_type, band):
    if not (band == 'fb'):
        sys.exit('Energy band not supported')

    # lower limit on source flux so as not to overpredict unresolved CXRB.
    # Note that this number depends on the energy band
    S_cut = 9.8e-7

    if S < S_cut:
        return 0.0

    # From Lehmer et al. 2012
    # Change flux units to 1e-14 erg/cm^2/s
    if src_type == 'agn':
        K = 562.2  # 1e14 deg^-2 (erg/cm^2/s)^-1
        beta1 = 1.34
        beta2 = 2.35
        f_break = 0.81
    elif src_type == 'gal':
        K = 2.82
        beta1 = 2.4
    elif src_type == 'star':
        K = 4.07
        beta1 = 1.55
    else:
        sys.exit('Source type not supported')

    # 10^-14 erg cm^-2 s^-1
    S_ref = 1

    if src_type == 'agn' and S > f_break:
        dnds = K*(f_break/S_ref)**(beta2 - beta1)*(S/S_ref)**(-1*beta2)
    else:
        dnds = K*(S/S_ref)**(-1*beta1)

    return dnds

# integral of dN/dS, takes S in 1e-14 erg/cm^2/s
# returns 10^14 deg^-2
def int_dNdS(S_lo, S_hi, src_type, band):
    if not (band == 'fb'):
        sys.exit('Energy band not supported')

    # lower limit on source flux so as not to overpredict unresolved CXRB.
    # Note that this number depends on the energy band
    S_cut = 9.8e-7

    if S_lo < S_cut:
        S_lo = S_cut
    if S_hi < S_cut:
        return 0.0

    # From Lehmer et al. 2012
    # Change flux units to 1e-14 erg/cm^2/s
    if src_type == 'agn':
        K = 562.2  # 1e14 deg^-2 (erg/cm^2/s)^-1
        beta1 = 1.34
        beta2 = 2.35
        f_break = 0.81
    elif src_type == 'gal':
        K = 2.82
        beta1 = 2.4
    elif src_type == 'star':
        K = 4.07
        beta1 = 1.55
    else:
        sys.exit('Source type not supported')

    # 10^-14 erg cm^-2 s^-1
    S_ref = 1

    if src_type == 'agn':
        if S_hi <= f_break:
            int_dnds = K*S_ref**beta1/(1-beta1)*(S_hi**(1-beta1) - S_lo**(1-beta1))
        else:
            int_dnds = K*S_ref**beta1/(1-beta1)*(f_break**(1-beta1) - S_lo**(1-beta1))
            int_dnds += K*(f_break/S_ref)**(beta2-beta1)*S_ref**beta2/(1-beta2) \
                * (S_hi**(1-beta2) - f_break**(1-beta2))
    else:
        int_dnds = K*S_ref**beta1/(1-beta1)*(S_hi**(1-beta1) - S_lo**(1-beta1))

    return int_dnds

def dNdS_draw(S_draw, rand, norm, src_type, band):
    return int_dNdS(S_draw, np.inf, src_type, band)/norm - rand
    # return ((integrate.quad(dNdS, S_draw, np.inf, args=(type,band))[0])/norm - rand)

def plaw_cdf(n_ph, emin, emax, alpha, prng=np.random):
    u = prng.uniform(size=n_ph)
    if alpha == 1.0:
        e = emin*(emax/emin)**u
    else:
        oma = 1.0-alpha
        invoma = 1.0/oma
        e = emin**oma + u*(emax**oma-emin**oma)
        e **= invoma
    return e

def get_flux_scale(ind, fb_emin, fb_emax, spec_emin, spec_emax):
    if ind == 1.0:
        f_g = np.log(spec_emax/spec_emin)
    else:
        f_g = (spec_emax**(1.0-ind)-spec_emin**(1.0-ind))/(1.0-ind)
    if ind == 2.0:
        f_E = np.log(fb_emax/fb_emin)
    else:
        f_E = (fb_emax**(2.0-ind)-fb_emin**(2.0-ind))/(2.0-ind)
    fscale = f_g/f_E
    return fscale

def main():
    t_exp = 10 # exposure time, ksec
    eff_area = 40000   # effective area, cm^2
    eph_mean = 1   # mean photon energy, keV
    fov = 20    # edge of FOV, arcmin
    sources = []
    draw_srcs = True
    src_types = ['agn', 'gal', 'star']

    # parameters for making event file
    evt_prefix = "10ks_fast" # event file prefix
    ra_cen = 96.6 # degress, RA of field center
    dec_cen = -53.73 #degrees, Dec of field center
    nH = 0.05 # Galactic absorption, 1e22 cm^-2

    fb_emin = 0.5  # keV, low energy bound for full band flux
    fb_emax = 8.0  # keV, high energy bound for full band flux
    spec_emin = 0.1 # keV, minimum energy of mock spectrum
    spec_emax = 10.0 # keV, max energy of mock spectrum

    agn_ind = 1.2 # AGN photon index
    agn_z = 2.0 # AGN redshift
    gal_ind = 1.2 # galaxy photon index
    gal_z = 0.8 # galaxy redshift
    star_ind = 1.0 # star photon index

    dither_size = 16.0 # dither circle radius or box width in arcsec
    dither_shape = 'square'
    test = False
    prng = np.random

    redshifts = {"agn": agn_z, "gal": gal_z, "star": 0.0}
    indices = {"agn": agn_ind, "gal": gal_ind, "star": star_ind}
    fluxscale = {}
    for src_type in src_types:
        fluxscale[src_type] = get_flux_scale(indices[src_type], fb_emin, fb_emax,
                                             spec_emin, spec_emax)

    eph_mean_erg = eph_mean*erg_per_keV

    # integrate down to a flux where we expect to have roughly 0.1 photons
    # during the exposure
    S_min = 0.1*eph_mean_erg/(t_exp*1000*eff_area)
    S_min = S_min/1e-14
    # S_min = 9.8e-7
    mylog.info("The flux limit is %g." % (S_min*1e-14))
    fov_area = fov**2

    if test:
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('agn','fb'))[0]
        print(n_srcs)
        n_srcs = int_dNdS(S_min, np.inf, 'agn', 'fb')
        print(n_srcs)
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('gal','fb'))[0]
        print(n_srcs)
        n_srcs = int_dNdS(S_min, np.inf, 'gal', 'fb')
        print(n_srcs)
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('star','fb'))[0]
        print(n_srcs)
        n_srcs = int_dNdS(S_min, np.inf, 'star', 'fb')
        print(n_srcs)
        dnds = dNdS(1.0, 'agn', 'fb')
        print(dnds)

    # Calculate the number of sources with S>S_min in the FOV
    for src_type in src_types:
        # dNdS returns 10^14 deg^-2 (erg/cm^2/s)^-1, but we get a factor of
        # 10^-14 from dS in integral, so they cancel
        # n_srcs = integrate.quad(dNdS, S_min, np.inf, args=(type,'fb'))[0]
        n_srcs = int_dNdS(S_min, np.inf, src_type, 'fb')
        # scale to the FOV
        n_srcs_fov = n_srcs*fov_area/60**2
        mylog.info("Expect %d sources of type \"%s\" in the field." % (n_srcs_fov, src_type))
        # draw a random distribution of sources
        if draw_srcs:
            mylog.info("Drawing sources from distribution \"%s\"." % src_type)
            for i in range(0,int(round(n_srcs_fov,0))):
                rand = random.random()
                S = optimize.brentq(dNdS_draw, S_min, 1000, args=(rand, n_srcs, src_type, 'fb'))
                thissrc = Bgsrc(src_type, S*1e-14, redshifts[src_type], indices[src_type])
                sources.append(thissrc)

    num_sources = len(sources)

    if test:
        ngt17 = 0
        ngt16 = 0
        ngt15 = 0
        ngt14 = 0
        for source in sources:
            if source.src_type == 'agn':
                if source.flux > 1.0e-17:
                    ngt17 = ngt17+1
                if source.flux > 1.0e-16:
                    ngt16 = ngt16+1
                if source.flux > 1.0e-15:
                    ngt15 = ngt15+1
                if source.flux > 1.0e-14:
                    ngt14 = ngt14+1
        print("N_AGN > 1e-17 erg/cm^2/s (deg^-2):",ngt17*60**2/fov_area)
        print("N_AGN > 1e-16 erg/cm^2/s (deg^-2):",ngt16*60**2/fov_area)
        print("N_AGN > 1e-15 erg/cm^2/s (deg^-2):",ngt15*60**2/fov_area)
        print("N_AGN > 1e-14 erg/cm^2/s (deg^-2):",ngt14*60**2/fov_area)
        ngt17 = 0
        ngt16 = 0
        ngt15 = 0
        ngt14 = 0
        for source in sources:
            if source.src_type == 'gal':
                if source.flux > 1.0e-17:
                    ngt17 = ngt17+1
                if source.flux > 1.0e-16:
                    ngt16 = ngt16+1
                if source.flux > 1.0e-15:
                    ngt15 = ngt15+1
                if source.flux > 1.0e-14:
                    ngt14 = ngt14+1
        print("N_GAL > 1e-17 erg/cm^2/s (deg^-2):",ngt17*60**2/fov_area)
        print("N_GAL > 1e-16 erg/cm^2/s (deg^-2):",ngt16*60**2/fov_area)
        print("N_GAL > 1e-15 erg/cm^2/s (deg^-2):",ngt15*60**2/fov_area)
        print("N_GAL > 1e-14 erg/cm^2/s (deg^-2):",ngt14*60**2/fov_area)
        ngt17 = 0
        ngt16 = 0
        ngt15 = 0
        ngt14 = 0
        for source in sources:
            if source.src_type == 'star':
                if source.flux > 1.0e-17:
                    ngt17 = ngt17+1
                if source.flux > 1.0e-16:
                    ngt16 = ngt16+1
                if source.flux > 1.0e-15:
                    ngt15 = ngt15+1
                if source.flux > 1.0e-14:
                    ngt14 = ngt14+1
        print("N_STAR > 1e-17 erg/cm^2/s (deg^-2):",ngt17*60**2/fov_area)
        print("N_STAR > 1e-16 erg/cm^2/s (deg^-2):",ngt16*60**2/fov_area)
        print("N_STAR > 1e-15 erg/cm^2/s (deg^-2):",ngt15*60**2/fov_area)
        print("N_STAR > 1e-14 erg/cm^2/s (deg^-2):",ngt14*60**2/fov_area)

        sys.exit('done testing')

    mylog.info("Generating spectra from %d sources." % len(sources))
    dec_scal = np.fabs(np.cos(dec_cen*np.pi/180))
    ra_min = ra_cen - fov/(2*60*dec_scal)
    dec_min = dec_cen - fov/(2*60)
    all_energies = []
    all_ra = []
    all_dec = []

    with ProgressBar(num_sources) as pbar:
        for source in sources:

            # Using the energy flux, determine the photon flux by simple scaling
            ref_ph_flux = source.flux*fluxscale[source.src_type]*keV_per_erg
            # Now determine the number of photons we will generate
            n_ph = np.modf(ref_ph_flux*t_exp*1000.0*eff_area)
            n_ph = np.int64(n_ph[1]) + np.int64(n_ph[0] >= prng.uniform())

            if n_ph > 0:
                # Generate the energies in the source frame
                energies = plaw_cdf(n_ph, spec_emin, spec_emax, source.ind, prng=prng)
                # NOTE: Here is where we could put in intrinsic absorption if we wanted.
                # Local galactic absorption is done at the end.
                # Assign positions for this source
                ra = prng.random(size=n_ph)*fov/(60.0*dec_scal) + ra_min
                dec = prng.random(size=n_ph)*fov/60.0 + dec_min

                all_energies.append(energies)
                all_ra.append(ra)
                all_dec.append(dec)

            pbar.update()

    all_energies = np.concatenate(all_energies)
    all_ra = np.concatenate(all_ra)
    all_dec = np.concatenate(all_dec)

    all_nph = all_energies.size

    mylog.info("Generated %d photons from point sources." % all_nph)

    # Remove some of the photons due to Galactic foreground absorption.
    # We throw a lot of stuff away, but this is more general and still
    # faster. 
    absorb = get_wabs_absorb(all_energies, nH)
    randvec = prng.uniform(size=all_energies.size)
    all_energies = all_energies[randvec < absorb]
    all_ra = all_ra[randvec < absorb]
    all_dec = all_dec[randvec < absorb]

    all_nph = all_energies.size

    mylog.info("%d photons remain after foreground galactic absorption." % all_nph)

    all_flux = np.sum(all_energies)*erg_per_keV/(t_exp*1000*eff_area)

    write_photon_list(evt_prefix, evt_prefix, all_flux, all_ra, all_dec, all_energies, clobber=True)
    simput_file = evt_prefix + "_simput.fits"
    evt_file = evt_prefix + "_evt.fits"
    mylog.info("Running instrument simulator...")
    instrument_simulator(simput_file, evt_file,  t_exp*1000, 'hdxi', [ra_cen,dec_cen],
                         dither_size=dither_size, dither_shape=dither_shape, 
                         astro_bkgnd=None, instr_bkgnd_scale=0, clobber=True)

if __name__ == "__main__":
    main()
