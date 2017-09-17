import os
import numpy as np
import pickle as pickle
from astropy.io import fits
from sncosmo import Model, get_source, CCM89Dust
from astropy.cosmology import Planck15 as cosmo

"""
A bunch of handy utilities for working with the Nearby Supernova Factory
Internal Data Release.
"""

IDR_dir = '/Users/samdixon/repos/IDRTools/ALLEG2a_SNeIa'
META = os.path.join(IDR_dir, 'META.pkl')

C = 2.99792458e10
PLANCK = 6.62607e-27

meta = pickle.load(open(META, 'rb'))


def lambda_bin(wmin, wmax, velocity):
    """Returns velocity bins"""
    n = int(round(np.log10(float(wmax)/wmin) / np.log10(1 + velocity/3.0e5) + 1))
    binedges = np.logspace(np.log10(wmin), np.log10(wmax), n)
    bincenters = (binedges[0:-1] + binedges[1:])/2  # not log centers
    wave = np.around(bincenters)
    return [wave, binedges]


def recover_bin_edges(bin_centers):
    """Convert a list of bin centers to bin edges.
    We do a second order correction to try to get this as accurately as
    possible.
    For linear binning there is only machine precision error with either first
    or second order binning.
    For higher order binnings (eg: log), the fractional error is of order (dA /
    A)**2 for linear recovery and (dA / A)**4 for the second order recovery
    that we do here.
    """
    # First order
    o1 = (bin_centers[:-1] + bin_centers[1:]) / 2.

    # Second order correction
    o2 = 1.5*o1[1:-1] - (o1[2:] + o1[:-2]) / 4.

    # Estimate front and back edges
    f2 = 2*bin_centers[1] - o2[0]
    f1 = 2*bin_centers[0] - f2
    b2 = 2*bin_centers[-2] - o2[-1]
    b1 = 2*bin_centers[-1] - b2

    # Stack everything together
    bin_edges = np.hstack([f1, f2, o2, b2, b1])
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]

    return bin_starts, bin_ends


def rebin(old_bin_centers, flux, var):
    old_bin_starts, old_bin_ends = recover_bin_edges(old_bin_centers)
    new_bin_centers, new_bin_edges = lambda_bin(3300, 8600, 1000)
    new_bin_starts = new_bin_edges[:-1]
    new_bin_ends = new_bin_edges[1:]

    num_new_bins = len(new_bin_starts)
    num_old_bins = len(old_bin_starts)

    weights = np.ones(num_old_bins, dtype=float)

    new_flux_sum = np.zeros(num_new_bins)
    new_fluxvar_sum = np.zeros(num_new_bins)
    new_weights = np.zeros(num_new_bins)

    new_index = 0
    for old_index in range(num_old_bins):
        # Find index of start of old array in new array
        old_start = old_bin_starts[old_index]
        old_end = old_bin_ends[old_index]

        while True:
            if old_start < new_bin_starts[new_index]:
                if new_index == 0:
                    break
                new_index -= 1
                continue

            if old_start > new_bin_ends[new_index]:
                if new_index == num_new_bins - 1:
                    break
                new_index += 1
                continue
            break

        if old_start > new_bin_ends[new_index]:
            continue

        # Split the old bin's data between the new bins.
        while new_bin_starts[new_index] < old_end:
            # Figure out which fraction of the bin we have from the
            # interpolation.
            overlap_start = max(old_start, new_bin_starts[new_index])
            overlap_end = min(old_end, new_bin_ends[new_index])
            overlap = overlap_end - overlap_start

            weight = (
                weights[old_index] *
                overlap / (old_end - old_start)
            )

            new_weights[new_index] += weight
            new_flux_sum[new_index] += weight * flux[old_index]
            new_fluxvar_sum[new_index] += (
                weight**2 * var[old_index]
            )

            if new_index == num_new_bins - 1:
                break

            new_index += 1

        # We almost always go 1 past here, so jump back one to get the
        # search to start in (usually) the right place.
        if new_index > 1:
            new_index -= 1

    mask = new_weights == 0

    new_weights[mask] = 1.
    new_flux = new_flux_sum / new_weights
    new_fluxvar = new_fluxvar_sum / new_weights**2

    bin_widths = new_bin_ends - new_bin_starts
    new_flux *= bin_widths
    new_fluxvar *= (bin_widths * bin_widths)

    new_flux[mask] = np.nan
    new_fluxvar[mask] = np.nan

    return new_bin_centers, new_flux, new_fluxvar


def add_noise(wave, flux, s2n):
    """
    Add Gaussian noise with the given signal-to-noise characteristic
    """
    sigma = flux/s2n
    noise = np.random.randn(len(flux)) * sigma
    noised_flux = flux + noise
    noised_var = sigma**2
    return wave, noised_flux, noised_var


class Dataset(object):

    def __init__(self, data=meta, subset='training'):
        self.data = {}
        if subset is not None:
            self.subset = ''.join(subset)
            for k, v in data.items():
                k = k.replace('.', '_')
                k = k.replace('-', '_')
                if v['idr.subset'] in subset:
                    self.data[k] = v
        else:
            self.data = data
        self.sne_names = list(self.data.keys())
        self.sne = [Supernova(self.data, name) for name in self.sne_names]
        for k in self.data.keys():
            setattr(self, k, Supernova(self.data, k))

    def random_sn(self):
        """
        Returns a random supernova
        """
        return np.random.choice(self.sne, 1)[0]


class Supernova(object):

    def __init__(self, dataset, name):
        data = dataset[name]
        for k, v in data.items():
            k = k.replace('.', '_')
            setattr(self, k, v)
        setattr(self, 'hr', self.get_hr()[0])
        setattr(self, 'hr_err', self.get_hr()[1])
        setattr(self, 'distmod', self.get_distmod()[0])
        setattr(self, 'distmod_err', self.get_distmod()[1])
        self.spectra = [Spectrum(dataset, name, obs) for obs in self.spectra.keys()]
        # Sort spectra by SALT2 phase
        self.spectra = sorted(self.spectra, key=lambda x: x.salt2_phase)

    def spec_nearest_max(self, phase=0):
        """
        Returns the spectrum object for the observation closest to B-band max.
        """
        min_phase = min(np.abs(s.salt2_phase-phase) for s in self.spectra)
        return [s for s in self.spectra if np.abs(s.salt2_phase-phase) == min_phase][0]

    def lc(self, filter_name):
        """
        Finds the light curve in some SNf filter using synthetic photometry.
        """
        phase = [spec.salt2_phase for spec in self.spectra]
        mag = [spec.snf_magnitude(filter_name) for spec in self.spectra]
        return phase, mag

    def salt2_model_fluxes(self):
        """
        Creates the SALT2 model spectra flux based on the fit parameters.
        """
        source = get_source('SALT2', version='2.4')
        dust = CCM89Dust()
        model = Model(source=source, effects=[dust],
                      effect_names=['mw'], effect_frames=['obs'])
        model.set(z=0, t0=0, x0=self.salt2_X0, x1=self.salt2_X1,
                  c=self.salt2_Color, mwebv=self.salt2_target_mwebv)
        wave = np.arange(3272, 9200, 2)
        measured_phases = [spec.salt2_phase for spec in self.spectra]
        phases = np.linspace(min(measured_phases), max(measured_phases), 100)
        fluxes = model.flux(phases, wave)
        return phases, wave, fluxes

    def salt2_model_lc(self, filter_name):
        """
        Creates the SALT2 model light curve based on the fit parameters.
        """
        phases, wave, fluxes = self.salt2_model_fluxes()
        filter_edges = {'u': (3300., 4102.),
                        'b': (4102., 5100.),
                        'v': (5200., 6289.),
                        'r': (6289., 7607.),
                        'i': (7607., 9200.)}
        min_wave, max_wave = filter_edges[filter_name]
        mag = []
        for flux in fluxes:
            ref_flux = 3.631e-20 * C * 1e8 / wave**2
            flux_sum = np.sum((flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
            ref_flux_sum = np.sum((ref_flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
            mag.append(-2.5*np.log10(flux_sum/ref_flux_sum))
        return phases, mag

    def get_distmod(self):
        """
        Return the distance modulus from the SALT2 parameters.
        """
        MB, alpha, beta = -19.155510156376913, 0.15336666476334873, 2.7111339334687163  # Obtained from emcee fit (see Brian's code)
        dMB, dalpha, dbeta = 0.019457765851807848, 0.020340953530227517, 0.13066032343415704
        mu = self.salt2_RestFrameMag_0_B - MB + alpha * self.salt2_X1 - beta * self.salt2_Color
        dmu = np.sqrt(self.salt2_RestFrameMag_0_B_err**2+dMB**2+dalpha**2*self.salt2_X1**2+alpha**2*self.salt2_X1_err**2+beta**2*self.salt2_Color_err**2+dbeta**2+self.salt2_Color**2)
        return mu, dmu

    def get_hr(self):
        """
        Return the Hubble residual from the SALT2 parameters.
        """
        mu, dmu = self.get_distmod()
        cosmo_mu = cosmo.distmod(self.salt2_Redshift).value
        return mu-cosmo_mu, dmu


class Spectrum(object):
    def __init__(self, dataset, name, obs):
        self.sn_data = dataset[name]
        data = dataset[name]['spectra'][obs]
        for k, v in data.items():
            k = k.replace('.', '_')
            setattr(self, k, v)

    def merged_spec(self):
        """
        Returns the merged spectrum from the IDR FITS files.
        """
        path = os.path.join(IDR_dir, self.idr_spec_merged)
        f = fits.open(path)
        head = f[0].header
        flux = f[0].data
        var = f[1].data
        f.close()
        start = head['CRVAL1']
        end = head['CRVAL1']+head['CDELT1']*len(flux)
        npts = len(flux)+1
        wave = np.linspace(start, end, npts)[:-1]
        return wave, flux, var

    def rf_spec(self, bin_edges=None, s2n=None, renorm=True):
        """
        Returns the restframe spectrum from the IDR FITS files.
        """
        path = os.path.join(IDR_dir, self.idr_spec_restframe)
        f = fits.open(path)
        head = f[0].header
        flux = f[0].data
        var = f[1].data
        f.close()
        start = head['CRVAL1']
        end = head['CRVAL1']+head['CDELT1']*len(flux)
        npts = len(flux)+1
        wave = np.linspace(start, end, npts)[:-1]
        if renorm:
            #Flux is scaled by a relative distance factor to z=0.05 and multiplied by 1e15
            dl = (1 + self.sn_data['host.zhelio']) * cosmo.comoving_transverse_distance(self.sn_data['host.zcmb']).value
            dlref = cosmo.luminosity_distance(0.05).value
            flux = flux / ((1+self.sn_data['host.zhelio'])/(1+0.05) * (dl/dlref)**2 * 1e15)
            var = var / ((1+self.sn_data['host.zhelio'])/(1+0.05) * (dl/dlref)**2 * 1e15)**2
        if bin_edges is not None:
            wave, flux, var = self.rebin(wave, flux, var, bin_edges)
        if s2n is not None:
            wave, flux, var = self.add_noise(wave, flux, var, s2n)
        return wave, flux, var

    def salt2_model_fluxes(self):
        """
        Creates the SALT2 model spectra flux based on the fit parameters.
        """
        source = get_source('SALT2', version='2.4')
        model = Model(source=source)
        model.set(z=0, t0=0, x0=self.sn_data['salt2.X0'], x1=self.sn_data['salt2.X1'], c=self.sn_data['salt2.Color'])
        wave = np.arange(3272, 9200, 2)
        flux = model.flux(self.salt2_phase, wave)
        return wave, flux

    def magnitude(self, min_wave, max_wave):
        """
        Calculates the AB magnitude in a given top-hat filter.
        """
        wave, flux, var = self.rf_spec()
        ref_flux = 3.631e-20 * C * 1e8 / wave**2
        flux_sum = np.sum((flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
        ref_flux_sum = np.sum((ref_flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
        return -2.5*np.log10(flux_sum/ref_flux_sum)

    def snf_magnitude(self, filter_name, z=None):
        """
        Calculates the AB magnitude in a given SNf filter.
        """
        filter_edges = {'u': (3300., 4102.),
                        'b': (4102., 5100.),
                        'v': (5200., 6289.),
                        'r': (6289., 7607.),
                        'i': (7607., 9200.)}
        min_wave, max_wave = filter_edges[filter_name]
        return self.magnitude(min_wave, max_wave)

    def wfirst_s2n(self, z=1, noise=True):
        wfw, wfsn = np.loadtxt('/Users/samdixon/repos/IDRTools/data/wfirst_z{:0.2f}.txt'.format(z),
                               unpack=True, usecols=(0, 3))
        wave, flux, var = self.rf_spec()
        rebinned, flux = self.rebin(wave, flux, wfw)
        if noise:
            rebinned, flux, var = self.add_noise(rebinned, flux, wfsn)
        outwave = rebinned[(rebinned > min(wave)) & (rebinned < max(wave))]
        flux = flux[(rebinned > min(wave)) & (rebinned < max(wave))]
        var = var[(rebinned > min(wave)) & (rebinned < max(wave))]
        return outwave, flux, var


if __name__ == '__main__':
    d = Dataset()
    sn = d.PTF09dnl
    spec = sn.spec_nearest_max()
    wave, flux, var = spec.rf_spec()
    wfwave, wfflux, wfvar = spec.wfirst_s2n()
    import matplotlib.pyplot as plt
    plt.errorbar(wfwave, wfflux, yerr=np.sqrt(wfvar), alpha=0.3)
    plt.show()
