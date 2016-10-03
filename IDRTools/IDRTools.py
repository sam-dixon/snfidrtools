import os
import numpy as np
import cPickle as pickle
from astropy.io import fits
import sncosmo
from astropy.cosmology import Planck15 as cosmo
from IPython import embed

"""
A bunch of handy utilities for working with the Nearby Supernova Factory
Internal Data Release.
"""

IDR_dir = '/Users/samdixon/repos/IDRTools/ALLEG2a_SNeIa'
META = os.path.join(IDR_dir, 'META.pkl')
PHREN = '/Users/samdixon/repos/IDRTools/phrenology_2014_04_30_BEDELLv1.pkl'

C = 2.99792458e10
PLANCK = 6.62607e-27

meta = pickle.load(open(META, 'rb'))
phren = pickle.load(open(PHREN, 'rb'))

class Dataset(object):

    def __init__(self, data=meta, subset='training', load_phren=True):
        self.data = {}
        if subset is not None:
            for k, v in data.iteritems():
                k = k.replace('.', '_')
                k = k.replace('-', '_')
                if v['idr.subset'] == subset:
                    self.data[k] = v
        else:
            self.data = data
        self.sne_names = self.data.keys()
        self.sne = [Supernova(self.data, name, load_phren) for name in self.sne_names]
        for k in self.data.iterkeys():
            setattr(self, k, Supernova(self.data, k, load_phren))

    def random_sn(self, n=1):
        """
        Returns a random list of supernovae of length n
        """
        if n == 1:
            return np.random.choice(self.sne, 1)[0]
        else:
            return np.random.choice(self.sne, size=n, replace=False)


class Supernova(object):

    def __init__(self, dataset, name, load_phren=True, load_spec=True):
        data = dataset[name]
        for k, v in data.iteritems():
            k = k.replace('.', '_')
            setattr(self, k, v)
        if load_phren:
            if self.target_name in phren.iterkeys():
                self.in_phren = True
            else:
                self.in_phren = False
        else:
            self.in_phren = None
        if load_spec:
            self.spectra = [Spectrum(dataset, name, obs, self.in_phren) for obs in self.spectra.iterkeys()]
            # Sort spectra by SALT2 phase
            self.spectra = sorted(self.spectra, key=lambda x: x.salt2_phase)

    def get_spec_nearest_max(self):
        """
        Returns the spectrum object for the observation closest to B-band max.
        """
        min_phase = min(np.abs(s.salt2_phase) for s in self.spectra)
        return [s for s in self.spectra if np.abs(s.salt2_phase) == min_phase][0]

    def get_lc(self, filter_name):
        """
        Finds the light curve in some SNf filter using synthetic photometry.
        """
        phase = [spec.salt2_phase for spec in self.spectra]
        mag = [spec.get_snf_magnitude(filter_name) for spec in self.spectra]
        return phase, mag

    def get_salt2_model_fluxes(self):
        """
        Creates the SALT2 model spectra flux based on the fit parameters.
        """
        source = sncosmo.get_source('SALT2', version='2.4')
        model = sncosmo.Model(source=source)
        model.set(z=0, t0=0, x0=self.salt2_X0, x1=self.salt2_X1, c=self.salt2_Color)
        wave = np.arange(3272, 9200, 2)
        measured_phases = [spec.salt2_phase for spec in self.spectra]
        phases = np.linspace(min(measured_phases), max(measured_phases), 100)
        fluxes = model.flux(phases, wave)
        return phases, wave, fluxes

    def get_salt2_model_lc(self, filter_name):
        """
        Creates the SALT2 model light curve based on the fit parameters.
        """
        phases, wave, fluxes = self.get_salt2_model_fluxes()
        filter_edges = {'u' : (3300., 4102.),
                        'b' : (4102., 5100.),
                        'v' : (5200., 6289.),
                        'r' : (6289., 7607.),
                        'i' : (7607., 9200.)}
        min_wave, max_wave = filter_edges[filter_name]
        mag = []
        for flux in fluxes:
            ref_flux = 3.631e-20 * C * 1e8 / wave**2
            flux_sum = np.sum((flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
            ref_flux_sum = np.sum((ref_flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
            mag.append(-2.5*np.log10(flux_sum/ref_flux_sum))
        return phases, mag

class Spectrum(Supernova):
    def __init__(self, dataset, name, obs, load_phren=True):
        data = dataset[name]['spectra'][obs]
        for k, v in data.iteritems():
            k = k.replace('.', '_')
            setattr(self, k, v)
        super(Spectrum, self).__init__(dataset, name, load_phren, load_spec=False)

        if load_phren:
            if self.obs_exp in phren[self.target_name]['spectra'].iterkeys():
                for k, v in phren[self.target_name]['spectra'][self.obs_exp].iteritems():
                    if k is not None:
                        k = k.replace('.', '_')
                        setattr(self, k, v)

    def get_merged_spec(self):
        """
        Returns the merged spectrum from the IDR FITS files.
        """
        path = os.path.join(IDR_dir, self.idr_spec_merged)
        f = fits.open(path)
        head = f[0].header
        flux = f[0].data
        err = f[1].data
        f.close()
        start = head['CRVAL1']
        end = head['CRVAL1']+head['CDELT1']*len(flux)
        npts = len(flux)+1
        wave = np.linspace(start, end, npts)[:-1]
        return wave, flux, err

    def get_rf_spec(self):
        """
        Returns the restframe spectrum from the IDR FITS files.
        """
        path = os.path.join(IDR_dir, self.idr_spec_restframe)
        f = fits.open(path)
        head = f[0].header
        flux = f[0].data
        err = f[1].data
        f.close()
        start = head['CRVAL1']
        end = head['CRVAL1']+head['CDELT1']*len(flux)
        npts = len(flux)+1
        wave = np.linspace(start, end, npts)[:-1]
        #Flux is scaled by a relative distance factor to z=0.05 and multiplied by 1e15
        dl = (1 + self.host_zhelio) * cosmo.comoving_transverse_distance(self.host_zcmb)
        dlref = cosmo.luminosity_distance(0.05)
        flux = flux / ((1+self.host_zhelio)/(1+0.05) * (dl/dlref)**2 * 1e15)
        return wave, flux, err

    def get_salt2_model_fluxes(self):
        """
        Creates the SALT2 model spectra flux based on the fit parameters.
        """
        source = sncosmo.get_source('SALT2', version='2.4')
        model = sncosmo.Model(source=source)
        model.set(z=0, t0=0, x0=self.salt2_X0, x1=self.salt2_X1, c=self.salt2_Color)
        wave = np.arange(3272, 9200, 2)
        flux = model.flux(self.salt2_phase, wave)
        return wave, flux

    def get_magnitude(self, min_wave, max_wave):
        """
        Calculates the AB magnitude in a given top-hat filter.
        """
        wave, flux, flux_err = self.get_rf_spec()
        ref_flux = 3.631e-20 * C * 1e8 / wave**2
        flux_sum = np.sum((flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
        ref_flux_sum = np.sum((ref_flux * wave * 2 / PLANCK / C)[(wave > min_wave) & (wave < max_wave)])
        return -2.5*np.log10(flux_sum/ref_flux_sum)

    def get_snf_magnitude(self, filter_name, z=None):
        """
        Calculates the AB magnitude in a given SNf filter.
        """
        filter_edges = {'u' : (3300., 4102.),
                        'b' : (4102., 5100.),
                        'v' : (5200., 6289.),
                        'r' : (6289., 7607.),
                        'i' : (7607., 9200.)}
        min_wave, max_wave = filter_edges[filter_name]
        return self.get_magnitude(min_wave, max_wave)


if __name__ == '__main__':
    d = Dataset()
    sn = d.SN2005cf
    spec = sn.get_spec_nearest_max()
    # embed()
