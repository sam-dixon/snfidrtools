import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from IPython import embed
from astropy.io import fits

"""
A bunch of handy utilities for working with the Nearby Supernova Factory
Internal Data Release.
"""

IDR_dir = '/Users/samdixon/repos/snifs/ALLEG2a_SNeIa'
META = os.path.join(IDR_dir, 'META.pkl')


class Dataset(object):

    def __init__(self, idr_dir=IDR_dir, meta=META, subset='training'):
        self.idr_dir = idr_dir
        self.meta = pickle.load(open(meta, 'rb'))
        self.data = {}
        if subset is not None:
            for k, v in self.meta.iteritems():
                k = k.replace('.', '_')
                k = k.replace('-', '_')
                if v['idr.subset'] == subset:
                    self.data[k] = v
        else:
            self.data = self.meta
        self.sne_names = self.data.keys()
        self.sne = [Supernova(v) for v in self.data.itervalues()]
        for k, v in self.data.iteritems():
            setattr(self, k, Supernova(v))

    def random_sne(self, n=1):
        """
        Returns a random list of supernovae of length n
        """
        if n == 1:
            return np.random.choice(self.sne, 1)[0]
        else:
            return np.random.choice(self.sne, size=n, replace=False)


class Supernova(object):

    def __init__(self, data):
        for k, v in data.iteritems():
            k = k.replace('.', '_')
            setattr(self, k, v)
        self.spectra = [Spectrum(obs) for obs in self.spectra.itervalues()]

    def get_spec_nearest_max(self):
        """
        Returns the spectrum object for the observation closest to B-band max.
        """
        min_phase = min(np.abs(s.salt2_phase) for s in self.spectra if s.salt2_phase > 0)
        return [s for s in self.spectra if s.salt2_phase == min_phase][0]


class Spectrum(object):

    def __init__(self, data):
        for k, v in data.iteritems():
            k = k.replace('.', '_')
            setattr(self, k, v)
        try:
            roots, peaks = self.get_interp_feature_spec()
            self.l_abs = roots[peaks == min(peaks)]
            self.v_abs = self.vel_space(self.l_abs)
        except:
            pass

    def get_rf_spec(self):
        """
        Returns the restframe spectrum info from the IDR FITS files.
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
        return wave, flux, err


if __name__ == '__main__':
    d = Dataset()
    sn = d.random_sne()
    spec = sn.get_spec_nearest_max()
    embed()
