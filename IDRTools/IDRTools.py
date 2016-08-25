import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from IPython import embed
from astropy.io import fits
from scipy.interpolate import UnivariateSpline

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
        min_phase = min(s.salt2_phase for s in self.spectra if s.salt2_phase > 0)
        return [s for s in self.spectra if s.salt2_phase == min_phase][0]


class Spectrum(object):

    def __init__(self, data):
        for k, v in data.iteritems():
            k = k.replace('.', '_')
            setattr(self, k, v)

    def vel_space(self, wave, l_range=(5685, 6570), l0=6355):
        """
        Returns the feature spectrum in velocity space using the relativistic
        Doppler formula (units are km/s).
        """
        c = 3e5  # speed of light in km/s
        dl = wave-l0
        ddl = dl/l0
        v = c*((ddl+1)**2-1)/((ddl+1)**2+1)
        return v

    def smooth_spec(self, wave, f_sn, f_var, n_l=30, smooth_fac=0.005):
        """
        Smooth the input spectrum using the algorithm from Blondin et al 2007.
        """
        f_ts = []
        for i in range(n_l/2, len(f_sn)-n_l/2):
            sig = wave[i]*smooth_fac
            sub = range(i-n_l/2, i+n_l/2)
            x = wave[i]-wave[sub]
            g = 1/np.sqrt(2*np.pi)*np.exp(-1/sig**2*x**2)
            w = g/f_var[sub]
            f_ts_i = np.dot(w, f_sn[sub])/np.sum(w)
            f_ts.append(f_ts_i)
        return wave[n_l/2:-n_l/2], np.array(f_ts)

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

    def get_smoothed_rf_spec(self, n_l=30, smooth_fac=0.005):
        """
        Returns the smoothed restframe spectrum.
        """
        w, f, e = self.get_rf_spec()
        w_s, f_s = self.smooth_spec(w, f, e, n_l=n_l, smooth_fac=smooth_fac)
        return w_s, f_s

    def get_feature_spec(self, l_range=(5685, 6570)):
        """
        Returns a feature spectrum in l_range.
        """
        w, f, e = self.get_rf_spec()
        f = f[(w > l_range[0]) & (w < l_range[1])]
        e = e[(w > l_range[0]) & (w < l_range[1])]
        w = w[(w > l_range[0]) & (w < l_range[1])]
        return w, f, e

    def get_smoothed_feature_spec(self, l_range=(5685, 6570), n_l=30,
                                  smooth_fac=0.005):
        """
        Returns the smoothed feature spectrum.
        """
        w, f, e = self.get_feature_spec(l_range=l_range)
        w_s, f_s = self.smooth_spec(w, f, e, n_l=n_l, smooth_fac=smooth_fac)
        return w_s, f_s

    def get_interp_feature_spec(self, l_range=(5685, 6570), n_l=30,
                                smooth_fac=0.005, grid=0.1,
                                return_spl=False):
        """
        Returns the spline interpolated, smoothed feature spectrum
        """
        w, f = self.get_smoothed_feature_spec(l_range=l_range, n_l=n_l,
                                              smooth_fac=smooth_fac)
        spl = UnivariateSpline(w, f, k=4, s=0)
        n_pts = (max(w)-min(w))/grid + 1
        w_int = np.linspace(min(w), max(w), n_pts)
        f_int = spl(w_int)
        if return_spl:
            return w_int, f_int, spl
        else:
            return w_int, f_int

    def find_peaks(self, l_range=(5685, 6570), n_l=30, smooth_fac=0.005,
                   grid=0.1):
        """
        Finds the peak absorption and emission in specified profile.
        """
        w, f, spl = self.get_interp_feature_spec(l_range=l_range, n_l=n_l,
                                                 smooth_fac=smooth_fac,
                                                 grid=grid,
                                                 return_spl=True)
        roots = spl.derivative().roots()
        peaks = spl(roots)
        return roots, peaks


if __name__ == '__main__':
    d = Dataset()
    for sn in d.sne:
        print sn.target_name
        spec = sn.get_spec_nearest_max()
        w, f, e = spec.get_feature_spec()
        plt.plot(w, f, 'b-', alpha=0.5)
        try:
            w_int, f_int = spec.get_interp_feature_spec()
        except:
            continue
        plt.plot(w_int, f_int, 'k-')
        roots, peaks = spec.find_peaks()
        plt.plot(roots, peaks, 'ro')
        plt.xlabel('Wavelength [$\AA$]')
        plt.ylabel('Flux [erg/s/cm$^2$/$\AA$]')
        plt.title(sn.target_name)
        plt.savefig('/Users/samdixon/repos/velocities/plots/'+sn.target_name+'.png')
        plt.close()
