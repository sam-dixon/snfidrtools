import os
import sncosmo
import numpy as np
import pickle as pickle
from astropy.io import fits
from extinction import apply, ccm89
from collections import defaultdict
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

try:
    IDR_dir = '/Users/samdixon/data/ALLEG2a_SNeIa'
    os.listdir(IDR_dir)
except:
    IDR_dir = '/Users/maryliu/Desktop'

METAPATH = os.path.join(IDR_dir, 'META.pkl')
META = pickle.load(open(METAPATH, 'rb'))

DLREF = cosmo.luminosity_distance(0.05)


# Rebinning ###########################################################
def lambda_bin(wmin, wmax, velocity):
    n = int(round(np.log10(float(wmax)/wmin) / np.log10(1 + velocity/3.0e5) + 1))
    binedges = np.logspace(np.log10(wmin), np.log10(wmax), n)
    bincenters = (binedges[0:-1] + binedges[1:])/2  # not log centers
    wave = bincenters
    return [wave, binedges]


def recover_bin_edges(bin_centers):
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


def rebin(old_bin_centers, flux, var, new_bin_centers=None):
    old_bin_starts, old_bin_ends = recover_bin_edges(old_bin_centers)
    if new_bin_centers is None:
        new_bin_centers, new_bin_edges = lambda_bin(3300, 8600, 1000)
        new_bin_starts = new_bin_edges[:-1]
        new_bin_ends = new_bin_edges[1:]
    else:
        new_bin_starts, new_bin_ends = recover_bin_edges(new_bin_centers)

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


# SNfactory tophat band definitions ###################################

filter_edges = {'u': (3570., 4102.),
                'b': (4102., 5100.),
                'v': (5200., 6289.),
                'r': (6289., 7607.),
                'i': (7607., 8585.)}
SNF_BANDS = {}
for fname, edges in filter_edges.items():
    wave = [edges[0]-1., edges[0], edges[1], edges[1]+1.]
    trans = [0., 1., 1., 0.]
    SNF_BANDS[fname] = sncosmo.Bandpass(wave, trans, name='snf'+fname)

MAGSYS = sncosmo.get_magsystem('vega')


class Dataset(object):

    def __init__(self, data_dir=IDR_dir, subset=None):
        self.data_dir = data_dir
        data = pickle.load(open(os.path.join(data_dir, 'META.pkl'), 'rb'))
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


class Supernova(object):

    def __init__(self, dataset, name):
        data = dataset[name]
        for k, v in data.items():
            k = k.replace('.', '_')
            setattr(self, k, v)
        self.spectra = [Spectrum(dataset, name, obs) for obs in self.spectra.keys()]
        # Sort spectra by SALT2 phase if possible
        try:
            self.spectra = sorted(self.spectra, key=lambda x: x.salt2_phase)
        except AttributeError:
            pass
        # Remove spectra with processing flags
        self.spectra_noflags = []
        for spec in self.spectra:
            try:
                if (len(spec.procR_Flags) == 0) and (len(spec.procB_Flags) == 0):
                    self.spectra_noflags.append(spec)
            except AttributeError:
                continue

    def spec_nearest(self, phase=0):
        """
        Returns the spectrum object closest to the given phase.
        """
        min_phase = min(np.abs(s.salt2_phase-phase) for s in self.spectra_noflags)
        return [s for s in self.spectra_noflags if np.abs(s.salt2_phase-phase) == min_phase][0]

    def idr_lc(self):
        """
        Lightcurve found in the IDR
        """
        lc = defaultdict(list)
        for spec in self.spectra_noflags:
            lc['phase'].append(spec.salt2_phase)
            for name in SNF_BANDS.keys():
                lc[name].append(getattr(spec, 'mag_{}SNf'.format(name[-1].upper())))
        return lc

    def synth_lc(self):
        """
        Lightcurve synthesized from the spectrum
        """
        lc = defaultdict(list)
        for spec in self.spectra_noflags:
            lc['phase'].append(spec.salt2_phase)
            wave, flux, var = spec.merged_spec()
            snc_spec = sncosmo.Spectrum(wave, flux)
            for name, band in SNF_BANDS.items():
                try:
                    bandflux = snc_spec.bandflux(band)
                    lc[name].append(MAGSYS.band_flux_to_mag(bandflux, band))
                except ValueError:
                    lc[name].append(np.nan)
        return lc

    def synth_lc_array(self):
        """
        Lightcurve as a 5 x n_epochs numpy array (for fitting)
        """
        lcs, phases = [], []
        for spec in self.spectra_noflags:
            if spec.salt2_phase < -10 or spec.salt2_phase > 46:
                continue
            phases.append(spec.salt2_phase)
            wave, flux, var = spec.merged_spec()
            snc_spec = sncosmo.Spectrum(wave, flux)
            for name, band in SNF_BANDS.items():
                try:
                    bandflux = snc_spec.bandflux(band)
                    lcs.append(MAGSYS.band_flux_to_mag(bandflux, band))
                except ValueError:
                    lcs.append(np.nan)
        lcs = np.array(lcs)
        phases = np.array(phases)
        return phases, lcs.reshape(len(phases), 5)

    def spec_array(self):
        """
        Spectra as 288 x n_epochs numpy array (for fitting)
        """
        specs, phases = [], []
        for spec in self.spectra_noflags:
            if spec.salt2_phase < -10 or spec.salt2_phase > 46:
                continue
            phases.append(spec.salt2_phase)
            wave, flux, var = spec.rf_spec()
            rbwave, rbflux, _ = rebin(wave, flux, var)
            specs.append(rbflux)
        specs = np.array(specs)
        phases = np.array(phases)
        return specs, phases


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
        The fluxes are not normalized to z=0.05, Milky Way extinction
        has not been removed, and the wavelengths are in the observer
        frame. Flux units are ergs/s/cm^2/AA.
        """
        path = os.path.join(self.dataset.IDR_dir, self.idr_spec_merged)
        with fits.open(path) as f:
            head = f[0].header
            flux = f[0].data
            var = f[1].data
        start = head['CRVAL1']
        end = head['CRVAL1']+head['CDELT1']*len(flux)
        npts = len(flux)
        wave = np.linspace(start, end, npts)
        return wave, flux, var

    def rf_spec_from_merged(self):
        """
        Returns the rest frame spectrum as calculated by normalizing
        and dereddening the merged spectrum.
        """
        wave, flux, var = self.merged_spec()
        zhelio = self.sn_data['host.zhelio']
        zcmb = self.sn_data['host.zcmb']
        mwebv = self.sn_data['target.mwebv']
        # Remove dust extinction from Milky Way in observer frame
        flux = apply(ccm89(wave, -mwebv*3.1, 3.1), flux)
        # Convert observer frame wavelengths to rest frame
        wave = wave/(1+zhelio)
        # Convert flux to luminosity at z=0.05
        dl = (1+zhelio)*cosmo.comoving_transverse_distance(zcmb)
        cosmo_factor = (1+zhelio)/1.05*(dl/DLREF)**2
        flux = flux*cosmo_factor*1e15
        var = flux*(cosmo_factor*1e15)**2
        return wave, flux, var

    def rf_spec(self):
        """
        Returns the rest frame spectrum directly from the IDR
        Units are normalized ergs/s/cm^2/AA
        """
        path = os.path.join(self.dataset.IDR_dir, self.idr_spec_restframe)
        with fits.open(path) as f:
            head = f[0].header
            flux = f[0].data
            var = f[1].data
        start = head['CRVAL1']
        end = head['CRVAL1']+head['CDELT1']*len(flux)
        npts = len(flux)
        wave = np.linspace(start, end, npts)
        return wave, flux, var
