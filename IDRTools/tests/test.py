import unittest as ut
import IDRTools
import os
import numpy as np


class TestGlobal(ut.TestCase):

    def test_files_exists(self):
        self.assertTrue(os.path.isdir(IDRTools.IDR_dir))
        self.assertTrue(os.path.exists(IDRTools.METAPATH))


class TestDataset(ut.TestCase):

    def setUp(self):
        self.d_all = IDRTools.Dataset(subset=None)
        self.d_train = IDRTools.Dataset(subset='training')
        self.d_train_val = IDRTools.Dataset(subset=['training', 'validation'])
        self.d_bad_sne = IDRTools.Dataset(subset='bad')
        self.d_bad = IDRTools.Dataset(subset=['dfjdskl'])
        self.d_bad2 = IDRTools.Dataset(subset='kfjdk')

    def test_number_in_dataset(self):
        self.assertEqual(len(self.d_all), 389)
        self.assertEqual(len(self.d_train), 112)
        self.assertEqual(len(self.d_train_val), 223)
        self.assertEqual(len(self.d_bad_sne), 158)
        self.assertEqual(len(self.d_bad), 0)
        self.assertEqual(len(self.d_bad2), 0)


class TestSupernova(ut.TestCase):

    def setUp(self):
        self.sn = IDRTools.Dataset().SNF20080803_000

    def test_num_spectra(self):
        self.assertEqual(len(self.sn.spectra), 16)

    def test_attributes(self):
        self.assertEqual(self.sn.distmod, 37.15190054296799)
        self.assertEqual(self.sn.hr, 0.060198238035383156)

    def test_spec_at_max(self):
        max_spec = self.sn.spec_nearest_max()
        self.assertEqual(max_spec.salt2_phase, -1.0990963219460115)


class TestSpectrum(ut.TestCase):

    def setUp(self):
        self.sn = IDRTools.Dataset().SNF20080803_000
        self.spec = self.sn.spec_nearest_max()

    # def test_file_exists(self):
    #     fname = os.path.join(IDRTools.IDR_dir, self.spec.idr_spec_restframe)
    #     self.assertTrue(os.path.exists(fname))

    # def test_rf_spec(self):
    #     w, f, e = self.spec.rf_spec()
    #     self.assertEqual(len(w), len(f))
    #     self.assertEqual(len(f), len(e))

    # def test_merged_spec(self):
    #     w, f, e = self.spec.merged_spec()
    #     self.assertEqual(len(w), len(f))
    #     self.assertEqual(len(f), len(e))

    # def test_rebin(self):
    #     w = np.arange(2000, 8000, 2)
    #     flux = np.ones(len(w))
    #     var = np.ones(len(w))
    #     new_w = np.arange(1000, 9000, 4)
    #     new_wave, binned_flux, binned_var = self.spec.rebin(w, flux, var, new_w)
    #     print new_wave, binned_flux
        # np.testing.assert_array_equal(new_wave, np.array([1, 3]))
        # np.testing.assert_array_equal(binned_flux, [30, 70])
        # np.testing.assert_array_equal(binned_var, [15, 20])


if __name__ == '__main__':
    ut.main()
