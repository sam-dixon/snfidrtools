import unittest as ut
import IDRTools
import os


class TestGlobal(ut.TestCase):

    def test_files_exists(self):
        self.assertTrue(os.path.isdir(IDRTools.IDR_dir))
        self.assertTrue(os.path.exists(IDRTools.META))


class TestDataset(ut.TestCase):

    def setUp(self):
        self.d_all = IDRTools.Dataset(subset=None)
        self.d_train = IDRTools.Dataset()

    def test_number(self):
        self.assertEqual(len(self.d_all.data.keys()), 389)
        self.assertEqual(len(self.d_train.data.keys()), 112)
        self.assertEqual(len(self.d_all.data.keys()), len(self.d_all.sne))
        self.assertEqual(len(self.d_train.data.keys()), len(self.d_train.sne))


class TestSupernova(ut.TestCase):

    def setUp(self):
        self.sn = IDRTools.Dataset().SNF20080803_000

    def test_num_spectra(self):
        self.assertEqual(len(self.sn.spectra), 16)

    def test_attributes(self):
        self.assertEqual(self.sn.mu, 37.15002236666905)
        self.assertEqual(self.sn.hr, 0.05924636301190134)

    def test_spec_at_max(self):
        max_spec = self.sn.get_spec_nearest_max()
        self.assertEqual(max_spec.salt2_phase, -1.0990963219460115)


class TestSpectrum(ut.TestCase):

    def setUp(self):
        self.sn = IDRTools.Dataset().SNF20080803_000
        self.spec = self.sn.get_spec_nearest_max()

    def test_file_exists(self):
        fname = os.path.join(IDRTools.IDR_dir, self.spec.idr_spec_restframe)
        self.assertTrue(os.path.exists(fname))

    def test_get_rf_spec(self):
        w, f, e = self.spec.get_rf_spec()
        self.assertEqual(len(w), len(f))
        self.assertEqual(len(f), len(e))
        self.assertEqual(len(e), 3030)
        self.assertEqual(min(w), 3122.0)

class TestMath(ut.TestCase):

    def test_pcc(self):
        self.assertEqual(int(IDRTools.math.pearson_corr_coef(xrange(30), xrange(30))), 1)

if __name__ == '__main__':
    ut.main()
