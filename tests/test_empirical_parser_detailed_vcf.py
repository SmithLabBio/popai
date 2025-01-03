import unittest
import tempfile
import os
import numpy as np
import numpy.testing as npt
from popai.parse_input import ModelConfigParser
from popai.process_empirical import DataProcessor

class TestEmpiricalParser(unittest.TestCase):

    """Test the empirical data processor module."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_file = os.path.join(self.temp_dir.name, 'test_config.ini')

        # Create a sample config file for testing
        with open(self.temp_config_file, 'w', encoding='utf-8') as f:
            f.write("""
[Model]
species tree file = ./tests/species_tree_mini.nex
migration matrix = ./tests/migration_mini.txt
symmetric = True
secondary contact = True
divergence with gene flow = False
max migration events = 1
migration rate = U(1e-5, 1e-4)
constant Ne = True # population sizes equal across all populations

[Other]
output directory = ./examples/test_mini
seed = 1234
replicates = 10

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
alignments = None
popfile = ./tests/populations_mini_vcf.txt
vcf = ./tests/mini_dataset/alignment.vcf

            """)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empirical_data(self):

        """Ensure we correctly parse empirical data."""
        parser = ModelConfigParser(self.temp_config_file)
        config_values = parser.parse_config()

        data_processor = DataProcessor(config=config_values)

        # test numpy array (biallelic)
        empirical_array = data_processor.vcf_to_numpy()

        # NOTE::WHY DOES THIS SEEM WRONG!? CHECK FASTA TEST
        arr_comp = np.array([
            [-1,0,0,0],
            [-1,0,1,0],
            [0,0,1,1],
            [0,0,1,1],
            [1,1,0,0],
            [1,0,0,0]])
        npt.assert_array_equal(arr_comp, empirical_array)
        
        # test 2D sfs without downsampling
        empirical_2d_sfs = data_processor.numpy_to_2d_sfs(
            empirical_array, downsampling={"pop1":2, "pop2":4}, replicates = 1)
        sfs_comp = {('pop2', 'pop1'): np.array(
            [[0., 1., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.]])}
        self.assertEqual(set(sfs_comp.keys()), set(empirical_2d_sfs[0].keys()))
        for key in sfs_comp:
            npt.assert_array_equal(sfs_comp[key], empirical_2d_sfs[0][key])
        
        # test 3D sfs without downsampling or binning
        empirical_msfs, avg_sites = data_processor.numpy_to_msfs(
            empirical_array, downsampling={"pop1":2, "pop2":4}, replicates = 1)
        msfs_comp = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        npt.assert_array_equal(msfs_comp, empirical_msfs[0])

        # test 3D sfs without downsampling but with binning
        empirical_msfs, avg_sites = data_processor.numpy_to_msfs(
            empirical_array, downsampling={"pop1":2, "pop2":4}, replicates = 1, nbins = 2)
        msfs_comp = np.array([2, 0, 1, 0])
        npt.assert_array_equal(msfs_comp, empirical_msfs[0])

        # test 2D sfs with downsampling
        empirical_2d_sfs = data_processor.numpy_to_2d_sfs(
            empirical_array, downsampling={"pop1":2, "pop2":2}, replicates = 3)
        sfs_downs_1 = {('pop2', 'pop1'): np.array([
                            [0., 1., 2.],
                            [1., 0., 0.],
                            [0., 0., 0.]])}
        sfs_downs_2 = {('pop2', 'pop1'): np.array([
                            [0., 1., 1.],
                            [2., 0., 0.],
                            [0., 0., 0.]])}
        sfs_downs_3 = {('pop2', 'pop1'): np.array([
                            [0., 1., 1.],
                            [0., 0., 0.],
                            [1., 0., 0.]])}
        self.assertEqual(set(sfs_downs_1.keys()), set(empirical_2d_sfs[0].keys()))
        self.assertEqual(set(sfs_downs_2.keys()), set(empirical_2d_sfs[1].keys()))
        self.assertEqual(set(sfs_downs_3.keys()), set(empirical_2d_sfs[2].keys()))
        for key in sfs_downs_1:
            npt.assert_array_equal(sfs_downs_1[key], empirical_2d_sfs[0][key])
        for key in sfs_downs_2:
            npt.assert_array_equal(sfs_downs_2[key], empirical_2d_sfs[1][key])
        for key in sfs_downs_3:
            npt.assert_array_equal(sfs_downs_3[key], empirical_2d_sfs[2][key])

if __name__ == '__main__':
    unittest.main()
