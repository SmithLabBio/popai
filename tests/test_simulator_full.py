import unittest
import tempfile
import os
import numpy as np
import numpy.testing as npt
from popai.parse_input import ModelConfigParser
from popai.generate_models import ModelBuilder
from popai.simulate_data import DataSimulator

class TestSimulator(unittest.TestCase):

    """Test the data simulator module."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_file = os.path.join(self.temp_dir.name, 'test_config.ini')

        # Create a sample config file for testing
        with open(self.temp_config_file, 'w', encoding='utf-8') as f:
            f.write("""
[Model]
species tree file = ./tests/species_tree.nex
migration matrix = ./tests/migration.txt
symmetric = True
secondary contact = True
divergence with gene flow = False
max migration events = 1
migration rate = U(1e-5, 1e-4)
constant ne = True

[Other]
output directory = ./examples/test
seed = 1234
replicates = 10

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
alignments = ./tests/alignments/
popfile = ./tests/populations.txt

            """)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_models(self):
        """Ensure the simulated data looks correct in terms of dimensions."""

        # read config file
        parser = ModelConfigParser(self.temp_config_file)
        config_values = parser.parse_config()

        # build models
        builder = ModelBuilder(config_values=config_values)
        divergence, secondary_contact, divergence_with_geneflow = builder.build_models()

        # parameterize models
        parameterized_models, labels, sp_tree_index = builder.draw_parameters(
            divergence, secondary_contact, divergence_with_geneflow)

        # simulate data
        downsampling={"A":2, "B": 4, "C":6}
        max_sites = 5
        data_simulator = DataSimulator(parameterized_models, labels, config=config_values, \
                                       cores=1, downsampling=downsampling, max_sites = max_sites, sp_tree_index=sp_tree_index)
        arrays = data_simulator.simulate_ancestry()
        sfs_2d = data_simulator.mutations_to_2d_sfs(arrays)
        msfs = data_simulator.mutations_to_sfs(arrays)
        binned_msfs = data_simulator.mutations_to_sfs(arrays, nbins=2)

        # check shapes
        self.assertEqual(len(arrays), 7)
        self.assertEqual(arrays[0][0].shape, (12, max_sites))
        self.assertEqual(sfs_2d[0][0][('A','B')].shape, (3,5))
        self.assertEqual(sfs_2d[0][0][('A','C')].shape, (3,7))
        self.assertEqual(sfs_2d[0][0][('C','B')].shape, (7,5))
        self.assertEqual(msfs[0][0].shape, (3*5*7,))
        self.assertEqual(binned_msfs[0][0].shape, (2*2*2,))

        # check SFS
        check_2d_sfs_AC = np.array([[0.,1.,0.,0.,0.,0.,0.],
                [2.,0.,0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.,0.,0.]])
        check_2d_sfs_AB = np.array([[0.,2.,0.,0.,0.],
                [1.,1.,0.,0.,0.],
                [0.,0.,0.,0.,0.]])
        check_2d_sfs_BC = np.array([[0.,2.,0.,1.,0.],
                [1.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.]])
        check_msfs = np.array([0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])
        check_msfs_binned = np.array([4, 1, 0, 0, 0, 0, 0, 0])
        npt.assert_array_equal(sfs_2d[3][0][('A','C')], check_2d_sfs_AC)
        npt.assert_array_equal(sfs_2d[3][0][('A','B')], check_2d_sfs_AB)
        npt.assert_array_equal(sfs_2d[3][0][('C','B')], check_2d_sfs_BC)
        npt.assert_array_equal(msfs[3][0], check_msfs)
        npt.assert_array_equal(binned_msfs[3][0], check_msfs_binned)

if __name__ == '__main__':
    unittest.main()
