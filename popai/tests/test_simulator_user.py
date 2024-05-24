import unittest
import tempfile
import os
from popai.parse_input import ModelConfigParser
from popai.process_user_models import ModelReader
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
divergence with gene flow = True  # Modified this line
max migration events = 2
migration rate = U(1e-5, 1e-4)
constant ne = True
user models = ./tests/user_models/

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
        reader = ModelReader(config_values=config_values)
        parameterized_models, labels = reader.read_models()

        # simulate data
        downsampling={"A":8, "B": 4, "C":6}
        max_sites = 332
        data_simulator = DataSimulator(parameterized_models, labels, config=config_values, \
                                       cores=1, downsampling=downsampling, max_sites = max_sites, user=True)
        arrays = data_simulator.simulate_ancestry()
        sfs_2d = data_simulator.mutations_to_2d_sfs(arrays)
        msfs = data_simulator.mutations_to_sfs(arrays)
        binned_msfs = data_simulator.mutations_to_sfs(arrays, nbins=4)

        # check shapes
        self.assertEqual(len(arrays), 3)
        self.assertEqual(arrays['Model3'][0].shape, (18, max_sites))
        self.assertEqual(sfs_2d['Model3'][0][('A','B')].shape, (9,5))
        self.assertEqual(sfs_2d['Model3'][0][('A','C')].shape, (9,7))
        self.assertEqual(sfs_2d['Model3'][0][('C','B')].shape, (7,5))
        self.assertEqual(msfs['Model3'][0].shape, (9*5*7,))
        self.assertEqual(binned_msfs['Model3'][0].shape, (4*4*4,))




if __name__ == '__main__':
    unittest.main()
