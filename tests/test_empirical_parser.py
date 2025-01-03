import unittest
import tempfile
import os
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
species tree file = ./tests/species_tree.nex
migration matrix = ./tests/migration.txt
symmetric = True
secondary contact = True
divergence with gene flow = False
max migration events = 2
migration rate = U(1e-5, 1e-4)
constant Ne = True # population sizes equal across all populations

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

    def test_empirical_data(self):

        """Ensure we correctly parse empirical data."""
        parser = ModelConfigParser(self.temp_config_file)
        config_values = parser.parse_config()

        data_processor = DataProcessor(config=config_values)
        empirical_array = data_processor.fasta_to_numpy()
        data_processor.find_downsampling(empirical_array)
        empirical_2d_sfs = data_processor.numpy_to_2d_sfs(
            empirical_array, downsampling={"A":8, "B":6, "C":6}, replicates = 10)
        empirical_msfs, sites = data_processor.numpy_to_msfs(
            empirical_array, downsampling={"A":8, "B": 6, "C":6}, replicates = 10)
        empirical_msfs_binned, sites_binned = data_processor.numpy_to_msfs(
            empirical_array, downsampling={"A":8, "B": 6, "C":6}, replicates = 10, nbins=4)


        self.assertEqual(len(empirical_2d_sfs), 10)
        self.assertEqual(len(empirical_msfs), 10)
        self.assertEqual(empirical_2d_sfs[0][('A','B')].shape, (9,7))
        self.assertEqual(empirical_2d_sfs[0][('A','C')].shape, (9,7))
        self.assertEqual(empirical_2d_sfs[0][('C','B')].shape, (7,7))
        self.assertEqual(empirical_msfs[0].shape, (9*7*7,))
        self.assertEqual(empirical_msfs_binned[0].shape, (4*4*4,))
        self.assertEqual(empirical_array.shape, (30,1038))

if __name__ == '__main__':
    unittest.main()
