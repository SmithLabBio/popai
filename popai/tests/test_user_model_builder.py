import unittest
import tempfile
import os
from popai.parse_input import ModelConfigParser
from popai.process_user_models import ModelReader

class TestModelConfigParser(unittest.TestCase):

    """Test the config parser module."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_file = os.path.join(self.temp_dir.name, 'test_config.ini')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_user_models(self):
        """Ensure correct behavior when divergence with gene flow is True."""
        # Create a modified config file
        temp_config_file_modified = os.path.join(self.temp_dir.name, 'test_config_modified.ini')
        with open(temp_config_file_modified, 'w', encoding='utf-8') as f:
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

        # Read modified config file
        parser_modified = ModelConfigParser(temp_config_file_modified)
        config_values_modified = parser_modified.parse_config()

        # Build models with modified config
        model_reader = ModelReader(config_values=config_values_modified)
        demographies, labels= model_reader.read_models()

        # check first instance of model 3
        pops_m3 = demographies[0].populations
        popA_init_m3 = pops_m3[0].initial_size
        time_1_m3 = demographies[0].events[0].time
        time_2_m3 = demographies[0].events[1].time
        
        pops_m1 = demographies[10].populations
        popA_init_m1 = pops_m1[0].initial_size
        time_1_m1 = demographies[10].events[0].time
        time_2_m1 = demographies[10].events[1].time

        pops_m2 = demographies[20].populations
        popA_init_m2 = pops_m2[0].initial_size
        time_1_m2 = demographies[20].events[0].time
        time_2_m2 = demographies[20].events[1].time
        migrate_m2 = demographies[20].migration_matrix[0,1]

        ## Assert
        self.assertEqual(len(pops_m3), 5)
        self.assertEqual(popA_init_m3, 49068.0)
        self.assertEqual(time_1_m3, 0)
        self.assertEqual(time_2_m3, 0)

        self.assertEqual(len(pops_m1), 5)
        self.assertEqual(popA_init_m1, 49545.0 )
        self.assertEqual(time_1_m1, 0)
        self.assertEqual(time_2_m1, 66325.0)

        self.assertEqual(len(pops_m2), 5)
        self.assertEqual(popA_init_m2, 36217.0 )
        self.assertEqual(time_1_m2, 16645.0)
        self.assertEqual(time_2_m2, 51595.0)
        self.assertAlmostEqual(migrate_m2, 2.5642808378226496e-05)


if __name__ == '__main__':
    unittest.main()
