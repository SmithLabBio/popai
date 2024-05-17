import unittest
import tempfile
import os
from popai.parse_input import ModelConfigParser

class TestModelConfigParser(unittest.TestCase):

    """Test the config parser module."""

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

    def test_parse_config(self):
        """Ensure we can parse the temp config file."""
        parser = ModelConfigParser(self.temp_config_file)
        parser.parse_config()

if __name__ == '__main__':
    unittest.main()
