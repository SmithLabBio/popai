import unittest
import tempfile
import os
from popai.parse_input import ModelConfigParser
from popai.generate_models import ModelBuilder

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
        """Ensure we get the correct number of models."""

        # read config file
        parser = ModelConfigParser(self.temp_config_file)
        config_values = parser.parse_config()

        # build models
        builder = ModelBuilder(config_values=config_values)
        divergence, secondary_contact, divergence_with_geneflow = builder.build_models()

        # assert
        self.assertEqual(len(divergence[0]), 3)
        self.assertEqual(len(secondary_contact[0]), 7)
        self.assertEqual(len(divergence_with_geneflow[0]), 0)

    def test_build_models_with_divergence_true(self):
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
        builder_modified = ModelBuilder(config_values=config_values_modified)
        divergence_modified, secondary_contact_modified, divergence_with_geneflow_modified = \
            builder_modified.build_models()

        # Assert
        self.assertEqual(len(divergence_modified[0]), 3)
        self.assertEqual(len(secondary_contact_modified[0]), 7)
        self.assertEqual(len(divergence_with_geneflow_modified[0]), 4)

    def test_build_models_with_asymmetric(self):
        """Ensure correct behavior when asymmetric."""
        # Create a modified config file
        temp_config_file_modified = os.path.join(self.temp_dir.name, 'test_config_modified.ini')
        with open(temp_config_file_modified, 'w', encoding='utf-8') as f:
            f.write("""
[Model]
species tree file = ./tests/species_tree.nex
migration matrix = ./tests/migration.txt
symmetric = False
secondary contact = True
divergence with gene flow = True  # Modified this line
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

        # Read modified config file
        parser_modified = ModelConfigParser(temp_config_file_modified)
        config_values_modified = parser_modified.parse_config()

        # Build models with modified config
        builder_modified = ModelBuilder(config_values=config_values_modified)
        divergence_modified, secondary_contact_modified, divergence_with_geneflow_modified = \
            builder_modified.build_models()

        # Assert
        self.assertEqual(len(divergence_modified[0]), 3)
        self.assertEqual(len(secondary_contact_modified[0]), 8)
        self.assertEqual(len(divergence_with_geneflow_modified[0]), 6)

    def test_build_models_with_mult_Trees(self):
        """Ensure correct behavior when there is more than one tree."""
        # Create a modified config file
        temp_config_file_modified = os.path.join(self.temp_dir.name, 'test_config_modified.ini')
        with open(temp_config_file_modified, 'w', encoding='utf-8') as f:
            f.write("""
[Model]
species tree file = ./tests/species_tree_mult.nex
migration matrix = ./tests/migration.txt;./tests/migration2.txt
symmetric = True
secondary contact = True
divergence with gene flow = False  # Modified this line
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

        # Read modified config file
        parser_modified = ModelConfigParser(temp_config_file_modified)
        config_values_modified = parser_modified.parse_config()

        # Build models with modified config
        builder_modified = ModelBuilder(config_values=config_values_modified)
        divergence_modified, secondary_contact_modified, divergence_with_geneflow_modified = \
            builder_modified.build_models()

        # Assert
        self.assertEqual(len(divergence_modified[0]), 3)
        self.assertEqual(len(secondary_contact_modified[0]), 4)
        self.assertEqual(len(divergence_with_geneflow_modified[0]), 0)
        # Assert
        self.assertEqual(len(divergence_modified[1]), 2)
        self.assertEqual(len(secondary_contact_modified[1]), 4)
        self.assertEqual(len(divergence_with_geneflow_modified[1]), 0)

    def test_build_models_with_largertree(self):
        """Ensure correct behavior when there is a larger tree."""
        # Create a modified config file
        temp_config_file_modified = os.path.join(self.temp_dir.name, 'test_config_modified.ini')
        with open(temp_config_file_modified, 'w', encoding='utf-8') as f:
            f.write("""
[Model]
species tree file = ./tests/species_tree_large.nex
migration matrix = ./tests/migration_large.txt
symmetric = True
secondary contact = True
divergence with gene flow = False  # Modified this line
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
alignments = None
popfile = ./tests/populations_large.txt
vcf = alignment_large.vcf
            """)

        # Read modified config file
        parser_modified = ModelConfigParser(temp_config_file_modified)
        config_values_modified = parser_modified.parse_config()

        # Build models with modified config
        builder_modified = ModelBuilder(config_values=config_values_modified)
        divergence_modified, secondary_contact_modified, divergence_with_geneflow_modified = \
            builder_modified.build_models()

        # Assert
        self.assertEqual(len(divergence_modified[0]), 7)
        self.assertEqual(len(secondary_contact_modified[0]), 2)
        self.assertEqual(len(divergence_with_geneflow_modified[0]), 0)



if __name__ == '__main__':
    unittest.main()
