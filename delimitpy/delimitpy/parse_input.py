"""This module contains all Classes for parsing user input."""

import configparser # ModelConfigParser
import os
import dendropy # ModelConfigParser
import pandas as pd # ModelConfigParser

class ModelConfigParser:

    """Parse user input from the configuration file."""

    def __init__(self, configfile):
        self.configfile = configfile

    def parse_config(self):

        """
        Parse a configuration file and return a dictionary containing the parsed values.

        Parameters:
            configfile (str): Path to the configuration file.

        Returns:
            dict: A dictionary containing the parsed configuration values.

        Raises:
            KeyError: If a required key is missing in the configuration file.
            ValueError: If there is an issue with parsing or converting the configuration values.
        """

        config = configparser.ConfigParser(inline_comment_prefixes="#")
        config.read(self.configfile)
        config_dict = {}

        try: # read all the keys
            config_dict['species tree'] = dendropy.Tree.get(
                path=config['Model']['species tree file'], schema="nexus")
            config_dict['replicates']=int(config['Other']['replicates'])
            config_dict['migration df']=pd.read_csv(config['Model']['migration matrix']\
                , index_col=0)
            config_dict['max migration events']=int(config['Model']['max migration events'])
            config_dict["migration rate"] = [float(val.strip("U(").strip(")")) \
                for val in config['Model']["migration rate"].split(",")]
            config_dict["output directory"] = str(config["Other"]["output directory"])
            config_dict["seed"] = int(config["Other"]["seed"])
            config_dict["symmetric"] = config.getboolean("Model", "symmetric")
            config_dict["secondary contact"] = config.getboolean("Model", "secondary contact")
            config_dict["divergence with gene flow"] = config.getboolean(
                "Model", "divergence with gene flow")
            config_dict["mutation rate"] = [float(val.strip("U(").strip(")")) for \
                val in config['Simulations']["mutation rate"].split(",")]
            config_dict["substitution model"] = config["Simulations"]["substitution model"]
            config_dict["fasta folder"] = config["Data"]["alignments"]
            config_dict["popfile"] = config["Data"]["popfile"]
        except KeyError as e:
            raise KeyError(f"Error in model config: Missing key in configuration file: {e}") from e
        except dendropy.utility.error.DataParseError as e:
            raise ValueError(f"Error parsing tree: {e}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Error in migration table: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred: {e}") from e

        try: # get special information

            # get population sampling info
            pop_df = pd.read_csv(config_dict["popfile"], delimiter='\t')
            config_dict["population dictionary"] = pop_df.set_index('individual')\
                ['population'].to_dict()
            config_dict["sampling dict"] = pop_df['population'].value_counts().to_dict()

            # get fastas and lengths
            fasta_list = os.listdir(config_dict["fasta folder"])
            fasta_list = [x for x in fasta_list if x.endswith('.fa') or x.endswith('.fasta')]
            taxa = dendropy.TaxonNamespace()
            config_dict['fastas'] = [dendropy.DnaCharacterMatrix.get(
                path=os.path.join(config_dict["fasta folder"], x), schema="fasta", \
                    taxon_namespace=taxa) for x in fasta_list]
            config_dict['lengths'] = [x.max_sequence_size for x in config_dict['fastas']]

            # get number variable sites
            config_dict['variable'] = self._count_variable(config_dict['fastas'])

        except dendropy.utility.error.DataParseError as e:
            raise ValueError(f"Error parsing tree: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred: {e}") from e

        return config_dict

    def _count_variable(self, fastas):

        """Count the number of variable sites in the fasta files, 
        while accounting for the presence of IUPAC ambiguity codes, 
        which are all treated as missing."""

        total = 0
        for item in fastas:
            sites = item.max_sequence_size
            for site in range(sites):
                site_list = []
                for individual in enumerate(item):
                    site_list.append(item[individual[0]][site])
                site_list = [str(x) for x in site_list]
                if len(set(site_list)) > 1 and not any(item not in \
                    ['A', 'T', 'C', 'G'] for item in site_list):
                    total+=1
                else:
                    unique_items = set(site_list) - set(['A', 'T', 'C', 'G'])
                    if len(set(site_list)) > 1+len(unique_items):
                        total+=1
        return total
