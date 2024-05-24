"""This module contains all Classes for parsing user input."""

import configparser # ModelConfigParser
import os
import dendropy # ModelConfigParser
import pandas as pd # ModelConfigParser
from collections import OrderedDict

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

        if not os.path.isfile(self.configfile):
            raise FileNotFoundError(f"The configuration file {self.configfile} does not exist.")

        config = configparser.ConfigParser(inline_comment_prefixes="#")
        config.read(self.configfile)
        config_dict = {}

        try: # determine whether user supplied models
            config_dict['user models'] = config["Model"]["user models"]
        except:
            config_dict['user models'] = None
        if config_dict['user models'] == 'None':
            config_dict['user models'] = None


        try: # read all the keys
            if config_dict['user models'] is None:
                config_dict['species tree'] = dendropy.TreeList.get(
                    path=config['Model']['species tree file'], schema="nexus")
                if len(config_dict["species tree"]) == 0:
                    raise ValueError("Error in species tree.")
                migration_paths = config['Model']['migration matrix'].split(";")
                config_dict['migration df']=[pd.read_csv(x\
                    , index_col=0) for x in migration_paths]
                config_dict['max migration events']=int(config['Model']['max migration events'])
                config_dict["migration rate"] = [float(val.strip("U(").strip(")")) \
                    for val in config['Model']["migration rate"].split(",")]
                config_dict["symmetric"] = config.getboolean("Model", "symmetric")
                config_dict["secondary contact"] = config.getboolean("Model", "secondary contact")
                config_dict["divergence with gene flow"] = config.getboolean(
                    "Model", "divergence with gene flow")
                config_dict["constant Ne"] = config.getboolean(
                    "Model", "constant Ne")
            config_dict['replicates']=int(config['Other']['replicates'])
            config_dict["seed"] = int(config["Other"]["seed"])
            config_dict["mutation rate"] = [float(val.strip("U(").strip(")")) for \
                val in config['Simulations']["mutation rate"].split(",")]
            config_dict["substitution model"] = config["Simulations"]["substitution model"]
            config_dict["popfile"] = config["Data"]["popfile"]
            if config["Data"]["alignments"] == "None":
                with open(config["Data"]["vcf"], 'r') as f:
                    config_dict["vcf"] = f.readlines()
            else:
                config_dict["fasta folder"] = config["Data"]["alignments"]
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
            config_dict["original population dictionary"] = pop_df.set_index('individual')\
                ['population'].to_dict()
            config_dict["sampling dict"] = OrderedDict(pop_df['population'].value_counts().to_dict())


            if config["Data"]["alignments"] == "None":

                config_dict["population dictionary"] = {}
                for key,value in config_dict["original population dictionary"].items():
                    config_dict["population dictionary"][f"{key}_a"] = value
                    config_dict["population dictionary"][f"{key}_b"] = value

                for key,value in config_dict["sampling dict"].items():
                    config_dict["sampling dict"][key] = value*2


                # get lengths
                lengths = [x for x in config_dict["vcf"] if "length" in x]
                lengths = [int(x.split("=")[3].split(">")[0]) for x in lengths]
                config_dict['lengths'] = lengths

                # get individuals
                individuals = [x for x in config_dict["vcf"] if x.startswith("#CHROM")][0]
                individuals = set([x.strip() for x in individuals.split("\t") if x not in ("#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT")])


            else:
                config_dict["population dictionary"] = config_dict["original population dictionary"]

                # get fastas and lengths
                fasta_list = os.listdir(config_dict["fasta folder"])
                fasta_list = [x for x in fasta_list if x.endswith('.fa') or x.endswith('.fasta')]
                config_dict['fastas'] = [dendropy.DnaCharacterMatrix.get(
                    path=os.path.join(config_dict["fasta folder"], x), schema="fasta", \
                        ) for x in fasta_list]
                config_dict['lengths'] = [x.max_sequence_size for x in config_dict['fastas']]

                # get number variable sites
                individuals = self._get_individuals(config_dict['fastas'])

            if set(pop_df['individual']) != individuals:
                raise Exception("Error: The lables in your alignment do not match the labels in your population dictionary.")


        except dendropy.utility.error.DataParseError as e:
            raise ValueError(f"Error parsing tree: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred: {e}") from e

        if config_dict['user models'] is None:
            if config_dict['constant Ne']:
                for tree in config_dict['species tree']:
                    mins = []
                    maxs = []
                    for node in tree.postorder_node_iter():
                        min_ne, max_ne = map(int, node.annotations['ne'].value.strip("'").split("-"))
                        mins.append(min_ne)
                        maxs.append(max_ne)
                    if len(set(mins)) > 1 or len(set(maxs)) > 1:
                        raise ValueError("Error due to using variable population size priors when setting constant Ne to True")

        return config_dict

    def _get_individuals(self, fastas):

        """Count the number of variable sites in the fasta files, 
        while accounting for the presence of IUPAC ambiguity codes, 
        which are all treated as missing."""
        individuals = []
        for item in fastas:
            individuals.extend([str(x) for x in item.taxon_namespace])
        individuals = [x.strip("'") for x in individuals]
        return set(individuals)
