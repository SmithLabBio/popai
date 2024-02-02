"""This module contains all Classes for parsing user input."""

import configparser # ModelConfigParser
import dendropy # ModelConfigParser
import pandas as pd # ModelConfigParser
import os


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

        try:
            config_dict['species tree'] = dendropy.Tree.get(path=config['Model']['species tree file'], schema="nexus")
            config_dict['replicates']=int(config['Other']['replicates'])
            config_dict['migration_df']=pd.read_csv(config['Model']['migration matrix'], index_col=0)
            config_dict['max migration events']=int(config['Model']['max migration events'])
            config_dict["migration_rate"] = [float(val.strip("U(").strip(")")) for val in config['Model']["migration rate"].split(",")]
            config_dict["output directory"] = str(config["Other"]["output directory"])
            config_dict["seed"] = int(config["Other"]["seed"])
            config_dict["symmetric"] = config.getboolean("Model", "symmetric")
            config_dict["secondary contact"] = config.getboolean("Model", "secondary contact")
            config_dict["divergence with gene flow"] = config.getboolean("Model", "divergence with gene flow")
            config_dict["mutation_rate"] = [float(val.strip("U(").strip(")")) for val in config['Simulations']["mutation rate"].split(",")]
        except KeyError as e:
            raise KeyError(f"Error in model config: Missing key in configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error in model config: {e}")
        
        try:
            config_dict["substitution model"] = config["Simulations"]["substitution model"]
        except KeyError as e:
            raise KeyError(f"Error in simulation config: Missing key in configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error in simulation config: {e}")

        try:
            config_dict["fasta_folder"] = config["Data"]["alignments"]
            config_dict["popfile"] = config["Data"]["popfile"]

            # get population sampling info
            pop_df = pd.read_csv(config_dict["popfile"], delimiter='\t')
            config_dict["population_dictionary"] = pop_df.set_index('trait')['species'].to_dict()
            config_dict["sampling_dict"] = pop_df['species'].value_counts().to_dict()

            # get fastas and lengths
            fasta_list = os.listdir(config_dict["fasta_folder"])
            fasta_list = [x for x in fasta_list if x.endswith('.fa') or x.endswith('.fasta')]
            config_dict['fastas'] = [dendropy.DnaCharacterMatrix.get(path=os.path.join(config_dict["fasta_folder"], x), schema="fasta") for x in fasta_list]
            config_dict['lengths'] = [x.max_sequence_size for x in config_dict['fastas']]

            config_dict['variable'] = self.count_variable(config_dict['fastas'])

        except KeyError as e:
            raise KeyError(f"Error in empirical data config: Missing key in configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error in empirical data config: {e}")

        return(config_dict)

    def count_variable(self, fastas):
    
        total = 0
    
        for item in fastas:
        
            sites = item.max_sequence_size

            for site in range(sites):
                
                site_list = []

                for individual in range(len(item)):
                
                    site_list.append(item[individual][site])
            
                if len(set(site_list)) > 1:
                    total+=1
        
        return(total)