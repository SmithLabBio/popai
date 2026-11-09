"""This module contains all Classes for parsing user input."""

import configparser # ModelConfigParser
import os
import re
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

         # determine whether user supplied models
        config_dict['user models'] = self._optional(config, "Model", "user models")

        try: # read all the keys
            if config_dict['user models'] is None:
                config_dict['species tree'] = dendropy.TreeList.get(
                    path=config['Model']['species tree file'], schema="nexus")
                if len(config_dict["species tree"]) == 0:
                    raise ValueError(f"The species tree file {config['Model']['species tree file']} has no trees.")
                migration_paths = config['Model']['migration matrix'].split(";")
                config_dict['migration df']=[pd.read_csv(x\
                    , index_col=0) for x in migration_paths]
                config_dict['max migration events']=int(config['Model']['max migration events'])
                try:
                    config_dict["migration rate"] = [float(val.strip("U(").strip(")")) \
                        for val in config['Model']["migration rate"].split(",")]
                except:
                    config_dict["migration rate"] = [val.strip("U(").strip(")") \
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
            alignments = self._optional(config, "Data", "alignments")
            if alignments is None:
                vcf = self._optional(config, "Data", "vcf")
                if vcf is None:
                    raise ValueError("Error in model config: the [Data] section needs alignments "
                                     "(a folder of fasta files) or vcf (a vcf file).")
                with open(vcf, 'r') as f:
                    config_dict["vcf"] = f.readlines()
            else:
                config_dict["fasta folder"] = alignments
        except KeyError as e:
            raise KeyError(f"Error in model config: Missing key in configuration file: {e}") from e
        except dendropy.utility.error.DataParseError as e:
            raise ValueError(f"Error parsing tree: {e}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Error in migration table: {e}") from e

        try: # get special information
            # get population sampling info
            pop_df = pd.read_csv(config_dict["popfile"], delimiter='\t', dtype=str)
            pop_df.columns = pop_df.columns.str.strip()
            pop_df = pop_df.apply(lambda column: column.str.strip())   # remove spaces around names'
            if not all(col in pop_df.columns for col in ['individual', 'population']):
                raise ValueError(f"The popfile must be tab-separated, with a header row naming the columns "
                                 f"individual and population. Found the columns: {', '.join(pop_df.columns)}.")
            if not isinstance(pop_df.index, pd.RangeIndex):
                raise ValueError("Some lines in the popfile have more than two columns. Each line should "
                                 "be an individual and its population, separated by a tab.")
            if pop_df[['individual', 'population']].isna().any().any():
                raise ValueError("Some lines in the popfile are missing an individual or a population.")
            config_dict["original population dictionary"] = pop_df.set_index('individual')\
                ['population'].to_dict()
            config_dict["sampling dict"] = OrderedDict(pop_df['population'].value_counts().to_dict())

            if "vcf" in config_dict:
                data_source = "vcf"
                config_dict["population dictionary"] = {}
                for key,value in config_dict["original population dictionary"].items():
                    config_dict["population dictionary"][f"{key}_a"] = value
                    config_dict["population dictionary"][f"{key}_b"] = value

                for key,value in config_dict["sampling dict"].items():
                    config_dict["sampling dict"][key] = value*2

                # length of each contig that has data
                config_dict['lengths'] = self._vcf_lengths(config_dict["vcf"], config["Data"])

                # get individuals
                individuals = [x for x in config_dict["vcf"] if x.startswith("#CHROM")][0]
                individuals = set([x.strip() for x in individuals.split("\t") if x not in ("#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT")])

            else:
                data_source = "alignment"
                config_dict["population dictionary"] = config_dict["original population dictionary"]

                # get fastas and lengths
                fasta_list = sorted(os.listdir(config_dict["fasta folder"]))
                fasta_list = [x for x in fasta_list if x.endswith('.fa') or x.endswith('.fasta') or x.endswith('.fas')]
                if not fasta_list:
                    raise ValueError(f"There are no .fa, .fas, or .fasta files in {config_dict['fasta folder']}.")
                config_dict['fastas'] = [dendropy.DnaCharacterMatrix.get(
                    path=os.path.join(config_dict["fasta folder"], x), schema="fasta", \
                        ) for x in fasta_list]
                config_dict['lengths'] = [x.max_sequence_size for x in config_dict['fastas']]

                # get number variable sites
                individuals = self._get_individuals(config_dict['fastas'])

            # Assert that popfile and alignment/vcf sample ids match
            popfile_id_set = set(pop_df["individual"])
            alignment_id_set = set(individuals)

            # Only in popfile
            unique_popfile_ids = popfile_id_set - alignment_id_set
            if unique_popfile_ids:
                raise ValueError(f"These samples are in the popfile but not in the {data_source} file: "
                                 f"{', '.join(sorted(unique_popfile_ids))}.")
            # Only in alignment
            unique_alignment_ids = alignment_id_set - popfile_id_set
            if unique_alignment_ids:
                raise ValueError(f"These samples are in the {data_source} file but not in the popfile: "
                                 f"{', '.join(sorted(unique_alignment_ids))}.")

        except dendropy.utility.error.DataParseError as e:
            raise ValueError(f"Error reading a fasta file: {e}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Error reading the popfile (it should be tab-separated, with two "
                             f"columns): {e}") from e
        
        if config_dict['user models'] is None:
            self._check_species_trees(config_dict['species tree'], config_dict['constant Ne'])

        return config_dict

    def _optional(self, config, section, key):
        """A setting that can be left out or set to None (in any case): its value, or None."""
        value = config.get(section, key, fallback=None)
        if value is None or value.strip().lower() in ('', 'none'):
            return None
        return value.strip()

    def _vcf_lengths(self, vcf, data):
        """The length of each contig that has data in the VCF, in the order they first appear.
        Lengths come from the ##contig header lines (##contig=<ID=1,length=1000>); a contig
        without one uses 'length' from the [Data] section of the config file."""

        # lengths given in the header
        header_lengths = {}
        for line in vcf:
            if not line.startswith('##'):
                break   # the header is over
            if line.startswith('##contig'):
                contig = re.search(r"[<,]ID=([^,>]+)", line)
                length = re.search(r"[<,]length=(\d+)", line)
                if contig and length:
                    header_lengths[contig.group(1)] = int(length.group(1))

        # contigs with data, in the order they first appear
        contigs = list(dict.fromkeys(line.split('\t')[0] for line in vcf
                                     if line.strip() and not line.startswith('#')))

        lengths, missing = [], []
        for contig in contigs:
            if contig in header_lengths:
                lengths.append(header_lengths[contig])
            elif 'length' in data:
                lengths.append(int(data['length']))
            else:
                missing.append(contig)
        if missing:
            shown = ', '.join(missing[:10]) + (', ...' if len(missing) > 10 else '')
            raise ValueError(f"The VCF has no length for contig(s) {shown}. Add ##contig=<ID=...,length=...> "
                             f"lines to the VCF header, or add 'length = ...' to the [Data] section "
                             f"of the config file.")
        return lengths

    def _get_individuals(self, fastas):

        """Count the number of variable sites in the fasta files, 
        while accounting for the presence of IUPAC ambiguity codes, 
        which are all treated as missing."""
        individuals = []
        for item in fastas:
            individuals.extend([str(x) for x in item.taxon_namespace])
        individuals = [x.strip("'") for x in individuals]
        return set(individuals)
