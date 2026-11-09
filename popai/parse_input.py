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
                config_dict['max migration events'] = self._whole_number(config, 'Model', 'max migration events', 0)
                config_dict["migration rate"] = self._uniform_prior(
                    config['Model']["migration rate"], "migration rate", allow_nm=True)
                config_dict["symmetric"] = config.getboolean("Model", "symmetric")
                config_dict["secondary contact"] = config.getboolean("Model", "secondary contact")
                config_dict["divergence with gene flow"] = config.getboolean(
                    "Model", "divergence with gene flow")
                config_dict["constant Ne"] = config.getboolean(
                    "Model", "constant Ne")
            config_dict['replicates'] = self._whole_number(config, 'Other', 'replicates', 1)
            config_dict["seed"] = self._whole_number(config, 'Other', 'seed', 0)
            config_dict["mutation rate"] = self._uniform_prior(
                config['Simulations']["mutation rate"], "mutation rate")
            config_dict["substitution model"] = config["Simulations"]["substitution model"].strip()
            if config_dict["substitution model"].lower() not in ('jc69', 'binary'):
                raise ValueError(f"The substitution model is '{config_dict['substitution model']}'. "
                                 f"Use JC69 or binary.")
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

            self._check_migration_tables(config_dict['species tree'], config_dict['migration df'],
                                         migration_paths)

        return config_dict

    def _check_species_trees(self, trees, constant_ne):
        """Check the priors written on each species tree: every node needs ne=min-max, and every
        internal node needs a name and div=min-max, with whole numbers and the smaller value
        first. With constant Ne, every node must have the same ne prior."""

        for number, tree in enumerate(trees, 1):
            ne_priors = set()
            for node in tree.postorder_node_iter():
                name = node.taxon.label if node.is_leaf() else node.label
                if not name:
                    raise ValueError(f"In species tree {number}, an internal node has no name. "
                                     f"Every population needs a name, e.g. (A,B)AB.")
                for key in (['ne'] if node.is_leaf() else ['ne', 'div']):
                    value = node.annotations.get_value(key)
                    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value).strip("'"))
                    if value is None or not match:
                        raise ValueError(f"In species tree {number}, {name} needs {key}=min-max with whole "
                                         f"numbers, e.g. [&{key}=10000-50000]. Found: {value}.")
                    low, high = int(match.group(1)), int(match.group(2))
                    if low > high:
                        raise ValueError(f"In species tree {number}, the {key} prior of {name} is "
                                         f"{low}-{high}. Put the smaller value first.")
                    if key == 'ne':
                        ne_priors.add((low, high))
            if constant_ne and len(ne_priors) > 1:
                shown = ', '.join(f"{low}-{high}" for low, high in sorted(ne_priors))
                raise ValueError(f"constant Ne is True, but species tree {number} has different ne "
                                 f"priors ({shown}). With constant Ne, every population needs the same "
                                 f"ne prior.")

    def _check_migration_tables(self, trees, tables, paths):
        """Each species tree needs its own migration table (in the same order, separated by ;
        in 'migration matrix'), naming the same populations as the tree, with T or F in every
        entry. Spaces around the names are removed."""

        if len(tables) != len(trees):
            raise ValueError(f"There are {len(trees)} species trees but {len(tables)} migration tables. "
                             f"Give one table per tree, separated by ; in 'migration matrix'.")
        for number, (tree, table, path) in enumerate(zip(trees, tables, paths), 1):
            populations = {node.taxon.label if node.is_leaf() else node.label
                           for node in tree.postorder_node_iter()}
            table.index = table.index.astype(str).str.strip()
            table.columns = table.columns.astype(str).str.strip()
            for what, names in (("rows", table.index), ("columns", table.columns)):
                if set(names) != populations:
                    raise ValueError(f"The {what} of the migration table {path.strip()} are {', '.join(names)}, "
                                     f"but species tree {number} has the populations "
                                     f"{', '.join(sorted(populations))}. They must match.")
            bad = [value for value in table.to_numpy().ravel()
                   if str(value).strip().upper() not in ('T', 'F', 'TRUE', 'FALSE')]
            if bad:
                raise ValueError(f"The migration table {path.strip()} should only contain T or F, "
                                 f"but it has {bad[0]}.")

    def _whole_number(self, config, section, key, minimum):
        """A setting that must be a whole number of at least minimum."""
        text = config[section][key]
        try:
            value = int(text)
        except ValueError:
            raise ValueError(f"'{key}' in [{section}] is '{text}', but it must be a whole number.") from None
        if value < minimum:
            raise ValueError(f"'{key}' in [{section}] is {value}, but it must be at least {minimum}.")
        return value

    def _uniform_prior(self, text, setting, allow_nm=False):
        """Read a prior written U(min, max). Returns [min, max] as numbers or, for migration
        rates given as numbers of migrants, as text such as ['1Nm', '2Nm']."""

        example = "U(1e-5, 1e-4) or U(0.5Nm, 2Nm)" if allow_nm else "U(5e-9, 5e-8)"
        match = re.fullmatch(r"\s*U\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)\s*", text)
        if not match:
            raise ValueError(f"The {setting} prior is '{text}'. Write it as {example}.")
        ends = [match.group(1), match.group(2)]

        is_nm = [end.endswith('Nm') for end in ends]
        if any(is_nm) and not allow_nm:
            raise ValueError(f"The {setting} prior is '{text}'. Nm values are only allowed for migration rates.")
        if is_nm[0] != is_nm[1]:
            raise ValueError(f"The {setting} prior is '{text}'. Both values must be rates, or both Nm values.")
        try:
            numbers = [float(end[:-2] if nm else end) for end, nm in zip(ends, is_nm)]
        except ValueError:
            raise ValueError(f"The {setting} prior is '{text}'. Write it as {example}.") from None
        if numbers[0] > numbers[1]:
            raise ValueError(f"The {setting} prior is '{text}'. Put the smaller value first.")
        if numbers[0] < 0:
            raise ValueError(f"The {setting} prior is '{text}', but it can't be negative.")
        return ends if is_nm[0] else numbers

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
