"""This module contains all Classes for simulating datasets under specified models using msprime."""
import logging
logging.getLogger('msprime').setLevel("WARNING")
import msprime
import numpy as np
import dendropy
from collections import Counter
from itertools import product
import sys

class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config):
        self.models = models
        self.labels = labels
        self.config = config
        

        self.rng = np.random.default_rng(self.config['seed'])
    
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def simulate_ancestry(self):



        ts_ancestry = []

        for demography in self.models:

            # figure out how to sample individuals
            sampling_dict = {}
            populations = [x.name for x in demography[0].populations]
            sampling_dict = {population: 0 for population in populations}

            for species in self.config['species tree'].leaf_nodes():
                if species.taxon.label not in populations:
                    search = True
                    searchnode = species
                    while search == True:
                        if searchnode.parent_node.label in populations:
                            sampling_dict[searchnode.parent_node.label]+=self.config['sampling_dict'][species.taxon.label]/2
                            search = False
                        else:
                            searchnode = searchnode.parent_node
                else:
                    sampling_dict[species.taxon.label] +=self.config['sampling_dict'][species.taxon.label]/2
            sampling_dict = {key: value for key, value in sampling_dict.items() if value != 0}

            model_ts = []

            for parameterization in enumerate(demography):
                parameter_ts = []
                
                fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))

                for k,length in enumerate(self.config['lengths']):
                    # simulate data
                    ts = msprime.sim_ancestry(sampling_dict, demography=parameterization[1], random_seed = fragment_seeds[k], sequence_length=length, recombination_rate=0)
                    parameter_ts.append(ts)

                model_ts.append(parameter_ts)
            
            ts_ancestry.append(model_ts)

        self.ts_ancestry = ts_ancestry

        return(ts_ancestry)

    def simulate_mutations(self):
        """Simulate mutations in msprime."""

        mts_mutations = []

        for model in self.ts_ancestry:

            mutation_rates = self.rng.uniform(low=self.config["mutation_rate"][0], high=self.config["mutation_rate"][1], size=self.config["replicates"])
            mutation_rates = np.round(mutation_rates, decimals=20)

            model_mts = []

            for replicate in enumerate(model):

                replicate_mts = []
                fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))

                for fragment in enumerate(replicate[1]):
                    mts = msprime.sim_mutations(fragment[1], rate=mutation_rates[replicate[0]], model=self.config["substitution model"], random_seed=fragment_seeds[fragment[0]])
                    replicate_mts.append(mts)

                model_mts.append(replicate_mts)
            
            mts_mutations.append(model_mts)
        
        self.mts_mutations = mts_mutations
        return(mts_mutations)

    def mutations_to_numpy(self):
        """Translate simulated mutations into numpy array."""

        integer_encoding_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        # create arrays for each fragment, and concatenate them together for each parameterization.
        all_arrays = []
        sizes = []

        for model in self.mts_mutations:

            model_array_list = []

            for dataset in model:

                fragment_array_list= []

                for fragment in dataset:
                
                    fragment_list = []

                    for variant in fragment.variants():
                        site = variant.site
                        alleles = np.array(variant.alleles)
                        nucleotide_encoding = alleles[variant.genotypes]
                        integer_encoding = [integer_encoding_dict[allele] for allele in nucleotide_encoding]
                        fragment_list.append(np.array(integer_encoding))
                    
                    try:
                        fragment_array = np.column_stack(fragment_list)
                        fragment_array_list.append(fragment_array)
                    except:
                        pass

                try:
                    dataset_array = np.column_stack(fragment_array_list)
                    sizes.append(dataset_array.shape[1])
                except:
                    dataset_array = np.array([])
                    sizes.append(0)

                model_array_list.append(dataset_array)
        
            all_arrays.append(model_array_list)

        # find the median length of all arrays, shorten arrays that are too short, and pad arrays that are too long.
        median_size = int(np.ceil(np.median(sizes)))

        self.logger.info(f"Median simulated data has {median_size} basepairs, and your input has {self.config['variable']} basepairs. If these numbers are very different, you may want to change some priors.")

        for sublist in all_arrays:
            for i, matrix in enumerate(sublist):
                try:
                    if matrix.shape[1] > self.config['variable']:
                        sublist[i] = matrix[:, :self.config['variable']]  # Assign modified matrix back to list
                    elif matrix.shape[1] < self.config['variable']:
                        num_missing_columns = self.config['variable'] - matrix.shape[1]
                        missing_columns = np.full((matrix.shape[0], num_missing_columns), -1)  # Create columns of -1s
                        modified_matrix = np.concatenate((matrix, missing_columns), axis=1)  # Add columns of -1s
                        sublist[i] = modified_matrix  # Assign modified matrix back to list
                except:
                    num_missing_columns = self.config['variable'] - 0
                    modified_matrix = np.full((sum(self.config["sampling_dict"].values()), num_missing_columns), -1)  # Create columns of -1s
                    sublist[i] = modified_matrix  # Assign modified matrix back to list

        self.all_arrays = all_arrays
        all_arrays = [item for sublist in all_arrays for item in sublist]
        return(all_arrays)

    def mutations_to_sfs(self):
        """Translate simulated mutations into site frequency spectra"""
        
        all_sfs = []
        population_names = sorted(self.config["sampling_dict"].keys())
        max_count = max(self.config["sampling_dict"].values())
        combos = product(range(max_count + 1), repeat=len(population_names))

        for model in self.all_arrays:

            model_replicates = []

            for replicate in model:

                # Generate all possible combinations of counts per population
                rep_sfs_dict = {combo_key: 0 for combo_key in map('_'.join, product(map(str, range(max_count + 1)), repeat=len(population_names)))}


                for site in range(replicate.shape[1]):

                    site_data = list(replicate[:,site])
                    
                    if len(set(site_data)) == 2:

                            # get minor allele
                            minor_allele = min(set(site_data), key=site_data.count)

                            # find poulation counts
                            counts_per_population = {}
                            start = 0
                            for population, size in self.config["sampling_dict"].items():
                                end = start + size
                                counts_per_population[population] = Counter(site_data[start:end])[minor_allele]
                                start = end
                            #print(counts_per_population)
                            string_for_count = [str(x) for x in list(counts_per_population.values())]
                            combo_key = '_'.join(string_for_count)
                            rep_sfs_dict[combo_key]+=1

                rep_sfs_dict = [value for value in rep_sfs_dict.values()]
                model_replicates.append(rep_sfs_dict)

            all_sfs.append(model_replicates)

        self.all_sfs = all_sfs
        all_sfs = [item for sublist in all_sfs for item in sublist]
        return(all_sfs)

    def mutations_to_stats(self):

        all_stats = []
        population_names = sorted(self.config["sampling_dict"].keys())

        for model in self.mts_mutations:

            for replicate in model:

                summary_stats_replicate = []


                for fragment in replicate:

                    fragment_stats = []

                    # diversity
                    fragment_stats.append(fragment.diversity())
                    # segregating sites
                    fragment_stats.append(fragment.segregating_sites())
                    # tajima's D
                    fragment_stats.append(fragment.Tajimas_D())

                    # diversity, segregating sites, Tajima's D within populations
                    start=0
                    popranges = {}
                    for population in population_names:
                        fragment_stats.append(fragment.diversity(sample_sets=[range(start,start+self.config["sampling_dict"][population])])[0])
                        fragment_stats.append(fragment.segregating_sites(sample_sets=[range(start,start+self.config["sampling_dict"][population])])[0])
                        fragment_stats.append(fragment.Tajimas_D(sample_sets=[range(start,start+self.config["sampling_dict"][population])])[0])
                        popranges[population] = range(start,start+self.config["sampling_dict"][population])
                        start += self.config["sampling_dict"][population]
                    
                    processed=[]
                    for population in population_names:
                        for population2 in population_names:
                            if population != population2 and (population2, population) not in processed:
                                fragment_stats.append(fragment.divergence([popranges[population],popranges[population2]]))
                                fragment_stats.append(fragment.f2([popranges[population],popranges[population2]]))
                                fragment_stats.append(fragment.Fst([popranges[population],popranges[population2]]))
                                processed.append((population, population2))
                    
                    summary_stats_replicate.append(fragment_stats)
                
                transposed_summary_stats_replicate = list(map(list, zip(*summary_stats_replicate)))
                replicate_means = [np.mean(sublist) for sublist in transposed_summary_stats_replicate]
                replicate_stds = [np.std(sublist) for sublist in transposed_summary_stats_replicate]
                replicate_statistics = np.array(replicate_means + replicate_stds)
                all_stats.append(replicate_statistics)
            
        return(all_stats)