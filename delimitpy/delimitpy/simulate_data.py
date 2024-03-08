"""This module contains all Classes for simulating datasets under specified models using msprime."""
import logging
import time # for testing only
from collections import Counter
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import msprime
import numpy as np
logging.getLogger('msprime').setLevel("WARNING")


class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config, cores, downsampling, max_sites):
        self.models = models
        self.labels = labels
        self.config = config
        self.cores = cores
        self.downsampling = downsampling
        self.max_sites = max_sites
        
        # check taht using even values
        key_even = all(value % 2 == 0 for value in self.downsampling.values())
        if not key_even:
            raise ValueError(f"Error in downampling, all keys must be even.")

        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def simulate_ancestry(self):

        """Perform ancestry simulations with msprime"""

        start_time = time.time()  # Record the start time
        ts_ancestry_dict = {}
        with ThreadPoolExecutor(max_workers=self.cores) as executor:
            # Submit tasks to the ThreadPoolExecutor
            futures = {
                executor.submit(self._run_ancestry_sims, demography[1]): demography[0]
                for demography in enumerate(self.models)
            }
            # Retrieve results as they become available
            for future in futures:
                model_index = futures[future]
                ts_ancestry_dict[f'Model_{model_index}'] = future.result()

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        self.logger.info("Ancestry simulation execution time: %s seconds.", execution_time)

        return ts_ancestry_dict

    def _run_ancestry_sims(self, demography):

        # figure out how to sample individuals
        sampling_dict = {}
        populations = [x.name for x in demography[0].populations]
        sampling_dict = {population: 0 for population in populations}
        for species in self.config['species tree'].leaf_nodes():
            if species.taxon.label not in populations:
                search = True
                searchnode = species
                while search:
                    if searchnode.parent_node.label in populations:
                        sampling_dict[searchnode.parent_node.label] += \
                            self.downsampling[species.taxon.label] / 2
                        search = False
                    else:
                        searchnode = searchnode.parent_node
            else:
                sampling_dict[species.taxon.label] += \
                    self.downsampling[species.taxon.label]/2
        sampling_dict = {key: value for key, value in sampling_dict.items() if value != 0}

        model_ts = []
        for parameterization in enumerate(demography):
            parameter_ts = []

            fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))
            for k,length in enumerate(self.config['lengths']):
                # simulate data
                ts = msprime.sim_ancestry(sampling_dict, demography=parameterization[1],
                                          random_seed = fragment_seeds[k], sequence_length=length,
                                          recombination_rate=0)
                parameter_ts.append(ts)
            model_ts.append(parameter_ts)

        return model_ts

    def simulate_mutations(self, ancestry_dict):

        """Simulate mutations in msprime."""

        start_time = time.time()  # Record the start time
        ts_mutation_dict = {}
        with ThreadPoolExecutor(max_workers=self.cores) as executor:
            # Submit tasks to the ThreadPoolExecutor
            futures = {executor.submit(self._run_mutation_sims, values):
                       key for key, values in ancestry_dict.items()}

            # Retrieve results as they become available
            for future in futures:
                model_index = futures[future]
                ts_mutation_dict[model_index] = future.result()

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        self.logger.info("Mutation simulation execution time: %s seconds.", execution_time)

        return ts_mutation_dict

    def _run_mutation_sims(self, ancestries):

        mutation_rates = self.rng.uniform(low=self.config["mutation_rate"][0],
                                          high=self.config["mutation_rate"][1],
                                          size=self.config["replicates"])
        mutation_rates = np.round(mutation_rates, decimals=20)

        model_mts = []

        for replicate in enumerate(ancestries):

            replicate_mts = []
            fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))

            for fragment in enumerate(replicate[1]):
                mts = msprime.sim_mutations(fragment[1], rate=mutation_rates[replicate[0]],
                                            model=self.config["substitution model"],
                                            random_seed=fragment_seeds[fragment[0]])
                replicate_mts.append(mts)

            model_mts.append(replicate_mts)

        return model_mts

    def mutations_to_numpy(self, mutation_dict):
        """Translate simulated mutations into numpy array."""

        integer_encoding_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        # create arrays for each fragment, and concatenate them together for each parameterization.
        all_arrays = {}
        sizes = []

        for model, values in mutation_dict.items():

            model_array_list = []

            for dataset in values:


                fragment_array_list= []

                for fragment in dataset:

                    fragment_list = []

                    for variant in fragment.variants():
                        alleles = np.array(variant.alleles)
                        nucleotide_encoding = alleles[variant.genotypes]
                        integer_encoding = [integer_encoding_dict[allele]
                                            for allele in nucleotide_encoding]
                        fragment_list.append(np.array(integer_encoding))

                    if len(fragment_list)>0:
                        fragment_array = np.column_stack(fragment_list)
                        fragment_array_list.append(fragment_array)

                if len(fragment_array_list)>0:
                    dataset_array = np.column_stack(fragment_array_list)
                    sizes.append(dataset_array.shape[1])
                else:
                    dataset_array = np.array([])
                    sizes.append(0)

                model_array_list.append(dataset_array)

            all_arrays[model] = model_array_list

        # shorten arrays that are too short, and pad arrays that are too long.
        median_size = int(np.ceil(np.median(sizes)))

        self.logger.info("Median simulated data has %s SNPs, and your input has %s SNPs."\
                         "If these numbers are very different, you may want to change some priors.", 
                         median_size, self.config['variable'])

        for model, values in all_arrays.items():
            for i, matrix in enumerate(values):
                if len(matrix) > 0:
                    if matrix.shape[1] > self.max_sites:
                        all_arrays[model][i] = matrix[:, :self.max_sites]
                    elif matrix.shape[1] < self.max_sites:
                        num_missing_columns = self.max_sites - matrix.shape[1]
                        missing_columns = np.full((matrix.shape[0], num_missing_columns), -1)
                        modified_matrix = np.concatenate((matrix, missing_columns), axis=1)
                        all_arrays[model][i] = modified_matrix
                else:
                    num_missing_columns = self.max_sites - 0
                    modified_matrix = np.full((sum(self.config["sampling_dict"].values()),
                                               num_missing_columns), -1)
                    all_arrays[model][i] = modified_matrix

        return all_arrays

    def mutations_to_sfs(self, numpy_array_dict, nbins=None):
        """Translate simulated mutations into site frequency spectra"""

        all_sfs = []

        # change the order of the sampling dictionary to match the population order in the models
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}

        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in reordered_dict.items():
            sampling_indices[key] = [current, value + current]
            current = current+value


        for values in numpy_array_dict.values():

            model_replicates = []

            for replicate in values:

                # Generate all possible combinations of counts per population
                combos = product(*(range(count + 1) for count in reordered_dict.values()))
                rep_sfs_dict = {'_'.join(map(str, combo)): 0 for combo in combos}

                for site in range(replicate.shape[1]):

                    site_data = list(replicate[:,site])

                    if len(set(site_data)) == 2:

                        # get minor allele
                        minor_allele = min(set(site_data), key=site_data.count)
                        # find poulation counts
                        counts_per_population = {}
                        for population in reordered_dict.keys():
                            site_data_pop = site_data[sampling_indices[population][0]:
                                                      sampling_indices[population][1]]
                            counts_per_population[population] = Counter(site_data_pop)[minor_allele]
                        string_for_count = [str(x) for x in list(counts_per_population.values())]
                        combo_key = '_'.join(string_for_count)
                        rep_sfs_dict[combo_key]+=1

                # convert SFS to binned
                if not nbins is None:
                    thresholds = []
                    for value in reordered_dict.values():
                        thresholds.append([int(np.floor(value/nbins*(x+1))) for x in range(nbins)])
                    threshold_combos = list(product(*thresholds))
                    binned_rep_sfs_dict = {'_'.join(map(str, combo)): 0 for combo in threshold_combos}

                    for key, value in rep_sfs_dict.items():
                        new_string = ''
                        for count, entry in enumerate(key.split('_')):
                            minthresh = min([x for x in thresholds[count] if int(entry) <= x])
                            new_string+=str(minthresh)
                            new_string+='_'
                        new_string = new_string.strip('_')
                        binned_rep_sfs_dict[new_string] += value
                    rep_sfs_dict = binned_rep_sfs_dict

                rep_sfs_dict = [value for value in rep_sfs_dict.values()]
                model_replicates.append(rep_sfs_dict)

            all_sfs.append(model_replicates)

        all_sfs = [item for sublist in all_sfs for item in sublist]
        return all_sfs

    def mutations_to_stats(self, mutation_dict):
        """Convert mutations to summary statistics."""

        all_stats = []

        # change the order of the sampling dictionary to match the population order in the models
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}

        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in reordered_dict.items():
            sampling_indices[key] = [current, value + current]
            current = current+value

        for values in mutation_dict.values():

            for replicate in values:

                summary_stats_replicate = []


                for fragment in replicate:

                    fragment_stats = []

                    # diversity
                    fragment_stats.append(fragment.diversity())
                    # segregating sites
                    fragment_stats.append(fragment.segregating_sites())
                    ## tajima's D
                    #fragment_stats.append(fragment.Tajimas_D())

                    # diversity, segregating sites, Tajima's D within populations

                    for population in reordered_dict.keys():
                        fragment_stats.append(fragment.diversity(sample_sets=\
                            range(sampling_indices[population][0],sampling_indices[population][1])))
                        fragment_stats.append(fragment.segregating_sites(sample_sets=\
                            range(sampling_indices[population][0],sampling_indices[population][1])))
                        #fragment_stats.append(fragment.Tajimas_D(sample_sets=\
                        #    range(sampling_indices[population][0],sampling_indices[population][1])))


                    processed=[]
                    for population in reordered_dict.keys():
                        for population2 in reordered_dict.keys():
                            if population != population2 and (population2, population)\
                                not in processed:
                                fragment_stats.append(fragment.divergence([range\
                                    (sampling_indices[population][0],sampling_indices\
                                    [population][1]), range(sampling_indices[population2][0]\
                                    ,sampling_indices[population2][1])]))
                                #fragment_stats.append(fragment.f2([range(sampling_indices\
                                #    [population][0],sampling_indices[population][1]),
                                #    range(sampling_indices[population2][0],\
                                #    sampling_indices[population2][1])]))
                                #fragment_stats.append(fragment.Fst([range(sampling_indices\
                                #    [population][0],sampling_indices[population][1]),
                                #    range(sampling_indices[population2][0],\
                                #    sampling_indices[population2][1])]))
                                processed.append((population, population2))
                    print(len(fragment_stats))
                    summary_stats_replicate.append(fragment_stats)

                transposed_summary_stats_replicate = list(map(list, zip(*summary_stats_replicate)))
                replicate_means = [np.mean(sublist) for sublist in
                                   transposed_summary_stats_replicate]
                replicate_stds = [np.std(sublist) for sublist in transposed_summary_stats_replicate]
                replicate_statistics = np.array(replicate_means + replicate_stds)
                all_stats.append(replicate_statistics)



        return all_stats

    def _create_numpy_2d_arrays(self, reordered_dict):

        # create empty dictionary to store arrays
        sfs_2d = {}

        # get a list of populations
        populations = list(reordered_dict.keys())

        # iterate over each pair of populations
        for i in range(len(populations)):
            for j in range(i+1, len(populations)):
                pop1 = populations[i]
                pop2 = populations[j]

                # create an empty 2D numpy array with the correct shape
                array_shape = (reordered_dict[pop1]+1, reordered_dict[pop2]+1)
                sfs_2d[(pop1, pop2)] = np.zeros(array_shape)

        return sfs_2d

    def mutations_to_2d_sfs(self, numpy_array_dict):
        """Translate simulated mutations into 2d site frequency spectra"""

        all_sfs = []

        # change the order of the sampling dictionary to match the population order in the models
        population_order = [x.name for x in self.models[-1][0].populations
                            if x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}

        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in reordered_dict.items():
            sampling_indices[key] = [current, value + current]
            current = current+value



        for values in numpy_array_dict.values():

            model_replicates = []

            for replicate in values:

                # generate the empty arrays
                sfs_2d = self._create_numpy_2d_arrays(reordered_dict)

                for site in range(replicate.shape[1]): # iterate over sites

                    site_data = list(replicate[:,site]) # get the data for that site

                    if len(set(site_data)) > 1: # if it is > 1, keep going

                        for key, value in sfs_2d.items(): # iterate over population pairs

                            # get the site data for those two populations
                            site_data_pop1 = site_data[sampling_indices[key[0]][0]:
                                                       sampling_indices[key[0]][1]]
                            site_data_pop2 = site_data[sampling_indices[key[1]][0]:
                                                       sampling_indices[key[1]][1]]
                            all_site_data = site_data_pop1+site_data_pop2

                            # check that this site has two variants in these populations
                            if len(set(all_site_data)) == 2:

                                # get minor allele
                                minor_allele = min(set(all_site_data), key=all_site_data.count)

                                # find counts in each population
                                pop1_count = Counter(site_data_pop1)[minor_allele]
                                pop2_count = Counter(site_data_pop2)[minor_allele]

                                # add to the sfs
                                sfs_2d[key][pop1_count, pop2_count] += 1

                model_replicates.append(sfs_2d)

            all_sfs.append(model_replicates)

        all_sfs = [item for sublist in all_sfs for item in sublist]
        return all_sfs
