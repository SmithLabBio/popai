"""This module contains all Classes for simulating datasets under specified models using msprime."""
import logging
import time # for testing only
from collections import Counter
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import msprime
import numpy as np
import matplotlib.pyplot as plt
logging.getLogger('msprime').setLevel("WARNING")
import sys
import threading

class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config, cores, downsampling, max_sites):
        self.models = models
        self.labels = labels
        self.config = config
        self.cores = cores
        self.downsampling = downsampling
        self.max_sites = max_sites

        # check that using even values
        key_even = all(value % 2 == 0 for value in self.downsampling.values())
        if not key_even:
            raise ValueError("Error in downampling, all keys must be even.")

        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # get overall simulating dict for full demography
        self.simulating_dict = self._get_simulating_dict()

    def _get_simulating_dict(self):
        simulating_dict = {}
        population_count = len(self.config['sampling dict'])
        count=0
        while len(simulating_dict) != population_count:
            populations = [x.name for x in self.models[count][0].populations]
            simulating_dict = {population: 0 for population in populations}
            for species in self.config['species tree'].leaf_nodes():
                if species.taxon.label not in populations:
                    search = True
                    searchnode = species
                    while search:
                        if searchnode.parent_node.label in populations:
                            simulating_dict[searchnode.parent_node.label] += \
                                self.downsampling[species.taxon.label] / 2
                            search = False
                        else:
                            searchnode = searchnode.parent_node
                else:
                    simulating_dict[species.taxon.label] += \
                        self.downsampling[species.taxon.label]/2
            simulating_dict = {key: value for key, value in simulating_dict.items() if value != 0}

            count+=1
        return simulating_dict

    def _get_simulating_dict_model(self, demography):

        # figure out how to sample individuals
        simulating_dict = {}
        populations = [x.name for x in demography.populations]
        simulating_dict = {population: 0 for population in populations}
        for species in self.config['species tree'].leaf_nodes():
            if species.taxon.label not in populations:
                search = True
                searchnode = species
                while search:
                    if searchnode.parent_node.label in populations:
                        simulating_dict[searchnode.parent_node.label] += \
                            self.downsampling[species.taxon.label] / 2
                        search = False
                    else:
                        searchnode = searchnode.parent_node
            else:
                simulating_dict[species.taxon.label] += \
                    self.downsampling[species.taxon.label]/2
        simulating_dict = {key: value for key, value in simulating_dict.items() if value != 0}
        return(simulating_dict)

    def simulate_ancestry(self):

        """Perform ancestry simulations with msprime"""

        start_time = time.time()  # Record the start time

        # dictionary for storing arrays and list for storing sizes.
        all_arrays = {}
        sizes = []
        lock = threading.Lock()  # Create a lock

        # Use ThreadPoolExecutor to parallelize the simulation for each demography
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._simulate_demography, ix, demography, lock, sizes) for ix, demography in enumerate(self.models)]

            # Gather results
            for future in futures:
                model_name, model_arrays = future.result()
                all_arrays[model_name] = model_arrays


        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        self.logger.info("Simulation execution time: %s seconds.", execution_time)

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
                    modified_matrix = np.full((sum(self.config["sampling dict"].values()),
                                               num_missing_columns), -1)
                    all_arrays[model][i] = modified_matrix


        return all_arrays


    def mutations_to_sfs(self, numpy_array_dict, nbins=None):

        """Convert numpy arrays to multidimensional site frequency spectra"""

        all_sfs = []

        # get indices for samples
        reordered_downsampling = {key: self.downsampling[key] for \
                                  key in self.config["sampling dict"]}

        current = 0
        sampling_indices = {}
        for key, value in reordered_downsampling.items():
            sampling_indices[key] = [current, value + current]
            current = current+value

        for values in numpy_array_dict.values():

            model_replicates = []

            for replicate in values:

                # Generate all possible combinations of counts per population
                combos = product(*(range(count + 1) for count in reordered_downsampling.values()))
                rep_sfs_dict = {'_'.join(map(str, combo)): 0 for combo in combos}

                for site in range(replicate.shape[1]):

                    site_data = list(replicate[:,site])

                    if len(set(site_data)) == 2:

                        # get minor allele
                        minor_allele = min(set(site_data), key=site_data.count)
                        # find poulation counts
                        counts_per_population = {}
                        for population in self.config['sampling dict'].keys():
                            site_data_pop = site_data[sampling_indices[population][0]:
                                                      sampling_indices[population][1]]
                            counts_per_population[population] = Counter(site_data_pop)[minor_allele]
                        string_for_count = [str(x) for x in list(counts_per_population.values())]
                        combo_key = '_'.join(string_for_count)
                        rep_sfs_dict[combo_key]+=1

                # convert SFS to binned
                if not nbins is None:
                    thresholds = []
                    for value in reordered_downsampling.values():
                        thresholds.append([int(np.floor(value/nbins*(x+1))) for x in range(nbins)])
                    threshold_combos = list(product(*thresholds))
                    binned_rep_sfs_dict = {'_'.join(map(str, combo)): \
                                           0 for combo in threshold_combos}

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
                model_replicates.append(np.array(rep_sfs_dict))

            all_sfs.append(model_replicates)

        return all_sfs

    def _create_numpy_2d_arrays(self):

        # create empty dictionary to store arrays
        sfs_2d = {}

        # get a list of populations
        populations = list(self.config['sampling dict'].keys())

        # iterate over each pair of populations
        for i, pop1 in enumerate(populations):
            for j, pop2 in enumerate(populations):
                if i < j:
                    # create an empty 2D numpy array with the correct shape
                    array_shape = (self.downsampling[pop1]+1, self.downsampling[pop2]+1)
                    sfs_2d[(pop1, pop2)] = np.zeros(array_shape)

        return sfs_2d

    def plot_2dsfs(self, sfs_list):
        """Plot average 2 dimensional Site frequency spectra."""
        count=0
        for item in sfs_list:
            averages = {}
            for key in item[0].keys():
                arrays = [d[key] for d in item]
                average_array  = np.mean(arrays, axis=0)
                averages[key] = average_array
            # Create heatmaps
            for key, value in averages.items():
                outfile  = os.path.join(self.config["output directory"], \
                                        f"2D_SFS_{key}_model_{count}.png")
                plt.imshow(value, cmap='viridis', origin="lower")
                plt.colorbar()  # Add colorbar to show scale
                plt.title(f"2D SFS {key} for model {count}.")
                plt.savefig(outfile)
                plt.close()
            count+=1

    def mutations_to_2d_sfs(self, numpy_array_dict):
        """Translate simulated mutations into 2d site frequency spectra"""

        all_sfs = []


        # get indices for samples
        reordered_downsampling = {key: self.downsampling[key] \
                                  for key in self.config["sampling dict"]}

        current = 0
        sampling_indices = {}
        for key, value in reordered_downsampling.items():
            sampling_indices[key] = [current, value + current]
            current = current+value

        for values in numpy_array_dict.values():

            model_replicates = []

            for replicate in values:

                # generate the empty arrays
                sfs_2d = self._create_numpy_2d_arrays()

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
                                #print(key, pop1_count, pop2_count)

                                # add to the sfs
                                sfs_2d[key][pop1_count, pop2_count] += 1

                model_replicates.append(sfs_2d)

            all_sfs.append(model_replicates)

        return all_sfs

    def _simulate_demography(self, ix, demography, lock, sizes):

        # get dictionary for simulations
        simulating_dict = self._get_simulating_dict_model(demography=demography[0])
        # draw mutation rates from priors
        mutation_rates = self.rng.uniform(low=self.config["mutation rate"][0],
                                              high=self.config["mutation rate"][1],
                                              size=self.config["replicates"])
        mutation_rates = np.round(mutation_rates, decimals=20)

        # iterate over parameterizations
        model_arrays = []
        for iy, parameterization in enumerate(demography):

            # list for storing arrays from this parameterization
            parameter_arrays = []

            # get seeds for simulating data for each fragment
            fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))

            # iterate over fragments and perform simulations
            for k,length in enumerate(self.config['lengths']):

                # simulate ancestries
                ts = msprime.sim_ancestry(simulating_dict, demography=parameterization,
                                              random_seed = fragment_seeds[k], sequence_length=length,
                                              recombination_rate=0)
                # add mutations
                mts = msprime.sim_mutations(ts, rate=mutation_rates[iy],
                                                model=self.config["substitution model"],
                                                random_seed=fragment_seeds[k])

                # get array
                array = mts.genotype_matrix().transpose()
                parameter_arrays.append(array)

            # combine the parameter_ts arrays across fragments
            dataset_array = np.column_stack(parameter_arrays)

            # Update sizes with lock
            with lock:
                sizes.append(dataset_array.shape[1])


            model_arrays.append(dataset_array)

        return f"Model_{ix}", model_arrays
