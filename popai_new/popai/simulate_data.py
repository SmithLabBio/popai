"""This module contains all Classes for simulating datasets under specified models using msprime."""
import logging
import time # for testing only
from collections import Counter
from itertools import product
import os
import msprime
import numpy as np
import matplotlib.pyplot as plt
logging.getLogger('msprime').setLevel("WARNING")
import sys
import pyslim

class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config, cores, downsampling, max_sites, user=False, sp_tree_index = False):
        self.models = models
        self.labels = labels
        self.config = config
        self.cores = cores
        self.downsampling = downsampling
        self.max_sites = max_sites
        self.user = user
        self.sp_tree_index = sp_tree_index

        # check that using even values
        key_even = all(value % 2 == 0 for value in self.downsampling.values())
        if not key_even:
            raise ValueError("Error in downampling, all keys must be even.")

        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_simulating_dict(self, tree):
        simulating_dict = {}
        population_count = len(self.config['sampling dict'])
        count=0
        while len(simulating_dict) != population_count:
            populations = [x.name for x in self.models[count][0].populations]
            simulating_dict = {population: 0 for population in populations}
            for species in tree.leaf_nodes():
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

    def _get_simulating_dict_model(self, demography, tree):

        # figure out how to sample individuals
        simulating_dict = {}
        populations = [x.name for x in demography.populations]
        simulating_dict = {population: 0 for population in populations}
        for species in tree.leaf_nodes():
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

    def _get_simulating_dict_demo(self, demography):
        # get sampling dict
        this_sampling_dict = {}
        initially_active = []
        sampled_inactive = []
        all_relevant_descendents = []
        all_descendents = []
        
        for population in demography.populations:
            if population.initially_active is None:
                initially_active.append(population.name)
            elif population.initially_active == False and population.default_sampling_time == 0.0:
                sampled_inactive.append(population.name)

        for population in sampled_inactive:
            to_check = [population]
            relevant_descendants = []
            while len(to_check) > 0:
                for item in to_check:
                    for event in demography.events:
                        if hasattr(event, 'ancestral'):
                            if event.ancestral == item:
                                to_check.extend(event.derived)
                                to_check.remove(item)
                                relevant_descendants.extend([x for x in to_check if x in initially_active])
                                all_descendents.extend([x for x in to_check])
                                to_check = [x for x in to_check if x not in initially_active]
                            
            this_sampling_dict[population] = 0
            for item in relevant_descendants:         
                this_sampling_dict[population] += self.config["sampling dict"][item]
            all_relevant_descendents.extend(relevant_descendants)
        
        revised_sampling_dictionary = {}

        initially_active = [x for x in initially_active if not x in all_relevant_descendents]

        for population in initially_active:
            this_sampling_dict[population] = self.config["sampling dict"][population]

        for key,value in this_sampling_dict.items():
            if key not in all_descendents:
                if value % 2 != 0:
                    raise Exception("Remember we simulate diploid individuals. If you have an odd number of samples, something has gone wrong.")
                revised_sampling_dictionary[key] = value // 2

        return(revised_sampling_dictionary, this_sampling_dict)

    def simulate_ancestry(self):

        """Perform ancestry simulations with msprime"""

        start_time = time.time()  # Record the start time

        # dictionary for storing arrays and list for storing sizes.
        all_arrays = {}
        sizes = []
        
        for ix, demography in enumerate(self.models):

            if self.user == True:
                matrix, sizes = self._simulate_demography_user(demography)
            else:
                matrix, sizes = self._simulate_demography(demography,  self.config['species tree'][self.sp_tree_index[ix]])

            if self.labels[ix] in all_arrays:
                all_arrays[self.labels[ix]].append(matrix)
            else:
                all_arrays[self.labels[ix]] = [matrix]

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time

        self.logger.info("Simulation execution time: %s seconds.", execution_time)

        # shorten arrays that are too short, and pad arrays that are too long.
        median_size = int(np.ceil(np.median(sizes)))

        self.logger.info("Median simulated data has %s biallelic SNPs."\
                         " If this is very different than the number of SNPs in your empirical data, you may want to change some priors.", 
                         median_size)

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

    def plot_2dsfs(self, sfs_list, output_directory):
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
                outfile  = os.path.join(output_directory, \
                                        f"2D_SFS_{key}_model_{count}.png")
                plt.imshow(value, cmap='viridis', origin="lower")
                plt.colorbar()  # Add colorbar to show scale
                plt.title(f"2D SFS {key} for model {count}.")
                plt.savefig(outfile)
                plt.close()
            count+=1

    def organize_matrix(self, array, simulating_dict, downsampling):

        taxon_names = [str(x).strip("'") for x in self.config['species tree'][0].taxon_namespace]
        new_simulating_dict = {}
        for key in simulating_dict:
            if key not in taxon_names:
                # find descendents
                for node in self.config['species tree'][0].preorder_node_iter():
                    if node.label == key:
                        descendants = [str(x.taxon).strip("'") for x in node.leaf_nodes()]
                        for descendant in descendants:
                            new_simulating_dict[descendant] = downsampling[descendant]
            else:
                new_simulating_dict[key] = downsampling[key]
        start = 0
        array_list = []
        indices = {}
        for key, value in new_simulating_dict.items():
            indices[key] = [int(start), int(start+value)]
            start += value
        for key, value in self.config['sampling dict'].items():
            pop_matrix = array[indices[key][0]:indices[key][1]]
            start+=value
            reference_row = pop_matrix[np.random.randint(pop_matrix.shape[0])]
            distances = np.linalg.norm(pop_matrix - reference_row, axis=1)
            sorted_indices = np.argsort(distances)
            sorted_array = pop_matrix[sorted_indices]
            array_list.append(sorted_array)
        array = np.vstack(array_list)
        return(array)

    def organize_matrix_user(self, array, simulating_dict):

        if list(simulating_dict.keys()) != list(self.config['sampling dict'].keys()):
            raise Exception("There is an issue with taxon order. Please contact the developers. This should not occur.")
        
        start = 0
        array_list = []
        for key, value in self.config['sampling dict'].items():
            pop_matrix = array[start:start+value]
            start+=value
            reference_row = pop_matrix[np.random.randint(pop_matrix.shape[0])]
            distances = np.linalg.norm(pop_matrix - reference_row, axis=1)
            sorted_indices = np.argsort(distances)
            sorted_array = pop_matrix[sorted_indices]
            array_list.append(sorted_array)
        array = np.vstack(array_list)
        return(array)

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

    def _simulate_demography(self, demography, tree):

        # get dictionary for simulations
        simulating_dict = self._get_simulating_dict_model(demography=demography, tree=tree)

        # draw mutation rates from priors
        mutation_rates = self.rng.uniform(low=self.config["mutation rate"][0],
                                              high=self.config["mutation rate"][1],
                                              size=1)[0]
        mutation_rates = np.round(mutation_rates, decimals=20)

        # list for storing arrays from this parameterization
        parameter_arrays = []

        # get seeds for simulating data for each fragment
        fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))

        # iterate over fragments and perform simulations
        for k,length in enumerate(self.config['lengths']):

            # simulate ancestries
            ts = msprime.sim_ancestry(simulating_dict, demography=demography,
                                        random_seed = fragment_seeds[k], sequence_length=length,
                                        recombination_rate=0)
            # add mutations
            mts = msprime.sim_mutations(ts, rate=mutation_rates,
                                        model=self.config["substitution model"],
                                        random_seed=fragment_seeds[k])

            # get array
            array = mts.genotype_matrix().transpose()
        
            # remove non-biallelic columns
            if array.shape[1] > 0:
                frequencies = np.array([[np.sum(array[:, j] == i) \
                    for i in range(0, np.max(array))] for j in range(array.shape[1])])
                nonbiallelic_columns = np.where(np.sum(frequencies != 0, axis=1) > 2)[0]
                array = np.delete(array, nonbiallelic_columns, axis=1)

            # organize matrix
            array = self.organize_matrix(array, simulating_dict, downsampling=self.downsampling)

            parameter_arrays.append(array)

        # combine the parameter_ts arrays across fragments
        dataset_array = np.column_stack(parameter_arrays)

        return dataset_array, dataset_array.shape[1]

    def _simulate_demography_user(self, demography):

        # get dictionary for simulations
        simulating_dict, retained_dict = self._get_simulating_dict_demo(demography=demography)

        # draw mutation rates from priors
        mutation_rates = self.rng.uniform(low=self.config["mutation rate"][0],
                                              high=self.config["mutation rate"][1],
                                              size=1)[0]
        mutation_rates = np.round(mutation_rates, decimals=20)

        # get seeds for simulating data for each fragment
        fragment_seeds = self.rng.integers(2**32, size=len(self.config['lengths']))

        # iterate over fragments and perform simulations
        parameter_arrays = []
        for k,length in enumerate(self.config['lengths']):

            # simulate ancestries
            ts = msprime.sim_ancestry(simulating_dict, demography=demography,
                random_seed = fragment_seeds[k], sequence_length=length,
                recombination_rate=0)

            # add mutations
            mts = msprime.sim_mutations(ts, rate=mutation_rates,
                                            model=self.config["substitution model"],
                                            random_seed=fragment_seeds[k])

            # get nucleoties
            array = mts.genotype_matrix().transpose()
            # concert to minor allele encoding
            encoded_array = np.empty_like(array)
            for i in range(array.shape[1]):
                unique_elements, counts = np.unique(array[array[:, i] != -1, i], return_counts=True)
                sorted_indices = np.argsort(counts)
                encoding_dict = {value: index for index, value in enumerate(sorted_indices)}
                for j, element in enumerate(unique_elements):
                    encoded_array[array[:, i] == element, i] = encoding_dict[j]
                encoded_array[array[:, i] == -1, i] = -1
            array = encoded_array
        
            # remove non-biallelic columns
            if array.shape[1] > 0:
                frequencies = np.array([[np.sum(array[:, j] == i) \
                    for i in range(0, np.max(array))] for j in range(array.shape[1])])
                nonbiallelic_columns = np.where(np.sum(frequencies != 0, axis=1) > 2)[0]
                array = np.delete(array, nonbiallelic_columns, axis=1)

                # organize matrix
                array = self.organize_matrix_user(array, retained_dict)

            parameter_arrays.append(array)

        # combine the parameter_ts arrays across fragments
        dataset_array = np.column_stack(parameter_arrays)

        return dataset_array, dataset_array.shape[1]
