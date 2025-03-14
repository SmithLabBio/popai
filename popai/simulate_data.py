"""This module contains all Classes for simulating datasets under specified models using msprime."""
import logging
import time # for testing only
from collections import Counter, OrderedDict
from itertools import product
import os
import msprime
import numpy as np
import matplotlib.pyplot as plt
logging.getLogger('msprime').setLevel("WARNING")
import sys
import pyslim
import dendropy
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from popai.utils import minor_encoding
from tqdm import tqdm
import pickle
from collections import defaultdict

class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config, cores, downsampling, max_sites, output, user=False, sp_tree_index = False, checkpoint = False):
        self.models = models
        self.labels = labels
        self.config = config
        self.cores = cores
        self.downsampling = downsampling
        self.max_sites = max_sites
        self.user = user
        self.sp_tree_index = sp_tree_index
        self.output = output
        self.checkpoint = checkpoint

        # to prevent pickling issues
        if 'fastas' in self.config:
            del self.config['fastas']

        if user == False and sp_tree_index == False:
            raise ValueError("Error in simulation command. You must either provide a species tree index list (output when constructing models), or use user-specified models.")

        # check that using even values
        key_even = all(value % 2 == 0 for value in self.downsampling.values())
        if not key_even:
            raise ValueError("Error in downampling, all keys must be even.")

        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def simulate_ancestry(self):

        """Perform ancestry simulations with msprime"""

        start_time = time.time()  # Record the start time

        # dictionary for storing arrays and list for storing sizes.
        all_arrays = {}
        all_sizes = []
        
        for ix, demography in enumerate(self.models):

            if ix % 100 == 0:
                print(f"Beginning simulation {ix} of {len(self.models)}.")

            if self.user == True:
                matrix, sizes = self._simulate_demography_user(demography)
            else:
                matrix, sizes = self._simulate_demography(demography,  self.config['species tree'][self.sp_tree_index[ix]])

            if self.labels[ix] in all_arrays:
                all_arrays[self.labels[ix]].append(matrix)
            else:
                all_arrays[self.labels[ix]] = [matrix]
            all_sizes.append(sizes)

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time

        self.logger.info("Simulation execution time: %s seconds.", execution_time)

        # shorten arrays that are too short, and pad arrays that are too long.
        median_size = int(np.ceil(np.median(all_sizes)))

        self.logger.info("Median simulated data has %s SNPs."\
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

    def process_demography(self, ix, demography):
        if self.user:
            matrix, sizes = self._simulate_demography_user(demography)
        else:
            matrix, sizes = self._simulate_demography(demography, self.config['species tree'][self.sp_tree_index[ix]])

        return matrix, sizes

    def simulate_ancestry_parallel(self):

        """Perform ancestry simulations with msprime"""

        # split models by model
        grouped_demographies = defaultdict(list)

        for index, (label, demography) in enumerate(zip(self.labels, self.models)):
            grouped_demographies[label].append((index, demography))

        # list for storing sizes
        for key, value in grouped_demographies.items():

            # checkpoint
            if self.checkpoint and os.path.exists(os.path.join(self.output, 'simulated_arrays_%s.pickle' % str(key))):
                print("Output for model %s already exists. Skipping model." % str(key))
                continue

            start_time = time.time()  # Record the start time

            # dictionary for storing arrays and list for storing sizes.
            all_arrays = []
            all_sizes = []
    
            with ProcessPoolExecutor(max_workers=self.cores) as executor:
                # Submit tasks
                futures = {
                    executor.submit(self.process_demography, ix, demography): ix
                    for ix, demography in value
                }
    
                # Collect results
                for future in tqdm(as_completed(futures), total=len(value), desc=f"Processing simulations (Model {str(key)})"):
                    matrix, sizes = future.result()
                    all_arrays.append(matrix)
                    all_sizes.append(sizes)
    
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate the execution time
    
            self.logger.info("Simulation execution time: %s seconds.", execution_time)
    
            for matrix_ix in range(len(all_arrays)):
                matrix = all_arrays[matrix_ix]
                if len(matrix) > 0:
                    if matrix.shape[1] > self.max_sites:
                        all_arrays[matrix_ix] = matrix[:, :self.max_sites]
                    elif matrix.shape[1] < self.max_sites:
                        num_missing_columns = self.max_sites - matrix.shape[1]
                        missing_columns = np.full((matrix.shape[0], num_missing_columns), -1)
                        modified_matrix = np.concatenate((matrix, missing_columns), axis=1)
                        all_arrays[matrix_ix]  = modified_matrix
                else:
                    num_missing_columns = self.max_sites - 0
                    modified_matrix = np.full((sum(self.config["sampling dict"].values()),
                                               num_missing_columns), -1)
                    all_arrays[matrix_ix]  = modified_matrix
            with open(os.path.join(self.output, 'simulated_arrays_%s.pickle' % str(key)), 'wb') as f:
                pickle.dump(all_arrays, f)


            # shorten arrays that are too short, and pad arrays that are too long.
            median_size = int(np.ceil(np.median(sizes)))
    
            self.logger.info("Median simulated data has %s SNPs."\
                             " If this is very different than the number of SNPs in your empirical data, you may want to change some priors.", 
                             median_size)
            
            
            del all_arrays, all_sizes

    def mutations_to_sfs(self, numpy_array_dict, nbins=None):

        """Convert numpy arrays to multidimensional site frequency spectra"""

        all_sfs = {}

        for i in set(self.labels):
            
            # read in array
            with open(os.path.join(self.output, 'simulated_arrays_%s.pickle' % str(i)), 'rb') as f:
                arrays = pickle.load(f)

            # create sfs
            all_sfs[str(i)] = []

            # get indices for samples
            reordered_downsampling = {key: self.downsampling[key] \
                                  for key in self.config["sampling dict"]}

            current = 0
            ds_sampling_indices = {}
            for key, value in reordered_downsampling.items():
                ds_sampling_indices[key] = [current, value + current]
                current = current+value
            current = 0
            sampling_indices = {}
            for key, value in self.config["sampling dict"].items():
                sampling_indices[key] = [current, value + current]
                current = current+value



            for replicate in arrays:

                # sample from arrays
                population_arrays = []
                for key, value in self.config["sampling dict"].items():
                    current_array = replicate[sampling_indices[key][0]:sampling_indices[key][1]]
                    subsampled_array =  current_array[self.rng.choice(current_array.shape[0], ds_sampling_indices[key][1] - ds_sampling_indices[key][0], replace=False)]
                    population_arrays.append(subsampled_array)
                downsampled_array = np.vstack(population_arrays)

                # Generate all possible combinations of counts per population
                combos = product(*(range(count + 1) for count in reordered_downsampling.values()))
                rep_sfs_dict = OrderedDict({'_'.join(map(str, combo)): 0 for combo in combos})

                for site in range(downsampled_array.shape[1]):

                    site_data = list(downsampled_array[:,site])

                    if len(set(site_data)) == 2:

                        # get minor allele
                        minor_allele = min(set(site_data), key=site_data.count)
                        # find poulation counts
                        counts_per_population = {}
                        for population in self.config['sampling dict'].keys():
                            site_data_pop = site_data[ds_sampling_indices[population][0]:
                                                      ds_sampling_indices[population][1]]
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
                    binned_rep_sfs_dict = OrderedDict({'_'.join(map(str, combo)): \
                                           0 for combo in threshold_combos})

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
                all_sfs[str(i)].append(np.array(rep_sfs_dict))
            
            # write to file
            with open(os.path.join(self.output, 'simulated_mSFS_%s.pickle' % str(i)), 'wb') as f:
                pickle.dump(all_sfs[str(i)], f)


        return all_sfs

    def mutations_to_2d_sfs(self):
        """Translate simulated mutations into 2d site frequency spectra"""

        all_sfs = {}


        for i in set(self.labels):
            # read in array
            with open(os.path.join(self.output, 'simulated_arrays_%s.pickle' % str(i)), 'rb') as f:
                arrays = pickle.load(f)

            # create sfs
            all_sfs[str(i)] = []
            all_arrays = [] 

            # get indices for samples
            reordered_downsampling = {key: self.downsampling[key] \
                                  for key in self.config["sampling dict"]}

            current = 0
            ds_sampling_indices = {}
            for key, value in reordered_downsampling.items():
                ds_sampling_indices[key] = [current, value + current]
                current = current+value

            current = 0
            sampling_indices = {}
            for key, value in self.config["sampling dict"].items():
                sampling_indices[key] = [current, value + current]
                current = current+value

            for replicate in arrays:

                # sample from arrays
                population_arrays = []
                for key, value in self.config["sampling dict"].items():
                    current_array = replicate[sampling_indices[key][0]:sampling_indices[key][1]]
                    subsampled_array =  current_array[self.rng.choice(current_array.shape[0], ds_sampling_indices[key][1] - ds_sampling_indices[key][0], replace=False)]
                    population_arrays.append(subsampled_array)
                downsampled_array = np.vstack(population_arrays)


                # generate the empty arrays
                sfs_2d = self._create_numpy_2d_arrays()

                for site in range(downsampled_array.shape[1]): # iterate over sites

                    site_data = list(downsampled_array[:,site]) # get the data for that site

                    if len(set(site_data)) > 1: # if it is > 1, keep going

                        for key, value in sfs_2d.items(): # iterate over population pairs

                            # get the site data for those two populations
                            site_data_pop1 = site_data[ds_sampling_indices[key[0]][0]:
                                                   ds_sampling_indices[key[0]][1]]
                            site_data_pop2 = site_data[ds_sampling_indices[key[1]][0]:
                                                   ds_sampling_indices[key[1]][1]]
                            all_site_data = site_data_pop1+site_data_pop2

                            # check that this site has two variants in these populations
                            if len(set(all_site_data)) == 2:

                                # get minor allele
                                counter = Counter(all_site_data)
                                min_count = min(counter.values())
                                least_common_numbers = [num for num, cnt in counter.items() if cnt == min_count]
                                minor_allele = self.rng.choice(least_common_numbers)

                                
                                # find counts in each population
                                pop1_count = Counter(site_data_pop1)[minor_allele]
                                pop2_count = Counter(site_data_pop2)[minor_allele]

                                # add to the sfs
                                sfs_2d[key][pop1_count, pop2_count] += 1

                all_sfs[str(i)].append(sfs_2d)

                # Convert dictionary to array for storage
                arrays = []
                for sfs in sfs_2d.values(): 
                    arrays.append(sfs)
                    print(sfs.shape)
                all_arrays.append(np.array(arrays, dtype=object))

            
            # write to file
            with open(os.path.join(self.output, 'simulated_2dSFS_%s.pickle' % str(i)), 'wb') as f:
                # pickle.dump(all_sfs[str(i)], f)
                pickle.dump(all_arrays, f)

        return all_sfs

    def plot_2dsfs(self, sfs_list, output_directory=None):
        """Plot average 2 dimensional Site frequency spectra."""

        for item in sfs_list.keys():
            
            average_sfs = {}
            
            for model, replicates in sfs_list.items():
                accumulation = {}
                counts = {}
                for replicate in replicates:
                    for comparison, sfs in replicate.items():
                        if comparison not in accumulation:
                            accumulation[comparison] = np.zeros_like(sfs)
                            counts[comparison] = 0
                        accumulation[comparison] += sfs
                        counts[comparison] += 1
                
                average_sfs[model] = {}
                for comparison, total_sfs in accumulation.items():
                    average_sfs[model][comparison] = total_sfs / counts[comparison]

        if output_directory==None:
            for model, comparisons in average_sfs.items():
                for comparison, avg_sfs in comparisons.items():
                    plt.imshow(avg_sfs, cmap='viridis', origin="lower")
                    plt.colorbar()
                    plt.title(f'{model} - {comparison}')
                    plt.show()
        else:
            for model, comparisons in average_sfs.items():
                for comparison, avg_sfs in comparisons.items():
                    outfile  = os.path.join(output_directory, \
                                            f"2D_SFS_{comparison}_{model}.png")
                    plt.imshow(avg_sfs, cmap='viridis', origin="lower")
                    plt.colorbar()
                    plt.title(f'{model} - {comparison}')
                    plt.savefig(outfile)
                    plt.close()
        

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

    def _organize_matrix(self, array_dict, simulating_dict, sp_tree):

        ordered_array_dict = OrderedDict()
        for key,value in self.config["sampling dict"].items():
            # check if key in array dict
            if key in array_dict.keys():
                ordered_array_dict[key] = array_dict[key]
                if ordered_array_dict[key].shape[0] == 0:
                    sys.exit()
            else:
                # find match
                found = False
                searchvalue = key
                while found == False:
                    for node in sp_tree.preorder_node_iter():
                        if not node.label == None:
                            spname = node.label
                        else:
                            spname = node.taxon.label
                        if spname == searchvalue:
                            parent = node.parent_node.label
                            if parent in array_dict.keys():
                                ordered_array_dict[key] = array_dict[parent][0:self.config['sampling dict'][key],:]
                                array_dict[parent] =  array_dict[parent][self.config['sampling dict'][key]:,:]
                                found = True
                            else:
                                searchvalue=parent


        array_list = []
        for key,value in ordered_array_dict.items():
            array_list.append(value)                            

        array = np.vstack(array_list)
        return(array)

    def _organize_matrix_user(self, array_dict, simulating_dict, demography):


        ordered_array_dict = OrderedDict()
        for key,value in self.config["sampling dict"].items():
            
            # check if key in array dict
            if key in array_dict.keys():
                ordered_array_dict[key] = array_dict[key]
                if ordered_array_dict[key].shape[0] == 0:
                    sys.exit()
            else:
                # find match
                found = False
                searchvalue = key
                while found == False:
                    for event in demography.events:
                        if hasattr(event, 'ancestral'):
                            if searchvalue in event.derived:
                                parent = event.ancestral
                                if parent in array_dict.keys():
                                    ordered_array_dict[key] = array_dict[parent][0:self.config['sampling dict'][key],:]
                                    array_dict[parent] =  array_dict[parent][self.config['sampling dict'][key]:,:]
                                    found = True
                                else:
                                    searchvalue=parent


        array_list = []
        for key,value in ordered_array_dict.items():
            array_list.append(value)                            

        array = np.vstack(array_list)

        return(array)

    def _get_simulating_dict(self, tree):
        simulating_dict = OrderedDict()
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

    def _simulate_demography(self, demography, tree):

        # get dictionary for simulations
        simulating_dict = self._get_simulating_dict_model(demography=demography, tree=tree)

        # keys mapping pops to ids
        id_map = OrderedDict()
        for key in simulating_dict:
            for population in demography.populations:
                id_map[population.name]=population.id 

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
            array_dict = OrderedDict()
            for key in simulating_dict.keys():
                array_dict[key] = mts.genotype_matrix(samples=mts.samples(id_map[key])).transpose()

            # organize matrix
            array = self._organize_matrix(array_dict, simulating_dict, sp_tree=tree)
            array = minor_encoding(array)

            parameter_arrays.append(array)

        # combine the parameter_ts arrays across fragments
        dataset_array = np.column_stack(parameter_arrays)

        return dataset_array, dataset_array.shape[1]

    def _simulate_demography_user(self, demography):

        # get dictionary for simulations
        simulating_dict = self._get_simulating_dict_demo(demography=demography)

        # keys mapping pops to ids
        id_map = OrderedDict()
        for key in simulating_dict:
            for population in demography.populations:
                id_map[population.name]=population.id 

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

            # get array
            array_dict = OrderedDict()
            for key in simulating_dict.keys():
                array_dict[key] = mts.genotype_matrix(samples=mts.samples(id_map[key])).transpose()

            ## organize matrix
            array = self._organize_matrix_user(array_dict, simulating_dict, demography)
            array = minor_encoding(array)


            parameter_arrays.append(array)

        # combine the parameter_ts arrays across fragments
        dataset_array = np.column_stack(parameter_arrays)

        return dataset_array, dataset_array.shape[1]

    def _get_simulating_dict_demo(self, demography):
        # get sampling dict
        this_sampling_dict = OrderedDict()
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
            
            # check to see if ancestor is in sampled_inactive
            for event in demography.events:
                if hasattr(event, 'ancestral'):
                    if population in event.derived:
                        this_ancestor = event.ancestral
            
            
            
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
            

            if this_ancestor == population or this_ancestor not in sampled_inactive:
                this_sampling_dict[population] = 0
                for item in relevant_descendants:         
                    this_sampling_dict[population] += self.config['sampling dict'][item]
                all_relevant_descendents.extend(relevant_descendants)
        

        revised_sampling_dictionary = OrderedDict()

        initially_active = [x for x in initially_active if not x in all_relevant_descendents]

        for population in initially_active:
            this_sampling_dict[population] = self.config['sampling dict'][population]

        for key,value in this_sampling_dict.items():
            if key not in all_descendents:
                if value % 2 != 0:
                    raise Exception("Remember we simulate diploid individuals. If you have an odd number of samples, something has gone wrong.")
                revised_sampling_dictionary[key] = value // 2
        

        return(revised_sampling_dictionary)

    def _get_simulating_dict_model(self, demography, tree):

        # figure out how to sample individuals
        simulating_dict = OrderedDict()
        populations = [x.name for x in demography.populations]
        simulating_dict = {population: 0 for population in populations}
        for species in tree.leaf_nodes():
            if species.taxon.label not in populations:
                search = True
                searchnode = species
                while search:
                    if searchnode.parent_node.label in populations:
                        simulating_dict[searchnode.parent_node.label] += \
                            self.config['sampling dict'][species.taxon.label] / 2
                        search = False
                    else:
                        searchnode = searchnode.parent_node
            else:
                simulating_dict[species.taxon.label] += \
                    self.config['sampling dict'][species.taxon.label]/2
        simulating_dict = OrderedDict({key: value for key, value in simulating_dict.items() if value != 0})
        return(simulating_dict)
