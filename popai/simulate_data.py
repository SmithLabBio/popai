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
from popai.utils import minor_encoding, sort_haplotypes
from tqdm import tqdm
import pickle
from collections import defaultdict
from popai.process_empirical import _apply_maxsites

class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config, cores, downsampling, max_sites=None, output='output', user=False, sp_tree_index = False, checkpoint = False, infinite = False):
        self.models = models
        self.labels = labels
        self.config = config
        self.cores = cores
        self.downsampling = downsampling
        self.max_sites = np.inf if max_sites is None else max_sites   # None: keep every SNP
        self.user = user
        self.sp_tree_index = sp_tree_index
        self.output = output
        self.checkpoint = checkpoint
        self.infinite = infinite
        if self.infinite:
            self.max_sites = len(self.config['lengths'])

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

            matrix, sizes = self.process_demography(ix, demography, self._draw_random_values())

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
                all_arrays[model][i] = _apply_maxsites(matrix, self.max_sites)

        return all_arrays

    def process_demography(self, ix, demography, draws):
        tree = None if self.user else self.config['species tree'][self.sp_tree_index[ix]]
        return self._simulate_demography(demography, tree, draws)

    def _draw_random_values(self):
        """Draw the random values for one replicate: its mutation rate, and the seeds for its
        ancestry and mutation simulations. These must be drawn in the main process: each worker
        process gets its own copy of self.rng, in the same state, so draws made inside the
        workers would be the same for every replicate."""
        n_loci = len(self.config['lengths'])
        draws = {
            'mutation rate': self.rng.uniform(low=self.config["mutation rate"][0],
                                              high=self.config["mutation rate"][1]),
            'ancestry seed': self.rng.integers(1, 2**32),
            'mutation seeds': self.rng.integers(1, 2**32, size=n_loci),
        }
        if self.infinite:
            draws['infinite seeds'] = self.rng.integers(1, 2**32, size=n_loci * 10)
        return draws

    def simulate_ancestry_parallel(self):

        """Perform ancestry simulations with msprime"""

        # split models by model
        grouped_demographies = defaultdict(list)

        for index, (label, demography) in enumerate(zip(self.labels, self.models)):
            grouped_demographies[label].append((index, demography))

        # list for storing sizes
        for key, value in grouped_demographies.items():

            # draw each replicate's random values here, in the main process (see _draw_random_values).
            # They are drawn even for models skipped by the checkpoint, so later models get the same values.
            draws = [self._draw_random_values() for _ in value]

            # checkpoint
            if self.checkpoint and os.path.exists(os.path.join(self.output, 'simulated_arrays_%s.pickle' % str(key))):
                print("Output for model %s already exists. Skipping model." % str(key))
                continue

            start_time = time.time()  # Record the start time

            # lists for storing arrays and sizes, in the order of the replicates
            all_arrays = [None] * len(value)
            all_sizes = [None] * len(value)

            with ProcessPoolExecutor(max_workers=self.cores) as executor:
                # Submit tasks, remembering each one's position in the list of replicates
                futures = {
                    executor.submit(self.process_demography, ix, demography, draw): position
                    for position, ((ix, demography), draw) in enumerate(zip(value, draws))
                }

                # Collect results as they finish, putting each in its replicate's position
                for future in tqdm(as_completed(futures), total=len(value), desc=f"Processing simulations (Model {str(key)})"):
                    position = futures[future]
                    all_arrays[position], all_sizes[position] = future.result()

            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate the execution time
    
            self.logger.info("Simulation execution time: %s seconds.", execution_time)
    
            for matrix_ix in range(len(all_arrays)):
                # keep the first max_sites SNPs, or pad with missing data (-1), as for empirical data
                modified_matrix = _apply_maxsites(all_arrays[matrix_ix], self.max_sites)

                # sort the rows within each population (the empirical array is sorted the same way)
                all_arrays[matrix_ix] = sort_haplotypes(modified_matrix, self.config["sampling dict"].values())

            with open(os.path.join(self.output, 'simulated_arrays_%s.pickle' % str(key)), 'wb') as f:
                pickle.dump(all_arrays, f)


            # shorten arrays that are too short, and pad arrays that are too long.
            median_size = int(np.ceil(np.median(all_sizes)))

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
                replicate = replicate[:, ~(replicate >= 2).any(axis=0)]
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
                replicate = replicate[:, ~(replicate >= 2).any(axis=0)]
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

    def _sample_sets(self, demography, tree):
        """One msprime SampleSet per population in the sampling dict, in that order, with one
        diploid individual per two haplotypes. Each population's individuals come from the
        population itself or, if it has merged with others, from the population it merged into.
        Because the sample sets are in sampling-dict order, the rows of the genotype matrix
        (two per individual) are too."""

        names = {population.name for population in demography.populations}
        merged_into = {derived: event.ancestral for event in demography.events
                       if hasattr(event, 'derived') and event.time == 0 for derived in event.derived}

        sample_sets = []
        for population, size in self.config['sampling dict'].items():
            source = population
            # species tree models leave out populations that are merged: use the closest ancestor in the model
            if source not in names and tree is not None:
                node = tree.find_node_with_taxon_label(source)
                while node is not None and (node.taxon.label if node.is_leaf() else node.label) not in names:
                    node = node.parent_node
                if node is not None:
                    source = node.taxon.label if node.is_leaf() else node.label
            # user models merge populations with splits at time 0: use the population they merge into
            while source in merged_into:
                source = merged_into[source]
            if source not in names:
                raise ValueError(f"The population {population} in the popfile isn't in the model "
                                 f"(populations: {', '.join(sorted(names))}).")
            if size % 2:
                raise ValueError(f"Population {population} has {size} haplotypes. popai simulates diploid "
                                 f"individuals, so every population needs an even number of haplotypes.")
            sample_sets.append(msprime.SampleSet(size // 2, population=source, ploidy=2))
        return sample_sets

    def _simulate_demography(self, demography, tree, draws):
        """Simulate one replicate: its loci, with the haplotypes (rows) in sampling-dict order.
        tree is the replicate's species tree, or None for user models."""

        sample_sets = self._sample_sets(demography, tree)
        n_loci = len(self.config['lengths'])

        # use median length instead
        median_length = int(np.median(self.config['lengths']))

        # list for storing arrays from this parameterization
        parameter_arrays = []

        if self.infinite:

            num_sites=0
            mts=[]
            while num_sites < n_loci:
                ts_iterator = msprime.sim_ancestry(sample_sets, demography=demography, random_seed = draws['ancestry seed'], discrete_genome=False, num_replicates=n_loci*10)

                for k, ts in enumerate(ts_iterator):
                    if num_sites < n_loci:

                        # add mutations
                        mt = msprime.sim_mutations(ts, rate=draws['mutation rate'],
                                                            discrete_genome=False,
                                                            model=msprime.BinaryMutationModel(),
                                                            random_seed = draws['infinite seeds'][k])
                        if mt.num_mutations > 0:
                            mts.append(mt)
                            num_sites+=1

                    else:
                        break

            # the first site of each locus
            array = np.concatenate([mts_item.genotype_matrix().transpose()[:, [0]] for mts_item in mts], axis=1)
            parameter_arrays.append(minor_encoding(array))

        else:
            ts_iterator = msprime.sim_ancestry(sample_sets, demography=demography, random_seed = draws['ancestry seed'], sequence_length=median_length, recombination_rate=0, num_replicates=n_loci)
            for k, ts in enumerate(ts_iterator):

                # add mutations
                mts = msprime.sim_mutations(ts, rate=draws['mutation rate'],
                                            model=self.config["substitution model"],
                                            random_seed=draws['mutation seeds'][k])

                # rows are haplotypes, in sampling-dict order; columns are sites
                parameter_arrays.append(minor_encoding(mts.genotype_matrix().transpose()))

        # combine the parameter_ts arrays across fragments
        dataset_array = np.column_stack(parameter_arrays)

        return dataset_array, dataset_array.shape[1]