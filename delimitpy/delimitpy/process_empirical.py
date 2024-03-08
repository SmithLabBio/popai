"""This module contains the Class for processing empirical data."""
import logging
import dendropy
import numpy as np
import itertools
from collections import Counter
from itertools import product

class DataProcessor:

    """Simulate data under specified demographies."""

    def __init__(self, models, config):
        self.models = models
        self.config = config

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # set seed
        self.rng = np.random.default_rng(self.config['seed'])

    def fasta_to_numpy(self):
        
        """Convert a list of fasta files into a numpy array."""
        integer_encoding_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

        # concat the fasta files
        fastas_all = dendropy.DnaCharacterMatrix.concatenate(self.config['fastas'])

        # get order in which to process individuals
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}

        # list for storing results
        encoded_alignments = []

        # iterate over populations and create ordered numpy array
        for population in reordered_dict.keys():

            # get all samples from that population/species
            samples_from_population = [key for key, value in self.config["population_dictionary"].items() if value == population]
            
            for item in samples_from_population:
                encoded_string = np.array(self._encode_string(string=str(fastas_all[item]), encoding_dict=integer_encoding_dict))
                encoded_alignments.append(encoded_string)

        # convert to numpy array
        encoded_alignments = np.array(encoded_alignments)

        # remove invariable columns
        frequencies = np.array([[np.sum(encoded_alignments[:, j] == i) for i in range(0, 4)] for j in range(encoded_alignments.shape[1])])
        invariant_columns = np.where(np.sum(frequencies == 0, axis=1) >= 3)[0]
        filtered_alignments = np.delete(encoded_alignments, invariant_columns, axis=1)
        
        ## for testing randomly replace some values with -1
        #encoded_alignment = self._randomly_replace(array=filtered_alignments, percentage=0.2)
        #print('ERROR MAKE SURE YOU REMOVE RANDOM MISSING CODE PLEASE.')

        #return(filtered_alignments, encoded_alignment)
        return(filtered_alignments)

    def _encode_string(self, string, encoding_dict):
        encoded_string = []
        for char in string:
            if char in encoding_dict:
                encoded_string.append(encoding_dict[char])
            else:
                encoded_string.append(-1)
        return encoded_string

    def find_downsampling(self, encoded_alignment):
        """This funciton will convert an empirical alignment to a site frequency spectrum.
        It needs to deal with missing data in an intelligent way (e.g., downsampling)."""
        
        # get order in which to process individuals
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}

        sampled = 0
        pop_failure_masks = []
        
        for population in reordered_dict.values():
            this_array = encoded_alignment[sampled:sampled+population,:]
            mask = (this_array == -1)
            count_minus_ones = np.sum(mask, axis=0)
            failure_masks = {j: (count_minus_ones <= (population-j)) for j in range(1, population+1)}
            pop_failure_masks.append(failure_masks)
            sampled += population
        all_combinations = itertools.product(*pop_failure_masks)
        results = {}
        for thresholds in all_combinations:
            if all(value % 2 == 0 for value in thresholds):
                thethresholds = [pop_failure_masks[j][thresholds[j]] for j in range(len(pop_failure_masks))]
                combined_mask = np.logical_and.reduce(thethresholds)
                results[thresholds] = np.sum(combined_mask)
        results = {key: value for key, value in results.items() if value != 0}
        
        return results

    def numpy_to_2d_sfs(self, encoded_alignment, downsampling, replicates = 1):

        # get order in which to process individuals
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}

        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in reordered_dict.items():
            sampling_indices[key] = [current, value + current]
            current = current+value

        sampled = 0
        pop_failure_masks = []
        # remove columns that do not meet thresholds
        for name, population in reordered_dict.items():
            this_array = encoded_alignment[sampled:sampled+population,:]
            mask = (this_array==-1)
            count_minus_ones = np.sum(mask, axis=0)
            failure_mask = count_minus_ones<=downsampling[name]
            pop_failure_masks.append(failure_mask)
            sampled += population
        combined_mask = np.logical_and.reduce(pop_failure_masks)
        filtered_encoded_alignment = encoded_alignment[:,combined_mask]

        # create empty sfs
        # iterate over each pair of populations
        populations = list(reordered_dict.keys())
        sfs_2d = {}
        sfs_list = []
        for i in range(len(populations)):
            for j in range(i+1, len(populations)):
                pop1 = populations[i]
                pop2 = populations[j]

                # create an empty 2D numpy array with the correct shape
                array_shape = (downsampling[pop1]+1, downsampling[pop2]+1)
                sfs_2d[(pop1, pop2)] = np.zeros(array_shape)

        # conver to list of sfs, one per replicate
        sfs_list += [sfs_2d] * replicates

        for site in range(filtered_encoded_alignment.shape[1]): # iterate over sites
            site_data = list(filtered_encoded_alignment[:,site]) # get the data for that site
            if (len(set(site_data)) == 2 and -1 not in site_data) or (len(set(site_data)) == 3 and -1 in site_data): # if it is > 1, keep going
                for key, value in sfs_2d.items(): # iterate over population pairs
                    # get the site data for those two populations
                    site_data_pop1 = site_data[sampling_indices[key[0]][0]:
                                               sampling_indices[key[0]][1]]
                    site_data_pop2 = site_data[sampling_indices[key[1]][0]:
                                               sampling_indices[key[1]][1]]
                    
                    # remove negative ones
                    site_data_pop1 = [x for x in site_data_pop1 if not x == -1]
                    site_data_pop2 = [x for x in site_data_pop2 if not x == -1]

                    # sample random sites
                    site_data_pop1_sampled = [list(self.rng.choice(site_data_pop1, downsampling[key[0]])) for m in range(replicates)]
                    site_data_pop2_sampled = [list(self.rng.choice(site_data_pop2, downsampling[key[1]])) for m in range(replicates)]
                    
                    for k in range(replicates):
                        all_site_data = site_data_pop1_sampled[k]+site_data_pop2_sampled[k]

                        # check that this site has two variants in these populations
                        if len(set(all_site_data)) == 2:
                            # get minor allele
                            minor_allele = min(set(all_site_data), key=all_site_data.count)
                            # find counts in each population
                            pop1_count = Counter(site_data_pop1_sampled[k])[minor_allele]
                            pop2_count = Counter(site_data_pop2_sampled[k])[minor_allele]
                            # add to the sfs
                            sfs_list[k][key][pop1_count, pop2_count] += 1
                        # convert SFS to binned
        
        return sfs_list

    def numpy_to_msfs(self, encoded_alignment, downsampling, replicates = 1, nbins=None):
        """Translate simulated mutations into site frequency spectra"""

        all_sfs = []

        # change the order of the sampling dictionary to match the population order in the models
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}
        downsampling_reordered = {key: downsampling[key] for key in population_order}

        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in reordered_dict.items():
            sampling_indices[key] = [current, value + current]
            current = current+value

        sampled = 0
        pop_failure_masks = []
        # remove columns that do not meet thresholds
        for name, population in reordered_dict.items():
            this_array = encoded_alignment[sampled:sampled+population,:]
            mask = (this_array==-1)
            count_minus_ones = np.sum(mask, axis=0)
            failure_mask = count_minus_ones<=downsampling[name]
            pop_failure_masks.append(failure_mask)
            sampled += population
        combined_mask = np.logical_and.reduce(pop_failure_masks)
        filtered_encoded_alignment = encoded_alignment[:,combined_mask]

        for rep in range(replicates):

            # Generate all possible combinations of counts per population
            combos = product(*(range(count + 1) for count in downsampling_reordered.values()))
            rep_sfs_dict = {'_'.join(map(str, combo)): 0 for combo in combos}

            for site in range(filtered_encoded_alignment.shape[1]): # iterate over sites
                site_data = list(filtered_encoded_alignment[:,site]) # get the data for that site
                if (len(set(site_data)) == 2 and -1 not in site_data) or (len(set(site_data)) == 3 and -1 in site_data): # if it is > 1, keep going

                    # get minor allele
                    minor_allele = min(set(site_data), key=site_data.count)

                    # find poulation counts
                    counts_per_population = {}
                    for population in reordered_dict.keys():
                        site_data_pop = site_data[sampling_indices[population][0]:
                                                sampling_indices[population][1]]
                        site_data_pop = [x for x in site_data_pop if not x == -1]
                        site_data_pop = list(self.rng.choice(site_data_pop, downsampling[population]))\
                        
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

            all_sfs.append(rep_sfs_dict)


        return all_sfs

    def calc_sumstats(self):
        """Calculate summary statistics."""

        # get order in which to process individuals
        population_order = [x.name for x in self.models[-1][0].populations if
                            x.default_sampling_time is None]
        reordered_dict = {key: self.config["sampling_dict"][key] for key in population_order}


        fastas_all = dendropy.DnaCharacterMatrix.concatenate(self.config['fastas'])

        statistics = []

        # nucleotide diversity
        statistics.append(dendropy.calculate.popgenstat.nucleotide_diversity(fastas_all))

        # segregating sites
        statistics.append(dendropy.calculate.popgenstat.num_segregating_sites(fastas_all))

        # iterate over populations and calculate diversity and the number of segregating sites
        for population in reordered_dict.keys():


            # get all samples from that population/species
            samples_from_population = [key for key, value in self.config["population_dictionary"].items() if value == population]

            # compute pi
            statistics.append(self._nucleotide_diversity(fastas_all, samples_from_population))

            # compute S
            statistics.append(self._segregating_sites(fastas_all, samples_from_population))

        processed=[]
        for population in reordered_dict.keys():
            for population2 in reordered_dict.keys():
                if population != population2 and (population2, population)\
                    not in processed:

                    samples_from_population_1 = [key for key, value in self.config["population_dictionary"].items() if value == population]
                    samples_from_population_2 = [key for key, value in self.config["population_dictionary"].items() if value == population2]

                    p1 = []
                    p2 = []
                    for idx, t in enumerate(fastas_all.taxon_namespace):
                        if t.label in samples_from_population_1:
                            p1.append(fastas_all[t])
                        elif t.label in samples_from_population_2:
                            p2.append(fastas_all[t])

                    pp = dendropy.calculate.popgenstat.PopulationPairSummaryStatistics(p1, p2)

                    # add dxy
                    statistics.append(pp.average_number_of_pairwise_differences_between/fastas_all.sequence_size)

                    processed.append((population, population2))

        return statistics

    def _randomly_replace(self, array, percentage):
        mask = self.rng.choice([True, False], size=array.shape, p=[percentage, 1-percentage])
        array[mask] = -1
        return array
    
    def _nucleotide_diversity(self, alignment, samples):

        # Convert the alignment to a list of sequences
        sequences = []
        for item in samples:
            sequences.append(str(alignment[item]))

        # Get the length of the alignment
        alignment_length = len(sequences[0])

        # Initialize the pairwise differences matrix
        pairwise_diff = np.zeros((len(sequences), len(sequences)))

        # Compute pairwise differences
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                for a,b in zip(sequences[i], sequences[j]):
                    if a in ['A', 'T', 'C', 'G'] and b in ['A', 'T', 'C', 'G']:
                        if a != b:
                            pairwise_diff[i, j] += 1
                            pairwise_diff[j, i] += 1

        # Calculate nucleotide diversity (pi)
        pi = np.sum(pairwise_diff) / (len(sequences)* (len(sequences)-1) * alignment_length)

        return pi

    def _segregating_sites(self, alignment, samples):
        alignment_length = alignment.sequence_size
        segregating_sites = 0
        for position in range(alignment_length):
            nucleotides = [str(alignment[x][position]) for x in samples]
            nucleotides = [x for x in nucleotides if x in ['A', 'T', 'C', 'G']]
            if len(set(nucleotides)) > 1:
                segregating_sites += 1
        return segregating_sites