"""This module contains the Class for processing empirical data."""
import logging
import itertools
from collections import Counter, OrderedDict
from itertools import product
import copy
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from popai.utils import minor_encoding
class DataProcessor:

    """Process empirical data."""

    def __init__(self, config):
        self.config = config

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # set seed
        self.rng = np.random.default_rng(self.config["seed"])

    def fasta_to_numpy(self):

        """Convert a list of fasta files into a numpy array."""

        # dictionary for encoding
        integer_encoding_dict = {"A": 0, "T": 1, "C": 2, "G": 3}

        # list for storing results
        encoded_alignments = []

        # iterate over populations and create ordered numpy array
        for population in self.config["sampling dict"].keys():

            # get all samples from that population/species
            samples_from_population = [key for key, value in \
                self.config["population dictionary"].items() if value == population]

            for item in samples_from_population:

                encoded_string = []

                for alignment in self.config["fastas"]:
                    if item in alignment:
                        encoded_string_current = np.array(self._encode_string(
                            string=str(alignment[item]), encoding_dict=integer_encoding_dict))
                    else:
                        encoded_string_current = [-1]*alignment.max_sequence_size

                    encoded_string.extend(encoded_string_current)

                encoded_alignments.append(encoded_string)

        # convert to numpy array
        encoded_alignments = np.array(encoded_alignments)

        # remove invariable columns
        frequencies = np.array([[np.sum(encoded_alignments[:, j] == i) \
            for i in range(0, 4)] for j in range(encoded_alignments.shape[1])])
        invariant_columns = np.where(np.sum(frequencies == 0, axis=1) >= 3)[0]
        filtered_alignments = np.delete(encoded_alignments, invariant_columns, axis=1)

        # concert to minor allele encoding
        encoded_array = minor_encoding(filtered_alignments)


        self.logger.info("""Empirical data has %s SNPs. If this is very different than the number of SNPs in your simulated data, you may want to change some priors.""",
                         encoded_array.shape[1])

        return encoded_array

    def vcf_to_numpy(self):

        """Convert VCF to numpy array."""

        # vcf to table
        start_index = [i for i, item in enumerate(self.config['vcf']) if 'CHROM' in item][0]
        relevant_lines = '\n'.join(self.config['vcf'][start_index:])
        vcf_table = pd.read_csv(io.StringIO(relevant_lines), sep="\t", index_col=False)

        # get DP index
        format_lines = [x for x in self.config['vcf'] if 'FORMAT' in x]
        dp_index = [i for i, item in enumerate(format_lines) if 'DP' in item][0]

        encoded_alignments = []
        for population in self.config['sampling dict'].keys():

            # get all samples for that poulation/species
            samples_from_population = [key for key, value in \
                self.config["original population dictionary"].items() if value == population]

            for item in samples_from_population:

                encoded_string_1 = []
                encoded_string_2 = []

                info = vcf_table[item]

                for row in info:
                    value1 = row.split("|")[0]
                    value2 = row.split("|")[1].split(":")[0]
                    dp = row.split(":")[dp_index]

                    if (value1 != "." or value2 != ".") and dp != '0':

                        encoded_string_1.append(int(value1))
                        encoded_string_2.append(int(value2))

                    else:
                        encoded_string_1.append(-1)
                        encoded_string_2.append(-1)

                encoded_alignments.append(encoded_string_1)
                encoded_alignments.append(encoded_string_2)

        encoded_alignments = np.array(encoded_alignments)

        # remove invariable columns
        frequencies = np.array([[np.sum(encoded_alignments[:, j] == i) \
            for i in range(0, 4)] for j in range(encoded_alignments.shape[1])])
        invariant_columns = np.where(np.sum(frequencies == 0, axis=1) >= 3)[0]
        filtered_alignments = np.delete(encoded_alignments, invariant_columns, axis=1)

        encoded_array = minor_encoding(filtered_alignments)


        self.logger.info("Empirical data has %s SNPs. If this is very different than the number of SNPs in your simulated data, you may want to change some priors.", encoded_array.shape[1])

        return encoded_array

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

        sampled = 0
        pop_failure_masks = []

        for population in self.config['sampling dict'].values():
            this_array = encoded_alignment[sampled:sampled+population,:]
            mask = this_array == -1
            count_minus_ones = np.sum(mask, axis=0)
            failure_masks = {population-j: (count_minus_ones <= j) \
                             for j in range(population)}
            pop_failure_masks.append(failure_masks)
            sampled += population
        all_combinations = itertools.product(*pop_failure_masks)
        results = {}
        for thresholds in all_combinations:
            if all(value % 2 == 0 for value in thresholds):
                thethresholds = [pop_failure_masks[j][thresholds[j]] \
                                 for j in range(len(pop_failure_masks))]
                combined_mask = np.logical_and.reduce(thethresholds)
                results[thresholds] = np.sum(combined_mask)
        results = {key: value for key, value in results.items() if value != 0}

        return results

    def numpy_to_2d_sfs(self, encoded_alignment, downsampling, replicates = 1):

        """Convert numpy array to 2d SFS."""

        # check that using even values
        key_even = all(value % 2 == 0 for value in downsampling.values())
        if not key_even:
            raise ValueError("Error in downampling, all keys must be even.")

        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in self.config["sampling dict"].items():
            sampling_indices[key] = [current, value + current]
            current = current+value

        # remove columns that do not meet thresholds
        sampled = 0
        pop_failure_masks = []
        for name, population in self.config["sampling dict"].items():
            this_array = encoded_alignment[sampled:sampled+population,:]
            mask = this_array==-1
            count_minus_ones = np.sum(mask, axis=0)
            failure_mask = count_minus_ones <= population - downsampling[name]
            pop_failure_masks.append(failure_mask)
            sampled += population
        combined_mask = np.logical_and.reduce(pop_failure_masks)
        filtered_encoded_alignment = encoded_alignment[:,combined_mask]

        # create empty sfs
        # iterate over each pair of populations
        populations = list(self.config["sampling dict"].keys())
        sfs_2d = {}
        sfs_list = []
        for i, pop1 in enumerate(populations):
            for j, pop2 in enumerate(populations):
                if i < j:
                    # create an empty 2D numpy array with the correct shape
                    array_shape = (downsampling[pop1]+1, downsampling[pop2]+1)
                    sfs_2d[(pop1, pop2)] = np.zeros(array_shape)

        # conver to list of sfs, one per replicate
        sfs_list = []
        for _ in range(replicates):
            sfs_list.append(copy.deepcopy(sfs_2d))

        for site in range(filtered_encoded_alignment.shape[1]): # iterate over sites
            site_data = list(filtered_encoded_alignment[:,site]) # get the data for that site

            if (len(set(site_data)) == 2 and -1 not in site_data) or \
                (len(set(site_data)) == 3 and -1 in site_data): # if it is > 1, keep going

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
                    if len(site_data_pop1) > downsampling[key[0]]:
                        site_data_pop1_sampled = [list(self.rng.choice(site_data_pop1, \
                            downsampling[key[0]])) for m in range(replicates)]
                    else:
                        site_data_pop1_sampled = [site_data_pop1]*replicates

                    if len(site_data_pop2) > downsampling[key[1]]:
                        site_data_pop2_sampled = [list(self.rng.choice(site_data_pop2, \
                            downsampling[key[1]])) for m in range(replicates)]
                    else:
                        site_data_pop2_sampled = [site_data_pop2]*replicates


                    for k in range(replicates):

                        all_site_data = site_data_pop1_sampled[k]+site_data_pop2_sampled[k]
                        #print(all_site_data)

                        # check that this site has two variants in these populations
                        if len(set(all_site_data)) == 2:
                            # get minor allele
                            counter = Counter(all_site_data)
                            min_count = min(counter.values())
                            least_common_numbers = [num for num, cnt in counter.items() if cnt == min_count]
                            minor_allele = self.rng.choice(least_common_numbers)
                            #print(minor_allele)
                            # find counts in each population
                            pop1_count = Counter(site_data_pop1_sampled[k])[minor_allele]
                            pop2_count = Counter(site_data_pop2_sampled[k])[minor_allele]
                            #print(pop1_count, pop2_count)
                            # add to the sfs
                            sfs_list[k][key][pop1_count, pop2_count] += 1


        return sfs_list

    def plot_2dsfs(self, sfs_list, output_directory):
        """Plot average 2 dimensional Site frequency spectra."""

        averages = {}
        for key in sfs_list[0].keys():
            arrays = [d[key] for d in sfs_list]
            average_array  = np.mean(arrays, axis=0)
            averages[key] = average_array
        # Create heatmaps
        for key, value in averages.items():
            outfile  = os.path.join(output_directory, f"2D_SFS_{key}_empirical.png")
            plt.imshow(value, cmap='viridis', origin="lower")
            plt.colorbar()  # Add colorbar to show scale
            plt.title(f"2D SFS {key}")
            plt.savefig(outfile)
            plt.close()

    def numpy_to_msfs(self, encoded_alignment, downsampling, replicates = 1, nbins=None):

        """Convert numpy array to multidimensional site frequency spectra"""

        # check that using even values
        key_even = all(value % 2 == 0 for value in downsampling.values())
        if not key_even:
            raise ValueError("Error in downampling, all keys must be even.")

        all_sfs = []


        # get indices for samples
        current = 0
        sampling_indices = {}
        for key, value in self.config["sampling dict"].items():
            sampling_indices[key] = [current, value + current]
            current = current+value
        reordered_downsampling = OrderedDict({key: downsampling[key] for key in self.config["sampling dict"]})

        # remove columns that do not meet thresholds
        sampled = 0
        pop_failure_masks = []
        for name, population in self.config["sampling dict"].items():
            this_array = encoded_alignment[sampled:sampled+population,:]
            mask = this_array==-1
            count_minus_ones = np.sum(mask, axis=0)
            failure_mask = count_minus_ones <= population - downsampling[name]
            pop_failure_masks.append(failure_mask)
            sampled += population
        combined_mask = np.logical_and.reduce(pop_failure_masks)
        filtered_encoded_alignment = encoded_alignment[:,combined_mask]

        for _ in range(replicates):

            # Generate all possible combinations of counts per population
            combos = product(*(range(count + 1) for count in reordered_downsampling.values()))
            rep_sfs_dict = OrderedDict({'_'.join(map(str, combo)): 0 for combo in combos})

            for site in range(filtered_encoded_alignment.shape[1]): # iterate over sites
                site_data = list(filtered_encoded_alignment[:,site]) # get the data for that site
                if (len(set(site_data)) == 2 and -1 not in site_data) \
                    or (len(set(site_data)) == 3 and -1 in site_data): # if it is > 1, keep going

                    # get minor allele
                    minor_allele = 1

                    # find poulation counts
                    counts_per_population = {}
                    for population in self.config['sampling dict'].keys():

                        site_data_pop = site_data[sampling_indices[population][0]:
                                                sampling_indices[population][1]]

                        if downsampling != self.config['sampling dict']:
                            site_data_pop = [x for x in site_data_pop if not x == -1]
                            site_data_pop = list(self.rng.choice(site_data_pop, \
                                downsampling[population]))

                        counts_per_population[population] = Counter(site_data_pop)[minor_allele]

                    string_for_count = [x for x in list(counts_per_population.values())]
                    if sum(string_for_count) != 0:
                        string_for_count = [str(x) for x in string_for_count]
                        combo_key = '_'.join(string_for_count)
                        rep_sfs_dict[combo_key]+=1

            # convert SFS to binned
            if not nbins is None:
                thresholds = []
                for value in reordered_downsampling.values():
                    thresholds.append([int(np.floor(value/nbins*(x+1))) for x in range(nbins)])
                threshold_combos = list(product(*thresholds))
                binned_rep_sfs_dict = OrderedDict({'_'.join(map(str, combo)): 0 for combo in threshold_combos})

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

            all_sfs.append(np.array(rep_sfs_dict))

            # calculate average number of sites used

        average_sites = int(np.ceil(np.mean([np.sum(x) for x in all_sfs])))
        print(f"We used an average of {average_sites} to construct the mSFS.")

        return(all_sfs, average_sites)
    