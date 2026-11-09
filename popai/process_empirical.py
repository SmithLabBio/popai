"""This module contains the Class for processing empirical data."""
import logging
from itertools import product
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

    def fasta_to_numpy(self, maxsites):

        """Convert a list of fasta files into a numpy array."""

        # bases are encoded A=0, T=1, C=2, G=3; anything else (N, gaps, ambiguity codes) is -1
        lookup = np.full(256, -1, dtype=np.int64)
        for base, code in {"A": 0, "T": 1, "C": 2, "G": 3}.items():
            lookup[ord(base)] = code

        # one row per sample, ordered by population; a sample absent from an alignment
        # is missing data for that alignment
        rows = []
        for sample in self._samples_in_order("population dictionary"):
            row = []
            for alignment in self.config["fastas"]:
                if sample in alignment:
                    row.append(lookup[np.frombuffer(str(alignment[sample]).encode(), dtype=np.uint8)])
                else:
                    row.append(np.full(alignment.max_sequence_size, -1, dtype=np.int64))
            rows.append(np.concatenate(row))

        return self._finish_array(np.array(rows), maxsites)

    def vcf_to_numpy(self, maxsites):

        """Convert VCF to numpy array."""

        # vcf to table
        start_index = [i for i, item in enumerate(self.config['vcf']) if item.startswith('#CHROM')][0]
        relevant_lines = '\n'.join(self.config['vcf'][start_index:])
        vcf_table = pd.read_csv(io.StringIO(relevant_lines), sep="\t", index_col=False, dtype=str)
        samples = self._samples_in_order("original population dictionary")
        n_sites = len(vcf_table)

        # split every genotype into its colon-separated fields (sites x samples, flattened;
        # an empty cell counts as missing)
        cells = [cell.split(":") if isinstance(cell, str) else ["./."]
                 for cell in vcf_table[samples].to_numpy().ravel()]

        # read depth, from DP's position in each site's own FORMAT field (trailing fields
        # may be left off, and some sites may not record DP at all)
        dp_positions = [fields.index('DP') if 'DP' in fields else -1
                        for fields in vcf_table['FORMAT'].astype(str).str.split(':')]
        dp_position = np.repeat(dp_positions, len(samples)).tolist()
        no_reads = np.fromiter((0 <= dp < len(fields) and fields[dp] == '0'
                                for fields, dp in zip(cells, dp_position)),
                               dtype=bool, count=len(cells))

        # alleles, parsed once for each distinct genotype (0/1, 1|1, ./. ...)
        codes, genotypes = pd.factorize(np.array([fields[0] for fields in cells], dtype=object))
        alleles = np.empty((len(genotypes), 2), dtype=np.int64)
        for ix, genotype in enumerate(genotypes):
            parts = genotype.replace("|", "/").split("/")
            if len(parts) != 2:
                sample = samples[np.nonzero(codes == ix)[0][0] % len(samples)]
                raise ValueError(f"Error: popai expects diploid genotypes, but sample "
                                 f"{sample} has genotype '{genotype}'.")
            alleles[ix] = [-1 if allele == "." else int(allele) for allele in parts]
        haplotypes = alleles[codes]                       # (sites * samples) x 2
        haplotypes[no_reads] = -1                         # a genotype with no reads is missing

        # rows: each sample's two haplotypes, samples ordered by population
        encoded = haplotypes.reshape(n_sites, len(samples), 2).transpose(1, 2, 0)
        return self._finish_array(encoded.reshape(2 * len(samples), n_sites), maxsites)

    def find_downsampling(self, encoded_alignment):
        """For each combination of even sample sizes (one per population), count the sites
        with at least that many called alleles in every population."""

        sizes = list(self.config['sampling dict'].values())
        pairs = [(block != -1).sum(axis=0) // 2 for block in self._population_blocks(encoded_alignment)]

        # sites_with[k1, k2, ...]: number of sites with at least 2*k1 called alleles in the
        # first population, 2*k2 in the second, and so on
        sites_with = np.zeros([n // 2 + 1 for n in sizes], dtype=np.int64)
        np.add.at(sites_with, tuple(pairs), 1)
        for axis in range(len(sizes)):
            sites_with = np.flip(np.cumsum(np.flip(sites_with, axis), axis=axis), axis)

        results = {}
        for thresholds in product(*(range(n - n % 2, 0, -2) for n in sizes)):
            count = sites_with[tuple(t // 2 for t in thresholds)]
            if count != 0:
                results[thresholds] = count
        return results

    def numpy_to_2d_sfs(self, encoded_alignment, downsampling, replicates = 1):

        """Convert numpy array to 2d SFS."""

        _check_even(downsampling)
        populations = list(self.config["sampling dict"].keys())
        pairs = [(pop1, pop2) for i, pop1 in enumerate(populations)
                 for pop2 in populations[i + 1:]]
        sites = self._usable_sites(encoded_alignment, downsampling)

        sfs_list = []
        for _ in range(replicates):
            # downsample each population once per replicate; count the higher-coded allele
            counts = self._downsampled_counts(sites, downsampling)
            sfs_2d = {}
            for pop1, pop2 in pairs:
                total = downsampling[pop1] + downsampling[pop2]
                count1, count2 = counts[pop1], counts[pop2]
                both = count1 + count2

                # keep sites with two alleles in these populations; count the minor allele,
                # choosing at random when the two are equally common
                variable = (both > 0) & (both < total)
                tie = 2 * both == total
                use_other = (2 * both > total) | (tie & (self.rng.random(len(both)) < 0.5))
                count1 = np.where(use_other, downsampling[pop1] - count1, count1)[variable]
                count2 = np.where(use_other, downsampling[pop2] - count2, count2)[variable]

                sfs = np.zeros((downsampling[pop1] + 1, downsampling[pop2] + 1))
                np.add.at(sfs, (count1, count2), 1)
                sfs_2d[(pop1, pop2)] = sfs
            sfs_list.append(sfs_2d)

        return sfs_list

    def plot_2dsfs(self, sfs_list, output_directory):
        """Plot average 2 dimensional Site frequency spectra."""

        for key in sfs_list[0].keys():
            average = np.mean([d[key] for d in sfs_list], axis=0)
            outfile  = os.path.join(output_directory, f"2D_SFS_{key}_empirical.png")
            plt.imshow(average, cmap='viridis', origin="lower")
            plt.colorbar()  # Add colorbar to show scale
            plt.title(f"2D SFS {key}")
            plt.savefig(outfile)
            plt.close()

    def numpy_to_msfs(self, encoded_alignment, downsampling, replicates = 1, nbins=None):

        """Convert numpy array to multidimensional site frequency spectra"""

        _check_even(downsampling)
        populations = list(self.config["sampling dict"].keys())
        sizes = [downsampling[population] for population in populations]
        total = sum(sizes)

        # with binning, each count maps to the smallest bin edge at or above it
        # (edges per population: floor(size / nbins * (x + 1)) for x = 0 .. nbins - 1)
        if nbins is None:
            to_cell = [np.arange(size + 1) for size in sizes]
            shape = [size + 1 for size in sizes]
        else:
            to_cell, shape = [], []
            for size in sizes:
                edges = np.unique([int(np.floor(size / nbins * (x + 1))) for x in range(nbins)])
                to_cell.append(np.searchsorted(edges, np.arange(size + 1)))
                shape.append(len(edges))

        sites = self._usable_sites(encoded_alignment, downsampling)

        all_sfs = []
        for _ in range(replicates):
            counts = self._downsampled_counts(sites, downsampling)
            both = sum(counts.values())

            # keep sites that still have two alleles, and count the minor allele after
            # downsampling (ties: the lower-coded allele, as in simulate_data)
            variable = (both > 0) & (both < total)
            use_other = 2 * both >= total
            cells = [to_cell[ix][np.where(use_other, size - counts[population],
                                          counts[population])[variable]]
                     for ix, (population, size) in enumerate(zip(populations, sizes))]
            flat = np.ravel_multi_index(cells, shape)
            all_sfs.append(np.bincount(flat, minlength=int(np.prod(shape))).astype(np.int64))

        average_sites = int(np.ceil(np.mean([np.sum(x) for x in all_sfs])))
        self.logger.info("We used an average of %s sites to construct the mSFS.", average_sites)

        return(all_sfs, average_sites)

    def _samples_in_order(self, dictionary):
        """Samples grouped by population, in sampling dict order."""
        return [sample for population in self.config["sampling dict"]
                for sample, value in self.config[dictionary].items() if value == population]

    def _population_blocks(self, encoded_alignment):
        """The rows of each population, in sampling dict order."""
        blocks, start = [], 0
        for size in self.config["sampling dict"].values():
            blocks.append(encoded_alignment[start:start + size])
            start += size
        return blocks

    def _finish_array(self, encoded_alignments, maxsites):
        """Drop invariant sites, recode alleles by frequency, and apply maxsites."""
        encoded_array = minor_encoding(_remove_invariant_columns(encoded_alignments))
        filtered_array = _apply_maxsites(encoded_array, maxsites)
        self.logger.info("""Empirical data has %s SNPs, but we are using %s SNPs""",
                         encoded_array.shape[1], filtered_array.shape[1])
        return filtered_array

    def _usable_sites(self, encoded_alignment, downsampling):
        """Sites with exactly two alleles (ignoring missing data) and enough called
        alleles in every population to downsample to the requested sizes."""
        enough = np.logical_and.reduce([
            (block != -1).sum(axis=0) >= downsampling[population]
            for population, block in zip(self.config["sampling dict"],
                                         self._population_blocks(encoded_alignment))])
        observed = encoded_alignment != -1
        low = np.where(observed, encoded_alignment, np.iinfo(encoded_alignment.dtype).max).min(axis=0)
        high = np.where(observed, encoded_alignment, -1).max(axis=0)
        two_alleles = (low < high) & ((encoded_alignment == low) | (encoded_alignment == high)
                                      | ~observed).all(axis=0)
        keep = enough & two_alleles
        return encoded_alignment[:, keep], high[keep]

    def _downsampled_counts(self, sites, downsampling):
        """For each population, sample the requested number of called alleles at every
        site without replacement, and count the higher-coded allele among them."""
        alignment, high = sites
        counts = {}
        for population, block in zip(self.config["sampling dict"], self._population_blocks(alignment)):
            keys = self.rng.random(block.shape)
            keys[block == -1] = 2.0                    # missing alleles sort last
            chosen = np.argsort(keys, axis=0)[:downsampling[population]]
            counts[population] = (np.take_along_axis(block, chosen, axis=0) == high).sum(axis=0)
        return counts

def _check_even(downsampling):
    if not all(value % 2 == 0 for value in downsampling.values()):
        raise ValueError("Error in downampling, all keys must be even.")

def _remove_invariant_columns(encoded_alignments):
    """Remove sites (columns) with fewer than two different alleles, ignoring missing data (-1)."""
    missing = encoded_alignments == -1
    largest = np.where(missing, -1, encoded_alignments).max(axis=0)
    smallest = np.where(missing, np.iinfo(encoded_alignments.dtype).max, encoded_alignments).min(axis=0)
    return encoded_alignments[:, smallest < largest]

def _apply_maxsites(encoded_array, maxsites):
    """Keep the first maxsites SNPs, or pad with missing data (-1) up to maxsites,
    as simulate_data does for simulated data. maxsites=np.inf keeps every SNP."""
    if encoded_array.shape[1] > maxsites:
        return encoded_array[:, :maxsites]
    if encoded_array.shape[1] < maxsites and maxsites != np.inf:
        num_missing_columns = maxsites - encoded_array.shape[1]
        missing_columns = np.full((encoded_array.shape[0], num_missing_columns), -1)
        return np.concatenate((encoded_array, missing_columns), axis=1)
    return encoded_array