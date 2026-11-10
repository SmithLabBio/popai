import numpy as np



def sort_haplotypes(arr, population_sizes):
    """Sort the rows (haplotypes) within each population by how similar they are to that
    population's first row (Euclidean distance, most similar first; ties keep their order).
    The rows are in population order, with population_sizes[i] rows for population i.
    Simulated and empirical arrays are both sorted this way before they are used as CNN input."""
    groups = np.split(arr, np.cumsum(list(population_sizes))[:-1])
    sorted_groups = []
    for group in groups:
        distances = np.linalg.norm(group - group[0:1], axis=1)
        sorted_groups.append(group[np.argsort(distances, kind='stable')])
    return np.vstack(sorted_groups)

def minor_encoding(arr):
    """Recode each site (column) so its most common allele is 0, the next most common 1,
    and so on. Ties go to the allele that appears first in the column. Missing data (-1)
    stays -1."""
    result = arr.copy()
    if arr.size == 0 or arr.max() < 0:
        return result
    n_rows = arr.shape[0]
    alleles = np.arange(arr.max() + 1)
    counts = np.array([(arr == a).sum(axis=0) for a in alleles])          # alleles x sites
    present = counts > 0
    first_row = np.array([(arr == a).argmax(axis=0) for a in alleles])     # first appearance
    first_row = np.where(present, first_row, n_rows)
    # sort alleles within each site: most common first, then by first appearance
    order = np.lexsort((first_row, -counts), axis=0)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(len(alleles))[:, None], axis=0)
    observed = arr != -1
    result[observed] = rank[arr[observed], np.nonzero(observed)[1]]
    return result

def population_blocks(arr, population_sizes):
    """The rows of each population: population_sizes[i] rows for population i, in order."""
    return np.split(arr, np.cumsum(list(population_sizes))[:-1])


def usable_sites(arr, downsampling, sampling_dict):
    """Sites with exactly two alleles (ignoring missing data) and enough called alleles in
    every population to downsample to the requested sizes. Returns those sites, and the
    higher-coded allele at each."""
    enough = np.logical_and.reduce([
        (block != -1).sum(axis=0) >= downsampling[population]
        for population, block in zip(sampling_dict, population_blocks(arr, sampling_dict.values()))])
    observed = arr != -1
    low = np.where(observed, arr, np.iinfo(arr.dtype).max).min(axis=0)
    high = np.where(observed, arr, -1).max(axis=0)
    two_alleles = (low < high) & ((arr == low) | (arr == high) | ~observed).all(axis=0)
    keep = enough & two_alleles
    return arr[:, keep], high[keep]


def downsampled_counts(sites, downsampling, sampling_dict, rng):
    """For each population, sample the requested number of called alleles at every site
    without replacement, and count the higher-coded allele among them."""
    alignment, high = sites
    counts = {}
    for population, block in zip(sampling_dict, population_blocks(alignment, sampling_dict.values())):
        keys = rng.random(block.shape)
        keys[block == -1] = 2.0                    # missing alleles sort last
        chosen = np.argsort(keys, axis=0)[:downsampling[population]]
        counts[population] = (np.take_along_axis(block, chosen, axis=0) == high).sum(axis=0)
    return counts


def two_d_sfs(arr, downsampling, sampling_dict, rng, replicates=1):
    """A list of 2D SFS, one per replicate downsampling. Each is a dict with one array per pair
    of populations: rows are minor-allele counts in the first population, columns in the second.
    Used for both empirical and simulated data."""
    populations = list(sampling_dict.keys())
    pairs = [(pop1, pop2) for i, pop1 in enumerate(populations) for pop2 in populations[i + 1:]]
    sites = usable_sites(arr, downsampling, sampling_dict)

    sfs_list = []
    for _ in range(replicates):
        # downsample each population once per replicate; count the higher-coded allele
        counts = downsampled_counts(sites, downsampling, sampling_dict, rng)
        sfs_2d = {}
        for pop1, pop2 in pairs:
            total = downsampling[pop1] + downsampling[pop2]
            count1, count2 = counts[pop1], counts[pop2]
            both = count1 + count2

            # keep sites with two alleles in these populations; count the minor allele,
            # choosing at random when the two are equally common
            variable = (both > 0) & (both < total)
            tie = 2 * both == total
            use_other = (2 * both > total) | (tie & (rng.random(len(both)) < 0.5))
            count1 = np.where(use_other, downsampling[pop1] - count1, count1)[variable]
            count2 = np.where(use_other, downsampling[pop2] - count2, count2)[variable]

            sfs = np.zeros((downsampling[pop1] + 1, downsampling[pop2] + 1))
            np.add.at(sfs, (count1, count2), 1)
            sfs_2d[(pop1, pop2)] = sfs
        sfs_list.append(sfs_2d)
    return sfs_list


def multi_sfs(arr, downsampling, sampling_dict, rng, replicates=1, nbins=None):
    """A list of multidimensional SFS (flattened), one per replicate downsampling, with cells
    for every combination of minor-allele counts (populations in sampling-dict order), or of
    bins with nbins. Used for both empirical and simulated data."""
    populations = list(sampling_dict.keys())
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

    sites = usable_sites(arr, downsampling, sampling_dict)

    all_sfs = []
    for _ in range(replicates):
        counts = downsampled_counts(sites, downsampling, sampling_dict, rng)
        both = sum(counts.values())

        # keep sites that still have two alleles, and count the minor allele after
        # downsampling (ties: the lower-coded allele)
        variable = (both > 0) & (both < total)
        use_other = 2 * both >= total
        cells = [to_cell[ix][np.where(use_other, size - counts[population],
                                      counts[population])[variable]]
                 for ix, (population, size) in enumerate(zip(populations, sizes))]
        flat = np.ravel_multi_index(cells, shape)
        all_sfs.append(np.bincount(flat, minlength=int(np.prod(shape))).astype(np.int64))
    return all_sfs