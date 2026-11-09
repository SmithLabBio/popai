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