import numpy as np


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