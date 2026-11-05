"""Tests for popai.process_empirical.DataProcessor (empirical data -> arrays and SFS).

Inputs are built in code (small VCFs as lists of lines, FASTA files in a temporary
folder), so each test shows exactly what goes in. Expected arrays without
downsampling are worked out by hand. With downsampling, tests check properties that
must hold whatever the random draws are, and compare against simulate_data, which
computes the simulated SFS that the empirical SFS is compared with.

Run with:  python -m pytest tests/test_process_empirical.py
"""

import logging
import os
from math import comb
import pickle
from collections import OrderedDict
from itertools import product

import dendropy
import numpy as np
import pytest

from popai.process_empirical import DataProcessor, _apply_maxsites
from popai.simulate_data import DataSimulator
from popai.utils import minor_encoding

logging.disable(logging.INFO)


# ---------------------------------------------------------------------------
# Building inputs
# ---------------------------------------------------------------------------

def vcf_lines(sites, samples, header_ids=("GT",), extra_header=()):
    """sites: list of (FORMAT, [genotype per sample]) or (FORMAT, [genotypes], ALT)."""
    lines = ["##fileformat=VCFv4.2\n"]
    lines += list(extra_header)
    lines += [f'##FORMAT=<ID={i},Number=1,Type=String,Description="{i}">\n' for i in header_ids]
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
    for pos, site in enumerate(sites, 1):
        fmt, genotypes = site[0], site[1]
        alt = site[2] if len(site) > 2 else "C"
        lines.append(f"1\t{pos}\t.\tA\t{alt}\t.\tPASS\t.\t{fmt}\t" + "\t".join(genotypes) + "\n")
    return lines


def vcf_config(lines, populations):
    """populations: OrderedDict population -> [samples], in the order rows should appear."""
    original = {s: p for p, samples in populations.items() for s in samples}
    return {
        "seed": 1,
        "vcf": lines,
        "original population dictionary": original,
        "population dictionary": {f"{s}_{h}": p for s, p in original.items() for h in "ab"},
        "sampling dict": OrderedDict((p, 2 * len(s)) for p, s in populations.items()),
    }


TWO_SAMPLES = OrderedDict([("p1", ["s1"]), ("p2", ["s2"])])


def parse_vcf(sites, populations=TWO_SAMPLES, maxsites=np.inf, samples=None, **kwargs):
    """Parse a small VCF. Genotypes are listed in the order of 'samples' (by default,
    the order the samples appear in 'populations')."""
    if samples is None:
        samples = [s for group in populations.values() for s in group]
    lines = vcf_lines(sites, samples, **kwargs)
    return DataProcessor(vcf_config(lines, populations)).vcf_to_numpy(maxsites)


def fasta_config(folder, alignments, populations):
    """alignments: list of {sample: sequence}; populations: OrderedDict population -> [samples]."""
    fastas = []
    for ix, seqs in enumerate(alignments):
        path = os.path.join(folder, f"alignment_{ix}.fa")
        with open(path, "w") as f:
            for name, seq in seqs.items():
                f.write(f">{name}\n{seq}\n")
        fastas.append(dendropy.DnaCharacterMatrix.get(path=path, schema="fasta"))
    return {
        "seed": 1,
        "fastas": fastas,
        "population dictionary": {s: p for p, samples in populations.items() for s in samples},
        "sampling dict": OrderedDict((p, len(s)) for p, s in populations.items()),
    }


def sfs_config(sizes):
    return {"seed": 1, "sampling dict": OrderedDict(sizes)}


def msfs_cells(sizes):
    return ["_".join(map(str, c)) for c in product(*(range(n + 1) for n in sizes))]


def nonzero_cells(msfs, sizes):
    return {cell: int(v) for cell, v in zip(msfs_cells(sizes), msfs) if v}


def random_biallelic(n_haplotypes, n_sites, seed, missing=0.0):
    rng = np.random.default_rng(seed)
    data = (rng.random((n_haplotypes, n_sites)) < rng.uniform(0.05, 0.5, n_sites)).astype(int)
    data = minor_encoding(data)
    data[rng.random(data.shape) < missing] = -1
    return data


def simulate_data_sfs(tmp_path, replicates, sizes, downsampling, nbins=None):
    """Run simulate_data's SFS code on given arrays (no simulation involved)."""
    with open(os.path.join(tmp_path, "simulated_arrays_0.pickle"), "wb") as f:
        pickle.dump(list(replicates), f)
    simulator = DataSimulator(models=None, labels=[0], config=sfs_config(sizes), cores=1,
                              downsampling=dict(downsampling), output=str(tmp_path), user=True)
    msfs = simulator.mutations_to_sfs(None, nbins=nbins)["0"]
    sfs_2d = simulator.mutations_to_2d_sfs()["0"]
    return msfs, sfs_2d


# ---------------------------------------------------------------------------
# 1. VCF to array
# ---------------------------------------------------------------------------

def test_vcf_rows_are_haplotypes_in_sampling_dict_order():
    populations = OrderedDict([("p2", ["s3", "s4"]), ("p1", ["s1", "s2"])])
    sites = [("GT", ["0/0", "0/0", "0/1", "1/1"]),   # s1 s2 s3 s4
             ("GT", ["1/1", "1/1", "1/1", "0/1"])]
    arr = parse_vcf(sites, populations, samples=["s1", "s2", "s3", "s4"])
    # rows: s3a s3b s4a s4b s1a s1b s2a s2b
    # site 1 -> 0 1 1 1 0 0 0 0 (0 is more common, so unchanged)
    # site 2 -> 1 1 0 1 1 1 1 1, recoded so the more common allele is 0
    expected = np.array([[0, 0], [1, 0], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(arr, expected)


DP_MASKED = np.array([[0, 1], [1, 0], [-1, 0], [-1, 0]])  # s2's first genotype is missing


@pytest.mark.parametrize("header_ids", [("GT", "AD", "DP"), ("AD", "DP", "GT")],
                         ids=["headers GT,AD,DP", "headers AD,DP,GT (GATK order)"])
def test_genotypes_with_no_reads_are_missing(header_ids):
    sites = [("GT:AD:DP", ["0/1:3,4:7", "1/1:0,0:0"]),
             ("GT:AD:DP", ["0/1:5,5:10", "1/1:0,8:8"])]
    np.testing.assert_array_equal(parse_vcf(sites, header_ids=header_ids), DP_MASKED)


def test_missing_genotype_without_trailing_fields():
    sites = [("GT:AD:DP", ["0/1:3,4:7", "./."]),
             ("GT:AD:DP", ["0/1:5,5:10", "1/1:0,8:8"])]
    np.testing.assert_array_equal(parse_vcf(sites, header_ids=("AD", "DP", "GT")), DP_MASKED)


def test_vcf_without_read_depth():
    sites = [("GT", ["0/1", "1/1"]), ("GT", ["0/0", "0/1"])]
    np.testing.assert_array_equal(parse_vcf(sites), [[1, 0], [0, 0], [0, 0], [0, 1]])


def test_read_depth_recorded_at_some_sites_only():
    # site 1: s1 has no reads, leaving only allele 1, so the site is dropped as invariant
    sites = [("GT:DP", ["0/1:0", "1/1:5"]), ("GT", ["0/1", "1/1"])]
    np.testing.assert_array_equal(parse_vcf(sites, header_ids=("GT", "DP")), [[1], [0], [0], [0]])


def test_half_missing_genotype_keeps_the_called_allele():
    sites = [("GT", ["./1", "0/0"]), ("GT", ["0|.", "1/1"])]
    np.testing.assert_array_equal(parse_vcf(sites), [[-1, 1], [1, -1], [0, 0], [0, 0]])


def test_haploid_genotypes_give_a_clear_error():
    with pytest.raises(ValueError, match="diploid"):
        parse_vcf([("GT", ["0", "1"]), ("GT", ["1", "1"])])


def test_header_line_mentioning_chrom():
    sites = [("GT", ["0/1", "1/1"]), ("GT", ["0/0", "0/1"])]
    arr = parse_vcf(sites, extra_header=["##contig=<ID=CHROM1,length=100>\n"])
    assert arr.shape == (4, 2)


def test_site_whose_alleles_include_a_high_alt_allele_is_kept():
    sites = [("GT", ["0/4", "4/4"], "C,G,T,AA"), ("GT", ["0/1", "1/1"])]
    np.testing.assert_array_equal(parse_vcf(sites), [[1, 1], [0, 0], [0, 0], [0, 0]])


def test_invariant_and_fully_missing_sites_are_removed():
    sites = [("GT", ["0/0", "0/0"]), ("GT", ["./.", "./."]), ("GT", ["1/1", "./."]),
             ("GT", ["0/1", "1/1"])]
    np.testing.assert_array_equal(parse_vcf(sites), [[1], [0], [0], [0]])


VARIABLE_SITES = [("GT", ["0/1", "1/1"]), ("GT", ["0/0", "0/1"]), ("GT", ["1/1", "0/0"])]


@pytest.mark.parametrize("maxsites,n_columns", [(2, 2), (3, 3), (5, 5), (np.inf, 3)])
def test_vcf_maxsites_keeps_first_snps_or_pads(maxsites, n_columns):
    full = parse_vcf(VARIABLE_SITES)
    arr = parse_vcf(VARIABLE_SITES, maxsites=maxsites)
    assert arr.shape == (4, n_columns)
    kept = min(3, n_columns)
    np.testing.assert_array_equal(arr[:, :kept], full[:, :kept])
    assert (arr[:, kept:] == -1).all()


def test_apply_maxsites():
    arr = np.arange(12).reshape(2, 6)
    np.testing.assert_array_equal(_apply_maxsites(arr, 4), arr[:, :4])
    np.testing.assert_array_equal(_apply_maxsites(arr, 6), arr)
    np.testing.assert_array_equal(_apply_maxsites(arr, np.inf), arr)
    padded = _apply_maxsites(arr, 8)
    np.testing.assert_array_equal(padded[:, :6], arr)
    assert (padded[:, 6:] == -1).all()


# ---------------------------------------------------------------------------
# 2. FASTA to array
# ---------------------------------------------------------------------------

FASTA_POPS = OrderedDict([("p1", ["a", "b"]), ("p2", ["c", "d"])])
FASTA_SEQS = {"a": "ACGTN", "b": "ACGAR", "c": "TCGA-", "d": "TCGTA"}
# columns: A/A/T/T (kept), C (invariant), G (invariant), T/A/A/T (kept),
# N/R/-/A (one called base, so invariant); A=0, T=1, then recoded so the first-seen
# allele of a tie is 0
FASTA_EXPECTED = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])


def test_fasta_encoding_and_missing_bases(tmp_path):
    config = fasta_config(tmp_path, [FASTA_SEQS], FASTA_POPS)
    np.testing.assert_array_equal(DataProcessor(config).fasta_to_numpy(np.inf), FASTA_EXPECTED)


def test_fasta_lowercase_bases(tmp_path):
    config = fasta_config(tmp_path, [{k: v.lower() for k, v in FASTA_SEQS.items()}], FASTA_POPS)
    np.testing.assert_array_equal(DataProcessor(config).fasta_to_numpy(np.inf), FASTA_EXPECTED)


def test_fasta_alignments_are_joined_and_absent_samples_are_missing(tmp_path):
    second = {"a": "AT", "b": "TT", "c": "AA"}   # d is not in this alignment
    config = fasta_config(tmp_path, [FASTA_SEQS, second], FASTA_POPS)
    arr = DataProcessor(config).fasta_to_numpy(np.inf)
    expected = np.hstack([FASTA_EXPECTED, [[0, 0], [1, 0], [0, 1], [-1, -1]]])
    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize("maxsites,n_columns", [(1, 1), (2, 2), (4, 4), (np.inf, 2)])
def test_fasta_maxsites(tmp_path, maxsites, n_columns):
    config = fasta_config(tmp_path, [FASTA_SEQS], FASTA_POPS)
    arr = DataProcessor(config).fasta_to_numpy(maxsites)
    assert arr.shape == (4, n_columns)
    kept = min(2, n_columns)
    np.testing.assert_array_equal(arr[:, :kept], FASTA_EXPECTED[:, :kept])


# ---------------------------------------------------------------------------
# 3. Choosing downsampling sizes
# ---------------------------------------------------------------------------

def test_find_downsampling_counts_usable_sites():
    #            no missing  p1 missing 1  p2 missing 2  p1 missing 3
    arr = np.array([[0, -1, 0, -1], [1, 0, 1, -1], [0, 1, 0, -1], [1, 0, 1, 0],
                    [0, 1, -1, 0], [1, 0, -1, 1], [0, 1, 0, 0], [1, 0, 1, 1]])
    dp = DataProcessor(sfs_config([("p1", 4), ("p2", 4)]))
    assert dp.find_downsampling(arr) == {(4, 4): 1, (4, 2): 2, (2, 4): 2, (2, 2): 3}


# ---------------------------------------------------------------------------
# 4. SFS without downsampling (no randomness, exact values)
# ---------------------------------------------------------------------------

# rows p1a p1b p2a p2b
MSFS_DATA = np.array([[1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 1, 1],
                      [0, 1, 1, 0, 0, 2, 0],
                      [0, 1, 0, 1, 0, 0, -1]])
# site 1: minor allele 1 -> 1_0       site 2: 2-2 tie, allele 0 counted -> 2_0
# site 3: tie -> 1_1                   site 4: minor allele 1 -> 0_1
# site 5: monomorphic; site 6: three alleles; site 7: missing data -> all dropped


def test_msfs_without_downsampling():
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 2)]))
    msfs, average = dp.numpy_to_msfs(MSFS_DATA, {"p1": 2, "p2": 2})
    assert nonzero_cells(msfs[0], [2, 2]) == {"1_0": 1, "2_0": 1, "1_1": 1, "0_1": 1}
    assert average == 4


def test_binned_msfs_without_downsampling():
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 2)]))
    msfs, _ = dp.numpy_to_msfs(MSFS_DATA, {"p1": 2, "p2": 2}, nbins=2)
    # bins per population: counts 0-1 -> 1, count 2 -> 2; cells 1_1, 1_2, 2_1, 2_2
    np.testing.assert_array_equal(msfs[0], [3, 0, 1, 0])


def test_2d_sfs_without_downsampling():
    # rows p1a p1b | p2a p2b p2c p2d; no site has a tie between the two alleles
    arr = np.array([[1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0]])
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 4)]))
    sfs = dp.numpy_to_2d_sfs(arr, {"p1": 2, "p2": 4})[0][("p1", "p2")]
    expected = np.zeros((3, 5))
    expected[1, 0] = expected[0, 2] = expected[2, 0] = expected[1, 1] = 1
    np.testing.assert_array_equal(sfs, expected)


@pytest.mark.parametrize("method", ["numpy_to_msfs", "numpy_to_2d_sfs"])
def test_odd_downsampling_sizes_are_rejected(method):
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 4)]))
    with pytest.raises(ValueError, match="even"):
        getattr(dp, method)(np.zeros((6, 3), dtype=int), {"p1": 2, "p2": 3})


# ---------------------------------------------------------------------------
# 5. SFS with downsampling (properties that hold for any random draws)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def downsampled():
    data = random_biallelic(14, 3000, seed=2, missing=0.1)   # p1: 6, p2: 8
    dp = DataProcessor(sfs_config([("p1", 6), ("p2", 8)]))
    ds = {"p1": 2, "p2": 4}
    msfs, _ = dp.numpy_to_msfs(data, ds, replicates=3)
    sfs_2d = dp.numpy_to_2d_sfs(data, ds, replicates=3)
    return msfs, [s[("p1", "p2")] for s in sfs_2d]


def test_downsampled_msfs_is_folded_with_no_monomorphic_sites(downsampled):
    msfs, _ = downsampled
    for replicate in msfs:
        assert replicate.sum() > 0
        for cell in nonzero_cells(replicate, [2, 4]):
            i, j = map(int, cell.split("_"))
            assert 0 < i + j <= 3   # minor allele: at least 1, at most half of 6


def test_downsampled_2d_sfs_is_folded_with_no_monomorphic_sites(downsampled):
    _, sfs_2d = downsampled
    for replicate in sfs_2d:
        assert replicate.sum() > 0
        for i, j in zip(*np.nonzero(replicate)):
            assert 0 < i + j <= 3


def test_population_kept_at_full_size_is_not_resampled():
    # p1 always carries one copy of each allele, so its minor-allele count must be 1
    data = random_biallelic(8, 2000, seed=4)
    data[0], data[1] = 0, 1
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 6)]))
    msfs, _ = dp.numpy_to_msfs(data, {"p1": 2, "p2": 2})
    assert {cell.split("_")[0] for cell in nonzero_cells(msfs[0], [2, 2])} == {"1"}
    sfs = dp.numpy_to_2d_sfs(data, {"p1": 2, "p2": 2})[0][("p1", "p2")]
    assert sfs[1].sum() == sfs.sum()


@pytest.mark.parametrize("method", ["msfs", "2d"])
def test_downsampling_is_without_replacement(method):
    # p1 = [0, 1] kept at 2; p2 = [0, 0, 1, 1] downsampled to 2. p2 keeps one copy of
    # each allele (count 1) with probability 4/6 without replacement, 2/4 with.
    arr = np.repeat(np.array([[0], [1], [0], [0], [1], [1]]), 20000, axis=1)
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 4)]))
    if method == "msfs":
        msfs, _ = dp.numpy_to_msfs(arr, {"p1": 2, "p2": 2})
        cells = nonzero_cells(msfs[0], [2, 2])
        fraction = sum(v for c, v in cells.items() if c.endswith("_1")) / sum(cells.values())
    else:
        sfs = dp.numpy_to_2d_sfs(arr, {"p1": 2, "p2": 2})[0][("p1", "p2")]
        fraction = sfs[:, 1].sum() / sfs.sum()
    assert fraction == pytest.approx(2 / 3, abs=0.02)


def test_sites_with_too_much_missing_data_are_dropped():
    # p2 has only 3 called alleles of 8 at every site, fewer than the 4 asked for
    data = random_biallelic(14, 500, seed=5)
    data[6:11] = -1
    dp = DataProcessor(sfs_config([("p1", 6), ("p2", 8)]))
    msfs, average = dp.numpy_to_msfs(data, {"p1": 2, "p2": 4})
    assert msfs[0].sum() == 0 and average == 0
    sfs = dp.numpy_to_2d_sfs(data, {"p1": 2, "p2": 4})[0][("p1", "p2")]
    assert sfs.sum() == 0


def triallelic_sites(n_sites):
    # p1: 0 0 | p2: 0 1 1 1 1 2 -- allele 2 is often lost when p2 is downsampled to 2
    return np.repeat(np.array([[0], [0], [0], [1], [1], [1], [1], [2]]), n_sites, axis=1)


def test_empirical_sfs_drops_sites_with_three_alleles_even_after_downsampling():
    dp = DataProcessor(sfs_config([("p1", 2), ("p2", 6)]))
    msfs, _ = dp.numpy_to_msfs(triallelic_sites(300), {"p1": 2, "p2": 2}, replicates=2)
    assert all(rep.sum() == 0 for rep in msfs)
    sfs_2d = dp.numpy_to_2d_sfs(triallelic_sites(300), {"p1": 2, "p2": 2}, replicates=2)
    assert all(rep[("p1", "p2")].sum() == 0 for rep in sfs_2d)


# ---------------------------------------------------------------------------
# 6. Empirical and simulated SFS are computed the same way
# ---------------------------------------------------------------------------

def complete_data_with_triallelic_sites(seed):
    data = random_biallelic(10, 1500, seed=seed)
    data[9, ::25] = 2          # some sites get a third allele
    return minor_encoding(data)   # as for real data: 0 most common, then 1, then 2


def test_msfs_matches_simulate_data_without_downsampling(tmp_path):
    sizes = [("p1", 4), ("p2", 6)]
    data = complete_data_with_triallelic_sites(6)
    empirical, _ = DataProcessor(sfs_config(sizes)).numpy_to_msfs(data, dict(sizes))
    simulated, _ = simulate_data_sfs(tmp_path, [data], sizes, dict(sizes))
    np.testing.assert_array_equal(empirical[0], simulated[0])


def test_binned_msfs_matches_simulate_data_without_downsampling(tmp_path):
    sizes = [("p1", 4), ("p2", 6)]
    data = complete_data_with_triallelic_sites(7)
    empirical, _ = DataProcessor(sfs_config(sizes)).numpy_to_msfs(data, dict(sizes), nbins=2)
    simulated, _ = simulate_data_sfs(tmp_path, [data], sizes, dict(sizes), nbins=2)
    np.testing.assert_array_equal(empirical[0], simulated[0])


def test_2d_sfs_matches_simulate_data_without_downsampling(tmp_path):
    # both break 2-2 style ties at random, so leave out sites with a tie
    sizes = [("p1", 4), ("p2", 6)]
    data = complete_data_with_triallelic_sites(8)
    data = data[:, (data == 1).sum(axis=0) != 5]
    empirical = DataProcessor(sfs_config(sizes)).numpy_to_2d_sfs(data, dict(sizes))[0]
    _, simulated = simulate_data_sfs(tmp_path, [data], sizes, dict(sizes))
    np.testing.assert_array_equal(empirical[("p1", "p2")], simulated[0][("p1", "p2")])


def exact_downsampled_msfs(data, sizes, downsampling):
    """Expected mSFS of complete biallelic data for two populations, downsampled
    without replacement (hypergeometric), folded after downsampling, with sites that
    become monomorphic dropped and ties counted for allele 0, as in simulate_data."""
    (n1, n2), (d1, d2) = [n for _, n in sizes], [downsampling[p] for p, _ in sizes]
    expected = np.zeros((d1 + 1) * (d2 + 1))
    for column in data.T:
        k1, k2 = column[:n1].sum(), column[n1:].sum()
        for a, b in product(range(d1 + 1), range(d2 + 1)):
            if a + b in (0, d1 + d2):
                continue
            p = (comb(k1, a) * comb(n1 - k1, d1 - a) / comb(n1, d1)
                 * comb(k2, b) * comb(n2 - k2, d2 - b) / comb(n2, d2))
            i, j = (a, b) if 2 * (a + b) < d1 + d2 else (d1 - a, d2 - b)
            expected[i * (d2 + 1) + j] += p
    return expected


def test_downsampled_msfs_matches_exact_expectation_and_simulate_data(tmp_path):
    sizes = [("p1", 6), ("p2", 4)]
    ds = {"p1": 2, "p2": 2}
    data = random_biallelic(10, 3000, seed=9)
    reps = 100
    exact = exact_downsampled_msfs(data, sizes, ds)
    empirical, _ = DataProcessor(sfs_config(sizes)).numpy_to_msfs(data, ds, replicates=reps)
    simulated, _ = simulate_data_sfs(tmp_path, [data] * reps, sizes, ds)
    for name, result in [("empirical", empirical), ("simulate_data", simulated)]:
        result = np.array(result)
        mean, se = result.mean(axis=0), result.std(axis=0) / np.sqrt(reps)
        possible = exact > 0
        assert (mean[~possible] == 0).all(), name
        assert (np.abs(mean - exact)[possible] < 5 * se[possible] + 1).all(), name


def test_simulate_data_drops_sites_with_three_alleles_even_after_downsampling(tmp_path):
    sizes = [("p1", 2), ("p2", 6)]
    msfs, sfs_2d = simulate_data_sfs(tmp_path, [triallelic_sites(300)] * 2, sizes,
                                     {"p1": 2, "p2": 2})
    assert all(rep.sum() == 0 for rep in msfs)
    assert all(rep[("p1", "p2")].sum() == 0 for rep in sfs_2d)


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------

def test_plot_2dsfs_writes_one_file_per_population_pair(tmp_path):
    data = random_biallelic(12, 200, seed=10)
    dp = DataProcessor(sfs_config([("p1", 4), ("p2", 4), ("p3", 4)]))
    sfs_2d = dp.numpy_to_2d_sfs(data, {"p1": 2, "p2": 2, "p3": 2}, replicates=2)
    dp.plot_2dsfs(sfs_2d, str(tmp_path))
    written = sorted(os.listdir(tmp_path))
    assert len(written) == 3 and all(name.endswith("_empirical.png") for name in written)