"""Empirical data read through the config parser (parse_input -> process_empirical).

test_empirical_parser_v2.py builds its configs by hand. These tests instead read the
small data files in tests/ through ModelConfigParser, so they also check how the
parser loads the popfile, FASTA folder and VCF, orders populations in the sampling
dict (in the order they first appear in the popfile), and turns VCF samples into
haplotypes.

Run with:  python -m pytest tests/test_empirical_parser.py
"""

import logging
import os
from collections import OrderedDict
from itertools import product

import numpy as np
import pytest

from popai.parse_input import ModelConfigParser
from popai.process_empirical import DataProcessor

logging.disable(logging.INFO)

TESTS = os.path.dirname(os.path.abspath(__file__))


def parse(tmp_path, popfile, alignments=None, vcf=None, tree="species_tree_mini.nex",
          migration="migration_mini.txt"):
    """Write a config for the given test files, parse it, and return the config values."""
    data = (f"alignments = {os.path.join(TESTS, alignments)}/" if alignments else
            f"alignments = None\nvcf = {os.path.join(TESTS, vcf)}")
    config = f"""
[Model]
species tree file = {os.path.join(TESTS, tree)}
migration matrix = {os.path.join(TESTS, migration)}
symmetric = True
secondary contact = True
divergence with gene flow = False
max migration events = 1
migration rate = U(1e-5, 1e-4)
constant Ne = True

[Other]
output directory = {tmp_path}
seed = 1234
replicates = 10

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
{data}
popfile = {os.path.join(TESTS, popfile)}
"""
    path = os.path.join(tmp_path, "config.txt")
    with open(path, "w") as f:
        f.write(config)
    return ModelConfigParser(path).parse_config()


def nonzero_cells(msfs, sizes):
    cells = ["_".join(map(str, c)) for c in product(*(range(n + 1) for n in sizes))]
    return {cell: int(v) for cell, v in zip(cells, msfs) if v}


# ---------------------------------------------------------------------------
# The mini datasets: 6 haplotypes, pop1 (2) then pop2 (4)
# ---------------------------------------------------------------------------
# FASTA (populations_mini.txt): n0, n1 in pop1; n2-n5 in pop2, one sequence each.
# VCF (populations_mini_vcf.txt): n0 in pop1; n1, n2 in pop2, two haplotypes each.
# The sampling dict lists populations in the order they first appear in the popfile
# (pop1, then pop2), so rows are pop1's haplotypes, then pop2's.
#
# Variable sites, rows n0 n1 | n2 n3 n4 n5 (FASTA) or n0a n0b | n1a n1b n2a n2b (VCF),
# recoded so 0 is the most common allele (ties: the allele seen first):
#   site 1: FASTA  T T | - A A A   ->  1 1 -1 0 0 0  (n2 has a gap)
#           VCF    0 0 | . . 1 1   ->  0 0 -1 -1 1 1 (n1 missing; a 2-2 tie, 0 seen first)
#   site 2: C T | T T T T          ->  1 0 0 0 0 0
#   site 3: T T | T C C C  (3-3)   ->  0 0 0 1 1 1
#   site 4: T T | T T G G          ->  0 0 0 0 1 1
FASTA_ARRAY = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 0, 1, 1], [0, 0, 1, 1]])
VCF_ARRAY = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 1, 0],
                      [1, 0, 1, 1], [1, 0, 1, 1]])
# The multiallelic versions replace site 1 with T T | G A A A (FASTA) or 0 0 | 2 1 1 1
# (VCF): three alleles, recoded A/1 -> 0, T/0 -> 1, G/2 -> 2.
MULTIALLELIC_ARRAY = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0],
                               [0, 0, 1, 1], [0, 0, 1, 1]])

MINI = {
    "fasta": dict(popfile="populations_mini.txt", alignments="mini_dataset"),
    "vcf": dict(popfile="populations_mini_vcf.txt", vcf="mini_dataset/alignment.vcf"),
    "fasta, multiallelic": dict(popfile="populations_mini.txt",
                                alignments="mini_dataset_multiallelic"),
    "vcf, multiallelic": dict(popfile="populations_mini_vcf.txt",
                              vcf="mini_dataset_multiallelic/alignment.vcf"),
}
EXPECTED_ARRAY = {"fasta": FASTA_ARRAY, "vcf": VCF_ARRAY,
                  "fasta, multiallelic": MULTIALLELIC_ARRAY,
                  "vcf, multiallelic": MULTIALLELIC_ARRAY}


def to_numpy(config):
    dp = DataProcessor(config)
    if "vcf" in config:
        return dp, dp.vcf_to_numpy(np.inf)
    return dp, dp.fasta_to_numpy(np.inf)


def test_fasta_config(tmp_path):
    config = parse(tmp_path, **MINI["fasta"])
    assert config["sampling dict"] == OrderedDict([("pop1", 2), ("pop2", 4)])
    assert list(config["sampling dict"]) == ["pop1", "pop2"]
    assert config["population dictionary"] == {
        "n0": "pop1", "n1": "pop1", "n2": "pop2", "n3": "pop2", "n4": "pop2", "n5": "pop2"}


def test_vcf_config_counts_two_haplotypes_per_sample(tmp_path):
    config = parse(tmp_path, **MINI["vcf"])
    assert list(config["sampling dict"].items()) == [("pop1", 2), ("pop2", 4)]
    assert config["population dictionary"] == {
        "n0_a": "pop1", "n0_b": "pop1", "n1_a": "pop2", "n1_b": "pop2",
        "n2_a": "pop2", "n2_b": "pop2"}


@pytest.mark.parametrize("dataset", list(MINI))
def test_mini_dataset_array(tmp_path, dataset):
    _, arr = to_numpy(parse(tmp_path, **MINI[dataset]))
    np.testing.assert_array_equal(arr, EXPECTED_ARRAY[dataset])


@pytest.mark.parametrize("dataset", list(MINI))
def test_mini_dataset_sfs_without_downsampling(tmp_path, dataset):
    # Site 1 never counts: it has missing data in pop2 (so it can't give 4 alleles) or,
    # in the multiallelic data, three alleles. The other sites, as minor-allele counts
    # in pop1_pop2 (site 3 is a tie, so allele 0 is counted): 1_0, 2_1, 0_2.
    dp, arr = to_numpy(parse(tmp_path, **MINI[dataset]))
    downsampling = {"pop1": 2, "pop2": 4}
    msfs, _ = dp.numpy_to_msfs(arr, downsampling)
    assert nonzero_cells(msfs[0], [2, 4]) == {"1_0": 1, "2_1": 1, "0_2": 1}
    sfs_2d = dp.numpy_to_2d_sfs(arr, downsampling)[0]
    assert list(sfs_2d) == [("pop1", "pop2")]
    assert sfs_2d[("pop1", "pop2")].sum() == 3


# ---------------------------------------------------------------------------
# A larger, three-population FASTA dataset (tests/alignments, populations.txt)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_populations(tmp_path_factory):
    config = parse(tmp_path_factory.mktemp("three_pops"), popfile="populations.txt",
                   alignments="alignments", tree="species_tree.nex", migration="migration.txt")
    dp = DataProcessor(config)
    return config, dp, dp.fasta_to_numpy(np.inf)


def test_three_populations_config_and_array(three_populations):
    config, _, arr = three_populations
    # 12 A, 8 B and 10 C individuals, in the order they appear in the popfile
    assert list(config["sampling dict"].items()) == [("A", 12), ("B", 8), ("C", 10)]
    assert len(config["fastas"]) == 10
    assert arr.shape == (30, 1038)


def test_three_populations_downsampling_options(three_populations):
    _, dp, arr = three_populations
    options = dp.find_downsampling(arr)
    assert (8, 6, 6) in options   # sizes for A, B, C
    assert all(all(size % 2 == 0 for size in key) for key in options)
    assert options[(12, 8, 10)] <= options[(8, 6, 6)] <= arr.shape[1]


def test_three_populations_sfs(three_populations):
    _, dp, arr = three_populations
    downsampling = {"A": 8, "B": 6, "C": 6}
    sfs_2d = dp.numpy_to_2d_sfs(arr, downsampling, replicates=2)
    assert len(sfs_2d) == 2
    assert {key: s.shape for key, s in sfs_2d[0].items()} == {
        ("A", "B"): (9, 7), ("A", "C"): (9, 7), ("B", "C"): (7, 7)}
    msfs, sites = dp.numpy_to_msfs(arr, downsampling, replicates=2)
    assert len(msfs) == 2 and msfs[0].shape == (9 * 7 * 7,)
    assert 0 < sites <= arr.shape[1]
    # folded across all three populations: at most half of the 20 sampled alleles
    for cell in nonzero_cells(msfs[0], [8, 6, 6]):
        assert 0 < sum(map(int, cell.split("_"))) <= 10
    binned, _ = dp.numpy_to_msfs(arr, downsampling, replicates=2, nbins=4)
    assert binned[0].shape == (4 * 4 * 4,)
    assert binned[0].sum() > 0   # separate random draws, so totals differ a little from msfs