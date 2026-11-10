"""Tests for popai/simulate_data.py (DataSimulator).

Run from the top popai folder:  python -m pytest tests/test_simulate_data.py -v

The models come from the small test data (tests/species_tree.nex, tests/alignments,
tests/populations.txt: populations A, B and C with 12, 8 and 10 haplotypes, 10 loci).
That gives 7 models (labels 0-6), with 3 replicates each. Model 0 merges A and B into AB,
and model 1 merges all three populations into ABC. The simulations that several tests
look at are run once, at the top of the file (the `simulated` fixture).
"""
import copy
import os
import pickle

import msprime
import numpy as np
import pytest

from popai.generate_models import ModelBuilder
from popai.parse_input import ModelConfigParser
from popai.process_user_models import ModelReader
from popai.simulate_data import DataSimulator
from popai.utils import multi_sfs, sort_haplotypes, two_d_sfs

CONFIG = """[Model]
species tree file = ./tests/species_tree.nex
migration matrix = ./tests/migration.txt
symmetric = True
secondary contact = True
divergence with gene flow = False
max migration events = 1
migration rate = U(1e-5, 1e-4)
constant Ne = True

[Other]
seed = 1234
replicates = 3

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
alignments = ./tests/alignments
popfile = ./tests/populations.txt
"""

SIZES = {"A": 12, "B": 8, "C": 10}         # haplotypes per population in tests/populations.txt
DOWNSAMPLING = {"A": 8, "B": 4, "C": 6}
MAX_SITES = 300
N_LOCI = 10
REPLICATES = 3


# ---------------------------------------------------------------- shared setup

@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """The parsed config and the models, labels and tree indices built from it."""
    path = tmp_path_factory.mktemp("config") / "config.txt"
    path.write_text(CONFIG)
    config = ModelConfigParser(str(path)).parse_config()
    builder = ModelBuilder(config)
    models, labels, sp_tree_index = builder.draw_parameters(*builder.build_models())
    return config, models, labels, sp_tree_index


def make_simulator(setup, output, **options):
    """A DataSimulator for the test models, with the defaults used in these tests."""
    config, models, labels, sp_tree_index = setup
    arguments = dict(cores=1, downsampling=dict(DOWNSAMPLING), max_sites=MAX_SITES,
                     output=str(output), sp_tree_index=sp_tree_index)
    arguments.update(options)
    return DataSimulator(models, labels, config=dict(config), **arguments)


@pytest.fixture(scope="module")
def simulated(setup, tmp_path_factory):
    """Simulate every model once, with 1 core. Returns the simulator and its output folder."""
    output = tmp_path_factory.mktemp("simulated")
    simulator = make_simulator(setup, output)
    simulator.simulate_ancestry_parallel()
    return simulator, output


def read_arrays(output, label):
    with open(os.path.join(output, f"simulated_arrays_{label}.pickle"), "rb") as f:
        return pickle.load(f)


def model_with(setup, populations):
    """The first model (and its tree) whose populations are exactly `populations`."""
    config, models, labels, sp_tree_index = setup
    for model, index in zip(models, sp_tree_index):
        if [population.name for population in model.populations] == populations:
            return model, config["species tree"][index]
    raise AssertionError(f"no model with populations {populations}")


# ---------------------------------------------------------------- checks in __init__

def test_needs_tree_index_or_user_models(setup, tmp_path):
    with pytest.raises(ValueError, match="species tree index"):
        make_simulator(setup, tmp_path, sp_tree_index=False)


@pytest.mark.parametrize("downsampling", [
    {"A": 8, "B": 4},                         # C left out
    {"A": 8, "B": 4, "C": 6, "D": 2},         # D isn't a population
])
def test_downsampling_needs_every_population(setup, tmp_path, downsampling):
    with pytest.raises(ValueError, match="one entry for each population"):
        make_simulator(setup, tmp_path, downsampling=downsampling)


@pytest.mark.parametrize("downsampling", [
    {"A": 7, "B": 4, "C": 6},                 # odd
    {"A": 14, "B": 4, "C": 6},                # A has only 12 haplotypes
    {"A": 0, "B": 4, "C": 6},                 # zero
])
def test_downsampling_must_be_even_and_fit(setup, tmp_path, downsampling):
    with pytest.raises(ValueError, match="even number"):
        make_simulator(setup, tmp_path, downsampling=downsampling)


def test_downsampling_can_use_every_haplotype(setup, tmp_path):
    make_simulator(setup, tmp_path, downsampling=dict(SIZES))


def test_no_max_sites_keeps_every_snp(setup, tmp_path):
    assert make_simulator(setup, tmp_path, max_sites=None).max_sites == np.inf


def test_infinite_keeps_one_snp_per_locus(setup, tmp_path):
    assert make_simulator(setup, tmp_path, infinite=True).max_sites == N_LOCI


# ---------------------------------------------------------------- random draws

def test_each_replicate_gets_its_own_draws(setup, tmp_path):
    """The old bug: every replicate used the same mutation rate and seeds."""
    simulator = make_simulator(setup, tmp_path)
    draws = [simulator._draw_random_values() for _ in range(20)]
    rates = [draw["mutation rate"] for draw in draws]
    print("mutation rates:", np.round(rates, 10))
    assert len(set(rates)) == 20
    assert all(1e-8 <= rate <= 1e-7 for rate in rates)
    assert len({draw["ancestry seed"] for draw in draws}) == 20
    assert all(len(draw["mutation seeds"]) == N_LOCI for draw in draws)
    assert "infinite seed" not in draws[0]


def test_draws_repeat_with_the_same_seed(setup, tmp_path):
    first = make_simulator(setup, tmp_path)
    second = make_simulator(setup, tmp_path)
    for _ in range(5):
        a, b = first._draw_random_values(), second._draw_random_values()
        assert a["mutation rate"] == b["mutation rate"]
        assert a["ancestry seed"] == b["ancestry seed"]
        assert np.array_equal(a["mutation seeds"], b["mutation seeds"])


def test_infinite_draws_a_seed_for_its_batches(setup, tmp_path):
    assert "infinite seed" in make_simulator(setup, tmp_path, infinite=True)._draw_random_values()


# ---------------------------------------------------------------- sample sets

def test_sample_sets_in_popfile_order(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path)
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    sample_sets = simulator._sample_sets(model, tree)
    print([(s.population, s.num_samples, s.ploidy) for s in sample_sets])
    assert [s.population for s in sample_sets] == ["A", "B", "C"]
    assert [s.num_samples for s in sample_sets] == [6, 4, 5]      # diploid individuals
    assert all(s.ploidy == 2 for s in sample_sets)


@pytest.mark.parametrize("populations, sources", [
    (["AB", "C", "ABC"], ["AB", "AB", "C"]),      # model 0: A and B merged
    (["ABC"], ["ABC", "ABC", "ABC"]),             # model 1: one population
])
def test_merged_populations_sampled_from_their_ancestor(setup, tmp_path, populations, sources):
    simulator = make_simulator(setup, tmp_path)
    model, tree = model_with(setup, populations)
    sample_sets = simulator._sample_sets(model, tree)
    assert [s.population for s in sample_sets] == sources
    assert [s.num_samples for s in sample_sets] == [6, 4, 5]


def user_style_demography(split_time=0):
    """A demography like a user model: A and B split from AB at split_time, AB and C from ABC."""
    demography = msprime.Demography()
    for name in ["A", "B", "C", "AB", "ABC"]:
        demography.add_population(name=name, initial_size=10000)
    demography.add_population_split(time=split_time, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=50000, derived=["AB", "C"], ancestral="ABC")
    return demography


def test_user_model_split_at_time_zero_is_a_merge(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path)
    merged = simulator._sample_sets(user_style_demography(split_time=0), None)
    assert [s.population for s in merged] == ["AB", "AB", "C"]
    separate = simulator._sample_sets(user_style_demography(split_time=100), None)
    assert [s.population for s in separate] == ["A", "B", "C"]


def test_population_missing_from_model(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path)
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10000)
    demography.add_population(name="B", initial_size=10000)
    with pytest.raises(ValueError, match="population C in the popfile"):
        simulator._sample_sets(demography, None)


def test_odd_number_of_haplotypes(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path)
    simulator.config["sampling dict"] = {"A": 11, "B": 8, "C": 10}
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    with pytest.raises(ValueError, match="even number of haplotypes"):
        simulator._sample_sets(model, tree)


# ---------------------------------------------------------------- one replicate

def test_replicate_shape_and_encoding(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path)
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    array, n_snps = simulator._simulate_demography(model, tree, simulator._draw_random_values())
    print("replicate array:", array.shape)
    assert array.shape == (sum(SIZES.values()), n_snps)
    assert n_snps > 0
    assert array.min() >= 0                               # no missing data before max_sites
    # minor encoding: 0 is the most common allele at every site
    assert ((array == 0).sum(axis=0) >= (array == 1).sum(axis=0)).all()


def test_replicate_repeats_with_the_same_draws(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path)
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    draws = simulator._draw_random_values()
    first, _ = simulator._simulate_demography(model, tree, draws)
    again, _ = simulator._simulate_demography(model, tree, draws)
    other, _ = simulator._simulate_demography(model, tree, simulator._draw_random_values())
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_each_locus_simulated_at_its_own_length(setup, tmp_path, monkeypatch):
    """Record where each locus's mutations land: every one must be inside its own length."""
    simulator = make_simulator(setup, tmp_path)
    lengths = [50, 5000, 50]
    simulator.config["lengths"] = lengths
    draws = simulator._draw_random_values()
    draws["mutation rate"] = 1e-6                         # high, so every locus gets mutations

    positions = []
    real_sim_mutations = msprime.sim_mutations

    def recording_sim_mutations(*args, **kwargs):
        mts = real_sim_mutations(*args, **kwargs)
        positions.append(mts.tables.sites.position)
        return mts

    monkeypatch.setattr(msprime, "sim_mutations", recording_sim_mutations)
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    simulator._simulate_demography(model, tree, draws)

    print("last mutation position per locus:", [p.max() for p in positions])
    assert len(positions) == len(lengths)
    for locus_positions, length in zip(positions, lengths):
        assert len(locus_positions) > 0
        assert locus_positions.max() < length
    assert positions[1].max() > 50                        # the long locus uses its whole length


def test_infinite_one_snp_from_each_locus(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path, infinite=True)
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    array, n_snps = simulator._simulate_demography(model, tree, simulator._draw_random_values())
    print("infinite-sites array:\n", array)
    assert array.shape == (sum(SIZES.values()), N_LOCI)
    assert set(np.unique(array)) == {0, 1}
    assert ((array == 1).sum(axis=0) > 0).all()           # every column is a SNP
    # the old bug reused seeds, so most loci were copies of each other
    assert len(np.unique(array, axis=1).T) > N_LOCI // 2


def test_infinite_gives_up_without_mutations(setup, tmp_path):
    simulator = make_simulator(setup, tmp_path, infinite=True)
    simulator.config["lengths"] = [100]                   # one locus keeps this quick
    draws = simulator._draw_random_values()
    draws["mutation rate"] = 0
    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    with pytest.raises(RuntimeError, match="mutation rate"):
        simulator._simulate_demography(model, tree, draws)


def test_copy_sent_to_workers_leaves_out_models(setup, tmp_path):
    """Each task copies the simulator to a worker. The copy must leave out the models, and
    still simulate the same replicate."""
    simulator = make_simulator(setup, tmp_path)
    state = simulator.__getstate__()
    assert "models" not in state and "labels" not in state and "sp_tree_index" not in state
    assert set(state["config"]) == {"sampling dict", "lengths", "substitution model"}

    model, tree = model_with(setup, ["A", "B", "AB", "C", "ABC"])
    draws = simulator._draw_random_values()
    worker_copy = pickle.loads(pickle.dumps(simulator))
    expected, _ = simulator._simulate_demography(model, tree, draws)
    result, _ = worker_copy._simulate_demography(model, tree, draws)
    assert np.array_equal(expected, result)


# ---------------------------------------------------------------- simulate_ancestry_parallel

def test_one_array_file_per_model(setup, simulated):
    simulator, output = simulated
    labels = sorted(set(setup[2]))
    for label in labels:
        arrays = read_arrays(output, label)
        print(f"model {label}: {[a.shape for a in arrays]}")
        assert len(arrays) == REPLICATES
        assert all(a.shape == (sum(SIZES.values()), MAX_SITES) for a in arrays)


def test_max_sites_cuts_or_pads(setup, simulated):
    """SNPs come first, with no missing data; any padding after them is all -1."""
    simulator, output = simulated
    for label in sorted(set(setup[2])):
        for array in read_arrays(output, label):
            padding = (array == -1).all(axis=0)
            n_snps = int((~padding).sum())
            assert not padding[:n_snps].any()             # padding only at the end
            assert (array[:, :n_snps] >= 0).all()


def test_rows_sorted_within_each_population(setup, simulated):
    simulator, output = simulated
    for label in sorted(set(setup[2])):
        for array in read_arrays(output, label):
            assert np.array_equal(array, sort_haplotypes(array, SIZES.values()))


def test_replicates_differ(setup, simulated):
    simulator, output = simulated
    for label in sorted(set(setup[2])):
        arrays = read_arrays(output, label)
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                assert not np.array_equal(arrays[i], arrays[j])


def test_same_results_with_more_cores(setup, simulated, tmp_path):
    simulator, output = simulated
    make_simulator(setup, tmp_path, cores=2).simulate_ancestry_parallel()
    for label in sorted(set(setup[2])):
        for one_core, two_cores in zip(read_arrays(output, label), read_arrays(tmp_path, label)):
            assert np.array_equal(one_core, two_cores)


def test_checkpoint_skips_finished_models(setup, simulated, tmp_path):
    """A model whose file exists is skipped, and the other models come out the same as
    without the checkpoint (the skipped model's random values are still drawn)."""
    simulator, output = simulated
    labels = sorted(set(setup[2]))
    with open(tmp_path / f"simulated_arrays_{labels[0]}.pickle", "wb") as f:
        pickle.dump("already done", f)

    make_simulator(setup, tmp_path, checkpoint=True).simulate_ancestry_parallel()

    assert read_arrays(tmp_path, labels[0]) == "already done"
    for label in labels[1:]:
        for expected, result in zip(read_arrays(output, label), read_arrays(tmp_path, label)):
            assert np.array_equal(expected, result)


# ---------------------------------------------------------------- SFS and plots

def test_msfs_shape_and_files(setup, simulated):
    simulator, output = simulated
    msfs = simulator.mutations_to_sfs(None)
    labels = [str(label) for label in dict.fromkeys(setup[2])]
    assert list(msfs) == labels
    for label in labels:
        assert len(msfs[label]) == REPLICATES
        assert all(sfs.shape == (9 * 5 * 7,) for sfs in msfs[label])
        with open(os.path.join(output, f"simulated_mSFS_{label}.pickle"), "rb") as f:
            saved = pickle.load(f)
        assert all(np.array_equal(a, b) for a, b in zip(saved, msfs[label]))


def test_binned_msfs_shape(simulated):
    simulator, output = simulated
    msfs = simulator.mutations_to_sfs(None, nbins=4)
    assert all(sfs.shape == (4 * 4 * 4,) for replicates in msfs.values() for sfs in replicates)


def test_2d_sfs_shape_and_files(setup, simulated):
    simulator, output = simulated
    sfs_2d = simulator.mutations_to_2d_sfs()
    shapes = {("A", "B"): (9, 5), ("A", "C"): (9, 7), ("B", "C"): (5, 7)}
    for label, replicates in sfs_2d.items():
        assert len(replicates) == REPLICATES
        for replicate in replicates:
            assert {pair: sfs.shape for pair, sfs in replicate.items()} == shapes
        with open(os.path.join(output, f"simulated_2dSFS_{label}.pickle"), "rb") as f:
            saved = pickle.load(f)
        assert len(saved) == REPLICATES
        assert [sfs.shape for sfs in saved[0]] == list(shapes.values())


def test_sfs_match_the_empirical_code(setup, simulated):
    """The simulated SFS are built by the same functions as the empirical SFS."""
    simulator, output = simulated
    config = setup[0]
    rng = copy.deepcopy(simulator.rng)       # the same random numbers the simulator will use
    msfs = simulator.mutations_to_sfs(None)
    for label in dict.fromkeys(setup[2]):
        for array, sfs in zip(read_arrays(output, label), msfs[str(label)]):
            expected = multi_sfs(array, DOWNSAMPLING, config["sampling dict"], rng)[0]
            assert np.array_equal(sfs, expected)

    rng = copy.deepcopy(simulator.rng)
    sfs_2d = simulator.mutations_to_2d_sfs()
    for label in dict.fromkeys(setup[2]):
        for array, sfs in zip(read_arrays(output, label), sfs_2d[str(label)]):
            expected = two_d_sfs(array, DOWNSAMPLING, config["sampling dict"], rng)[0]
            for pair in expected:
                assert np.array_equal(sfs[pair], expected[pair])


def test_plot_2dsfs_one_png_per_model_and_pair(setup, simulated, tmp_path):
    simulator, output = simulated
    simulator.plot_2dsfs(simulator.mutations_to_2d_sfs(), output_directory=str(tmp_path))
    expected = {f"2D_SFS_{pair}_{label}.png"
                for label in set(setup[2]) for pair in [("A", "B"), ("A", "C"), ("B", "C")]}
    assert set(os.listdir(tmp_path)) == expected


# ---------------------------------------------------------------- user models

USER_CONFIG = """[Model]
user models = {models}

[Other]
seed = 1234
replicates = 2

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
alignments = ./tests/alignments
popfile = ./tests/populations.txt
"""

POPULATIONS = """[Populations]
A=[10000,50000]
B=[10000,50000]
C=[10000,50000]
AB=[10000,50000]
ABC=[10000,50000]

"""

USER_MODELS = {
    # A and B are one population (split at time 0)
    "model_0.model": POPULATIONS + """[Events]
div_AB = split{time=0, descendants=[A,B], ancestor=AB}
div_ABC = split{time=[500000,1000000], descendants=[AB,C], ancestor=ABC}
""",
    # A and B are sister populations, with gene flow between them
    "model_1.model": POPULATIONS + """[Events]
div_AB = split{time=[10000,50000], descendants=[A,B], ancestor=AB}
mig_AB = symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=[1e-5,1e-4]}
div_ABC = split{time=[500000,1000000], descendants=[AB,C], ancestor=ABC}
""",
}


def test_user_models(tmp_path):
    models_folder = tmp_path / "models"
    models_folder.mkdir()
    for name, text in USER_MODELS.items():
        (models_folder / name).write_text(text)
    config_path = tmp_path / "config.txt"
    config_path.write_text(USER_CONFIG.format(models=models_folder))
    config = ModelConfigParser(str(config_path)).parse_config()

    models, labels = ModelReader(config_values=config).read_models()
    output = tmp_path / "output"
    output.mkdir()
    simulator = DataSimulator(models, labels, config=config, cores=1, downsampling=dict(DOWNSAMPLING),
                              max_sites=MAX_SITES, user=True, output=str(output))
    simulator.simulate_ancestry_parallel()

    for label in ["model_0", "model_1"]:
        arrays = read_arrays(output, label)
        assert len(arrays) == 2
        assert all(a.shape == (sum(SIZES.values()), MAX_SITES) for a in arrays)
    msfs = simulator.mutations_to_sfs(None)
    assert list(msfs) == ["model_0", "model_1"]
