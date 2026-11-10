"""Tests for training and applying predictors: popai/build_predictors.py, popai/dataset.py,
popai/cli_train_models.py and popai/cli_apply_models.py.

Run from the top popai folder:  python -m pytest tests/test_build_predictors.py -v
The whole file is skipped if TensorFlow or PyTorch isn't installed.

The first group of tests uses small made-up simulation files. The last group runs the whole
pipeline once on the test data (tests/species_tree.nex, tests/alignments,
tests/populations.txt): process_empirical_data with 3 replicates, simulate_data with 10
replicates of each of the 7 models, then train_models and apply_models with 1 epoch, so the
predictors are poor; the tests only check that everything runs and fits together.
"""
import argparse
import contextlib
import io
import os
import pickle
import shutil
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")
import torch  # noqa: E402

from popai import cli_apply_models, cli_process_empirical_data, cli_simulate_data, cli_train_models  # noqa: E402
from popai.build_predictors import plot_confusion_matrix, read_data, write_confusion_tsv  # noqa: E402
from popai.cli_train_models import NETWORK_SETTINGS, RF_SETTINGS, _key_value_action  # noqa: E402
from popai.dataset import PopaiDataset, PopaiDatasetLowMem, PopaiTrainingData, custom_collate_fn  # noqa: E402

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
replicates = 10

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
alignments = ./tests/alignments
popfile = ./tests/populations.txt
"""
DOWNSAMPLING = "{'A': 8, 'B': 4, 'C': 6}"
MAX_SITES = "50"
EMPIRICAL_REPS = 3
N_MODELS = 7
REPLICATES = 10


def write_simulations(folder, kind, replicates_per_model):
    """Write made-up simulated_<kind>_<name>.pickle files: {name: list of replicates}."""
    os.makedirs(folder, exist_ok=True)
    for name, replicates in replicates_per_model.items():
        with open(os.path.join(folder, f"simulated_{kind}_{name}.pickle"), "wb") as f:
            pickle.dump(replicates, f)


def run_cli(module, arguments):
    sys.argv = [module.__name__.split(".")[-1]] + [str(a) for a in arguments]
    module.main()


def prediction_rows(path):
    """The rows of a predictions table (tabulate's fancy_grid format), as arrays of probabilities."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("│ Replicate"):
                cells = [cell.strip() for cell in line.strip().strip("│").split("│")]
                rows.append(np.array([float(cell) for cell in cells[1:]]))
    return rows


def read_tsv(path):
    with open(path) as f:
        lines = [line.rstrip("\n").split("\t") for line in f]
    header = lines[0][1:]
    names = [line[0] for line in lines[1:]]
    counts = np.array([[int(value) for value in line[1:]] for line in lines[1:]])
    return header, names, counts


# ---------------------------------------------------------------- read_data (random forest)

def test_read_data_numbers_models_in_order(tmp_path):
    """User models numbered 3 and 1 become classes 1 -> 0 and 3 -> 1."""
    write_simulations(tmp_path, "mSFS", {"model_3": [np.full(5, 3)] * 4, "model_1": [np.full(5, 1)] * 4})
    arraydict, sfs, labels, label_to_int, int_to_label, nclasses = read_data(str(tmp_path))
    print("int_to_label:", int_to_label)
    assert int_to_label == {0: 1, 1: 3}
    assert label_to_int == {1: 0, 3: 1}
    assert nclasses == 2
    assert sfs.shape == (8, 5)
    assert (sfs[:4] == 1).all() and (sfs[4:] == 3).all()            # model 1's replicates first
    assert np.array_equal(labels, np.repeat(np.eye(2), 4, axis=0))


def test_read_data_without_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="simulated_mSFS"):
        read_data(str(tmp_path))


# ---------------------------------------------------------------- datasets (networks)

def test_dataset_labels_are_classes(tmp_path):
    """Model numbers that don't start at 0 still give classes 0 .. n-1 and one-hot labels with
    one column per model, in memory and with --low-memory."""
    write_simulations(tmp_path, "mSFS", {"model_1": [np.full(5, 1.0)] * 3, "model_3": [np.full(5, 3.0)] * 3})
    paths = sorted(str(p) for p in tmp_path.iterdir())
    in_memory, low_memory = PopaiDataset(paths), PopaiDatasetLowMem(paths)
    for dataset in (in_memory, low_memory):
        assert dataset.model_numbers == [1, 3]
        assert dataset.n_classes == 2
        assert dataset.labels == [0, 0, 0, 1, 1, 1]
        assert dataset.encoded_labels.shape == (6, 2)
    for i in range(6):
        assert np.array_equal(in_memory[i][0], low_memory[i][0])
        assert np.array_equal(in_memory[i][1], low_memory[i][1])


def test_collate_stacks_each_pair():
    """The CNN's batches: one float32 tensor per population pair, each SFS summing to 1."""
    rng = np.random.default_rng(1)
    shapes = [(9, 5), (9, 7), (5, 7)]
    batch = [([rng.integers(1, 10, shape) for shape in shapes], np.eye(3)[i]) for i in range(2)]
    pairs, labels = custom_collate_fn(batch)
    assert [tuple(pair.shape) for pair in pairs] == [(2, 9, 5), (2, 9, 7), (2, 5, 7)]
    assert all(pair.dtype == torch.float32 for pair in pairs)
    assert all(np.allclose(pair.numpy().sum(axis=(1, 2)), 1) for pair in pairs)
    assert tuple(labels.shape) == (2, 3) and labels.dtype == torch.float32


def test_training_data_split_and_shuffle_repeat(tmp_path):
    """The same seed gives the same train/test split and the same shuffled batches; the test
    set has 20% of each model's datasets."""
    write_simulations(tmp_path, "mSFS", {str(m): [np.full(5, float(m * 100 + r)) for r in range(10)] for m in range(3)})
    first = PopaiTrainingData(str(tmp_path), "simulated_mSFS_*.pickle", seed=7, batch_size=4)
    second = PopaiTrainingData(str(tmp_path), "simulated_mSFS_*.pickle", seed=7, batch_size=4)
    assert list(first.test_dataset.indices) == list(second.test_dataset.indices)
    test_labels = [first.dataset.labels[i] for i in first.test_dataset.indices]
    assert sorted(test_labels) == [0, 0, 1, 1, 2, 2]
    first_order = [x.numpy()[:, 0].tolist() for x, _ in first.train_loader]
    second_order = [x.numpy()[:, 0].tolist() for x, _ in second.train_loader]
    print("first epoch's batches:", first_order)
    assert first_order == second_order


# ---------------------------------------------------------------- confusion matrices

def test_confusion_matrix_has_every_model():
    """Model 2 is never true or predicted, but still gets a row and a column."""
    conf_matrix, plot = plot_confusion_matrix([0, 0, 1, 1], [0, 0, 0, 1], labels=[0, 1, 2], names=["1", "2", "3"])
    plot.close()
    assert conf_matrix.tolist() == [[2, 0, 0], [1, 1, 0], [0, 0, 0]]


def test_write_confusion_tsv(tmp_path):
    path = tmp_path / "confusion.tsv"
    write_confusion_tsv(np.array([[2, 0], [1, 1]]), ["1", "3"], str(path))
    assert path.read_text() == "\t1\t3\n1\t2\t0\n3\t1\t1\n"


# ---------------------------------------------------------------- --cnn/--fcnn/--cnnnpy/--rf settings

def settings_parser():
    parser = argparse.ArgumentParser(prog="train_models")
    parser.add_argument("--fcnn", action=_key_value_action(NETWORK_SETTINGS), nargs="?")
    parser.add_argument("--rf", action=_key_value_action(RF_SETTINGS), nargs="?")
    return parser


@pytest.mark.parametrize("arguments, expected", [
    (["--fcnn"], {"epochs": 10, "batch_size": 10, "learning_rate": 0.001}),
    (["--fcnn", "epochs=20:batch_size=32"], {"epochs": 20, "batch_size": 32, "learning_rate": 0.001}),
    (["--fcnn", "learning_rate=1e-4:epochs=5"], {"epochs": 5, "batch_size": 10, "learning_rate": 1e-4}),
    (["--rf", "ntrees=100"], {"ntrees": 100}),
])
def test_settings(arguments, expected):
    args = settings_parser().parse_args(arguments)
    assert getattr(args, arguments[0][2:]) == expected


@pytest.mark.parametrize("arguments, message", [
    (["--fcnn", "epoch=20"], "can't read 'epoch=20'"),
    (["--fcnn", "epochs=20,batch_size=32"], "epochs must be a whole number"),   # ',' instead of ':' 
    (["--fcnn", "epochs=2.5"], "epochs must be a whole number"),
    (["--fcnn", "learning_rate=fast"], "learning_rate must be a number"),
    (["--fcnn", "epochs=0"], "epochs must be more than 0"),
    (["--rf", "trees=100"], "can't read 'trees=100'"),
])
def test_bad_settings(arguments, message):
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors), pytest.raises(SystemExit):
        settings_parser().parse_args(arguments)
    print(errors.getvalue().strip().splitlines()[-1])
    assert message in errors.getvalue()


# ---------------------------------------------------------------- the whole pipeline

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Process the empirical data and simulate data once, for the tests below."""
    folder = tmp_path_factory.mktemp("pipeline")
    config = folder / "config.txt"
    config.write_text(CONFIG)
    run_cli(cli_process_empirical_data, ["--config", config, "--downsampling", DOWNSAMPLING,
                                         "--reps", EMPIRICAL_REPS, "--maxsites", MAX_SITES,
                                         "--output", folder / "empirical"])
    run_cli(cli_simulate_data, ["--config", config, "--downsampling", DOWNSAMPLING, "--maxsites", MAX_SITES,
                                "--simulate", "--output", folder / "simulated"])
    return folder


def train(pipeline, output, simulations="simulated", predictors=("--rf", "ntrees=20", "--fcnn", "epochs=1",
                                                                  "--cnn", "epochs=1", "--cnnnpy", "epochs=1")):
    run_cli(cli_train_models, ["--config", pipeline / "config.txt", "--simulations", pipeline / simulations,
                               "--output", pipeline / output, *predictors])
    return pipeline / output


def apply(pipeline, models, output, simulations="simulated", predictors=("--rf", "--fcnn", "--cnn", "--cnnnpy")):
    run_cli(cli_apply_models, ["--config", pipeline / "config.txt", "--models", pipeline / models,
                               "--empirical", pipeline / "empirical", "--simulations", pipeline / simulations,
                               "--output", pipeline / output, *predictors])
    return pipeline / output


@pytest.fixture(scope="module")
def trained(pipeline):
    return train(pipeline, "trained")


def test_train_models_output(trained):
    expected = {"rf.model.pickle", "fcnn.keras", "cnn.keras", "cnn_npy.keras"}
    for predictor in ["rf", "fcnn", "cnn", "cnn_npy"]:
        expected |= {f"{predictor}_confusion.png", f"{predictor}_confusion.tsv"}
    for network in ["fcnn", "cnn", "cnn_npy"]:
        expected |= {f"{network}_features.npy", f"{network}_labels.npy"}
    assert set(os.listdir(trained)) == expected


@pytest.mark.parametrize("predictor", ["rf", "fcnn", "cnn", "cnn_npy"])
def test_confusion_tsv_has_one_row_and_column_per_model(trained, predictor):
    header, names, counts = read_tsv(trained / f"{predictor}_confusion.tsv")
    print(f"{predictor}_confusion.tsv:\n", counts)
    assert header == names == [str(model) for model in range(N_MODELS)]
    assert (counts.sum(axis=1) == REPLICATES * 0.2).all()        # 20% of each model's datasets


def test_apply_models_output(pipeline, trained):
    applied = apply(pipeline, "trained", "applied")
    expected = {"rf_predictions.txt"}
    for network in ["fcnn", "cnn", "cnn_npy"]:
        expected |= {f"{network}_predictions.txt", f"{network}_empirical_features.npy", f"{network}_pca.png"}
    assert set(os.listdir(applied)) == expected

    for predictor, n_rows in [("rf", EMPIRICAL_REPS), ("fcnn", EMPIRICAL_REPS), ("cnn", EMPIRICAL_REPS), ("cnn_npy", 1)]:
        rows = prediction_rows(applied / f"{predictor}_predictions.txt")
        print(predictor, [np.round(row, 3) for row in rows])
        assert len(rows) == n_rows
        assert all(len(row) == N_MODELS and np.isclose(row.sum(), 1, atol=1e-4) for row in rows)
    # each empirical replicate has its own jSFS (downsampled differently), so its own prediction
    cnn_rows = prediction_rows(applied / "cnn_predictions.txt")
    assert not all(np.array_equal(cnn_rows[0], row) for row in cnn_rows[1:])


def test_training_repeats_with_the_same_seed(pipeline, trained):
    again = train(pipeline, "trained_again")
    for network in ["fcnn", "cnn", "cnn_npy"]:
        assert np.array_equal(np.load(trained / f"{network}_features.npy"), np.load(again / f"{network}_features.npy"))
    for predictor in ["rf", "fcnn", "cnn", "cnn_npy"]:
        assert read_tsv(trained / f"{predictor}_confusion.tsv") [2].tolist() == read_tsv(again / f"{predictor}_confusion.tsv")[2].tolist()


def test_models_numbered_from_one(pipeline):
    """User models model_1 and model_5: the confusion matrices and predictions name them 1 and 5."""
    source, target = pipeline / "simulated", pipeline / "numbered"
    target.mkdir()
    for kind in ["mSFS", "2dSFS", "arrays"]:
        for old, new in [("0", "model_1"), ("1", "model_5")]:
            shutil.copy(source / f"simulated_{kind}_{old}.pickle", target / f"simulated_{kind}_{new}.pickle")
    trained = train(pipeline, "numbered_trained", simulations="numbered",
                    predictors=("--rf", "ntrees=20", "--fcnn", "epochs=1"))
    for predictor in ["rf", "fcnn"]:
        header, names, _ = read_tsv(trained / f"{predictor}_confusion.tsv")
        assert header == names == ["1", "5"]
    applied = apply(pipeline, "numbered_trained", "numbered_applied", simulations="numbered", predictors=("--rf", "--fcnn"))
    for predictor in ["rf", "fcnn"]:
        with open(applied / f"{predictor}_predictions.txt") as f:
            text = f.read()
        assert "Model 1" in text and "Model 5" in text and "Model 0" not in text
