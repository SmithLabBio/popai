"""Run train_models and apply_models (popai/cli_train_models.py, popai/cli_apply_models.py,
popai/build_predictors.py, popai/dataset.py) on small simulations, so any print() lines you add
there show up, and check the fixes from the build_predictors review.

Run from the top popai folder:  python try_build_predictors.py
It takes a few minutes. Everything goes in try_build_predictors/, which is deleted first.

It uses the test data (tests/species_tree.nex, tests/alignments, tests/populations.txt), with
REPLICATES replicates per model, and:
  1. processes the empirical data with 12 replicates (issue 3: rep1 vs rep10, rep11)
  2. simulates the 7 species-tree models
  3. compares the in-memory and --low-memory datasets (issues 5 and 6)
  4. trains all four predictors twice with the same seed (issues 7 and 2)
  5. applies them, twice, and compares the predictions (issues 7, 3 and 4)
  6. trains and applies the FCNN on user models numbered 1 and 2, not 0 and 1 (issue 1)
  7. gives train_models a typo in --fcnn (issue 8)
Each step prints what it found. If a step fails, its error is printed and the next steps run.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")   # save plots to files, without opening windows

import glob
import pickle
import shutil
import sys
import traceback

import numpy as np

FOLDER = "try_build_predictors"
REPLICATES = 20          # per model; the networks hold out 20% for testing
MAX_SITES = 300
DOWNSAMPLING = "{'A': 8, 'B': 4, 'C': 6}"
EMPIRICAL_REPS = 12      # more than 10, to check that rep1 doesn't read rep10 or rep11
TRAINING = dict(rf="ntrees=50", fcnn="epochs=3", cnn="epochs=3", cnnnpy="epochs=2")

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
replicates = {replicates}

[Simulations]
mutation rate = U(1e-8, 1e-7)
substitution model = JC69

[Data]
alignments = ./tests/alignments
popfile = ./tests/populations.txt
"""

USER_CONFIG = """[Model]
user models = {models}

[Other]
seed = 1234
replicates = {replicates}

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
USER_MODELS = {   # numbered 1 and 2 on purpose (issue 1)
    "model_1.model": POPULATIONS + """[Events]
div_AB = split{time=0, descendants=[A,B], ancestor=AB}
div_ABC = split{time=[500000,1000000], descendants=[AB,C], ancestor=ABC}
""",
    "model_2.model": POPULATIONS + """[Events]
div_AB = split{time=[10000,50000], descendants=[A,B], ancestor=AB}
mig_AB = symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=[1e-5,1e-4]}
div_ABC = split{time=[500000,1000000], descendants=[AB,C], ancestor=ABC}
""",
}


def section(title):
    print(f"\n========== {title} ==========", flush=True)


def path(*parts):
    return os.path.join(FOLDER, *parts)


def run_cli(module, arguments):
    """Run a popai command's main() in this process, as if typed on the command line."""
    sys.argv = [module.__name__.split(".")[-1]] + arguments
    print("$", " ".join(sys.argv), flush=True)
    module.main()


def step(title, function):
    """Run one step, printing its error (if any) and carrying on."""
    section(title)
    try:
        function()
    except SystemExit as error:
        print(f"(exited with code {error.code})")
    except Exception:
        print("STEP FAILED:")
        traceback.print_exc()


def show(file_name, max_lines=20):
    with open(file_name) as f:
        lines = f.read().splitlines()
    print(f"--- {file_name}")
    print("\n".join(lines[:max_lines]) + ("\n..." if len(lines) > max_lines else ""))


# ---------------------------------------------------------------- the steps

def empirical():
    from popai import cli_process_empirical_data
    run_cli(cli_process_empirical_data, ["--config", path("config.txt"), "--downsampling", DOWNSAMPLING,
                                         "--reps", str(EMPIRICAL_REPS), "--maxsites", str(MAX_SITES),
                                         "--output", path("empirical")])
    array = np.load(path("empirical", "empirical.npy"))
    print("empirical.npy:", array.shape, array.dtype, "(should be int8)")
    print("jSFS files:", len(glob.glob(path("empirical", "*.jsfs"))), f"(3 pairs x {EMPIRICAL_REPS} replicates)")


def simulate():
    from popai import cli_simulate_data
    run_cli(cli_simulate_data, ["--config", path("config.txt"), "--downsampling", DOWNSAMPLING,
                                "--maxsites", str(MAX_SITES), "--simulate", "--cores", "2",
                                "--output", path("simulated")])
    with open(path("simulated", "simulated_arrays_0.pickle"), "rb") as f:
        arrays = pickle.load(f)
    print(f"model 0: {len(arrays)} arrays, shape {arrays[0].shape}, dtypes {sorted({a.dtype.name for a in arrays})} (should be int8)")


def datasets():
    from popai.dataset import PopaiDataset, PopaiDatasetLowMem
    paths = sorted(glob.glob(path("simulated", "simulated_arrays_*.pickle")))
    in_memory, low_memory = PopaiDataset(paths), PopaiDatasetLowMem(paths)
    print("datasets:", len(in_memory), "in memory,", len(low_memory), "low memory")
    print("model numbers:", in_memory.model_numbers, " classes:", in_memory.n_classes)
    same = all(np.array_equal(in_memory[i][0], low_memory[i][0]) and np.array_equal(in_memory[i][1], low_memory[i][1])
               for i in range(len(in_memory)))
    print("same datasets and labels, in the same order:", same)


def train(name):
    from popai import cli_train_models
    arguments = ["--config", path("config.txt"), "--simulations", path("simulated"), "--output", path(name)]
    for network, settings in TRAINING.items():
        arguments += [f"--{network}", settings]
    run_cli(cli_train_models, arguments)
    print("files:", sorted(os.listdir(path(name))))
    for tsv in sorted(glob.glob(path(name, "*_confusion.tsv"))):
        show(tsv)                                        # issue 2: one row and column per model


def apply(models, name):
    from popai import cli_apply_models
    run_cli(cli_apply_models, ["--config", path("config.txt"), "--models", path(models),
                               "--empirical", path("empirical"), "--simulations", path("simulated"),
                               "--output", path(name), "--rf", "--fcnn", "--cnn", "--cnnnpy"])
    print("files:", sorted(os.listdir(path(name))))      # issue 4: the *_pca.png files
    show(path(name, "cnn_predictions.txt"), max_lines=12)


def compare_runs():
    """Issue 7: the same seed should give the same trained predictors and predictions."""
    for network in ["rf", "fcnn", "cnn", "cnn_npy"]:
        with open(path("applied_a", f"{network}_predictions.txt")) as f:
            first = f.read()
        with open(path("applied_b", f"{network}_predictions.txt")) as f:
            second = f.read()
        line = f"{network}: predictions identical in both runs: {first == second}"
        if network != "rf":
            a = np.load(path("trained_a", f"{network}_features.npy"))
            b = np.load(path("trained_b", f"{network}_features.npy"))
            line += f"; largest difference in the training features: {np.abs(a - b).max():.3g}"
        print(line)


def empirical_jsfs_rows():
    """Issue 3: one prediction per empirical replicate, each from its own jSFS files."""
    with open(path("applied_a", "cnn_predictions.txt")) as f:
        rows = [line for line in f if line.startswith("│ Replicate")]
    print(f"CNN predictions for {len(rows)} replicates (should be {EMPIRICAL_REPS})")


def user_models():
    """Issue 1: user models numbered 1 and 2 (not 0 and 1)."""
    from popai import cli_simulate_data, cli_train_models, cli_apply_models
    models_folder = path("user_models")
    os.makedirs(models_folder)
    for name, text in USER_MODELS.items():
        with open(os.path.join(models_folder, name), "w") as f:
            f.write(text)
    with open(path("user_config.txt"), "w") as f:
        f.write(USER_CONFIG.format(models=models_folder, replicates=REPLICATES))
    run_cli(cli_simulate_data, ["--config", path("user_config.txt"), "--downsampling", DOWNSAMPLING,
                                "--maxsites", str(MAX_SITES), "--simulate", "--output", path("user_simulated")])
    run_cli(cli_train_models, ["--config", path("user_config.txt"), "--simulations", path("user_simulated"),
                               "--output", path("user_trained"), "--fcnn", "epochs=3", "--rf", "ntrees=50"])
    show(path("user_trained", "fcnn_confusion.tsv"))       # rows and columns 1 and 2
    run_cli(cli_apply_models, ["--config", path("user_config.txt"), "--models", path("user_trained"),
                               "--empirical", path("empirical"), "--simulations", path("user_simulated"),
                               "--output", path("user_applied"), "--fcnn", "--rf"])
    show(path("user_applied", "fcnn_predictions.txt"), max_lines=6)   # headers Model 1, Model 2
    show(path("user_applied", "rf_predictions.txt"), max_lines=6)


def typo():
    """Issue 8: a typo in --fcnn stops train_models with a one-line error."""
    from popai import cli_train_models
    run_cli(cli_train_models, ["--config", path("config.txt"), "--simulations", path("simulated"),
                               "--output", path("typo"), "--fcnn", "epoch=3"])


if __name__ == "__main__":   # needed for simulating with several cores
    shutil.rmtree(FOLDER, ignore_errors=True)
    os.makedirs(FOLDER)
    with open(path("config.txt"), "w") as f:
        f.write(CONFIG.format(replicates=REPLICATES))

    step("1. process_empirical_data, 12 replicates", empirical)
    step("2. simulate_data", simulate)
    step("3. in-memory and --low-memory datasets", datasets)
    step("4a. train_models, first run", lambda: train("trained_a"))
    step("4b. train_models, second run (same seed)", lambda: train("trained_b"))
    step("5a. apply_models, first run", lambda: apply("trained_a", "applied_a"))
    step("5b. apply_models, second run", lambda: apply("trained_b", "applied_b"))
    step("5c. same seed, same results?", compare_runs)
    step("5d. one prediction per empirical replicate", empirical_jsfs_rows)
    step("6. user models numbered 1 and 2", user_models)
    step("7. a typo in --fcnn", typo)
