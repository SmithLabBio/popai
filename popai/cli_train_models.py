import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input
from popai.dataset import PopaiTrainingData
from popai.build_predictors import CnnSFS, CnnNpy, NeuralNetSFS, RandomForestsSFS, train_model, test_model, write_confusion_tsv
from tensorflow import keras

def _key_value_action(defaults):
    """An argparse action for options like --cnn "epochs=20:batch_size=32": key=value settings
    separated by ':', each key one of those in defaults. The types and default values come from
    defaults. The option on its own uses the defaults."""
    kinds = {int: "a whole number", float: "a number"}
    class KeyValueAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            settings = dict(defaults)
            for item in (values.split(':') if values else []):
                key, _, value = item.partition('=')
                key = key.strip()
                if key not in defaults or not value.strip():
                    parser.error(f"{option_string}: can't read '{item}'. Give any of "
                                 f"{':'.join(f'{name}=...' for name in defaults)}, separated by ':'.")
                kind = type(defaults[key])
                try:
                    settings[key] = kind(value)
                except ValueError:
                    parser.error(f"{option_string}: {key} must be {kinds[kind]}, not '{value}'.")
                if settings[key] <= 0:
                    parser.error(f"{option_string}: {key} must be more than 0.")
            setattr(namespace, self.dest, settings)
    return KeyValueAction

NETWORK_SETTINGS = {'epochs': 10, 'batch_size': 10, 'learning_rate': 0.001}
RF_SETTINGS = {'ntrees': 500}

def main():
    parser = argparse.ArgumentParser(description='Command-line interface for my_package')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--simulations', help='Path to directory with simulated data.')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--low-memory', action='store_true', default=False, 
            help="Reads training datasets into memory as needed during training rather than all at once. Slows training due to increased file reads.")

    parser.add_argument('--cnn', action=_key_value_action(NETWORK_SETTINGS), nargs="?", help='Train CNN classifier of SFS. Optionally specify "epochs=XX:batch_size=YY:learning_rate=ZZ".')
    parser.add_argument('--fcnn', action=_key_value_action(NETWORK_SETTINGS), nargs="?", help='Train FCNN classifier of SFS. Optionally specify "epochs=XX:batch_size=YY:learning_rate=ZZ".')
    parser.add_argument('--cnnnpy', action=_key_value_action(NETWORK_SETTINGS), nargs="?", help='Train CNN classifier of alignments. Optionally specify "epochs=XX:batch_size=YY:learning_rate=ZZ".')
    parser.add_argument('--rf', action=_key_value_action(RF_SETTINGS), nargs="?", help='Train RF classifier of SFS. Optionally specify "ntrees=XX".')

    args = parser.parse_args()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory, or use --force.")
    # create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Parse the configuration file
    config_parser = parse_input.ModelConfigParser(args.config)
    config_values = config_parser.parse_config()

    if args.rf: # Random Forest
        random_forest_sfs_predictor = RandomForestsSFS(config_values, args.simulations)
        random_forest_sfs_model, random_forest_sfs_cm, random_forest_sfs_cm_plot = random_forest_sfs_predictor.build_rf_sfs(ntrees=args.rf['ntrees'])
        with open(os.path.join(args.output, 'rf.model.pickle'), 'wb') as f:
            pickle.dump(random_forest_sfs_model, f)
        random_forest_sfs_cm_plot.savefig(os.path.join(args.output, 'rf_confusion.png'))
        random_forest_sfs_cm_plot.close()
        names = [str(number) for number in random_forest_sfs_predictor.int_to_label.values()]
        write_confusion_tsv(random_forest_sfs_cm, names, os.path.join(args.output, 'rf_confusion.tsv'))

    if args.fcnn: # Fully conncted neural network with multidimensional SFS 
        keras.utils.set_random_seed(config_values["seed"])   # same starting weights every run
        data = PopaiTrainingData(args.simulations, "simulated_mSFS_*.pickle",
                config_values["seed"], args.low_memory, batch_size=args.fcnn['batch_size'])
        model = NeuralNetSFS(data.dataset.n_classes)
        train_model(model, data, args.output, "fcnn", epochs=args.fcnn['epochs'], batch_size=args.fcnn['batch_size'], learning_rate = args.fcnn['learning_rate'])
        test_model(model, data, args.output, "fcnn")

    if args.cnn: # CNN with 2D SFS
        keras.utils.set_random_seed(config_values["seed"])   # same starting weights every run
        data = PopaiTrainingData(args.simulations, "simulated_2dSFS_*.pickle",
                config_values["seed"], args.low_memory, batch_size=args.cnn['batch_size'], method='cnn')
        n_pairs = data.dataset[0][0].shape[0]
        model = CnnSFS(n_pairs, data.dataset.n_classes)
        train_model(model, data, args.output, "cnn", epochs=args.cnn['epochs'], batch_size=args.cnn['batch_size'], learning_rate = args.cnn['learning_rate'])
        test_model(model, data, args.output, "cnn")

    if args.cnnnpy: # CNN with SNP alignment
        keras.utils.set_random_seed(config_values["seed"])   # same starting weights every run
        data = PopaiTrainingData(args.simulations, "simulated_arrays_*.pickle",
                config_values["seed"], args.low_memory, batch_size=args.cnnnpy['batch_size'])
        n_sites = data.dataset[0][0].shape[1]
        model = CnnNpy(n_sites, config_values["sampling dict"], data.dataset.n_classes)
        train_model(model, data, args.output, "cnn_npy", epochs=args.cnnnpy['epochs'], batch_size=args.cnnnpy['batch_size'], learning_rate = args.cnnnpy['learning_rate'])
        test_model(model, data, args.output, "cnn_npy")

if __name__ == '__main__':
    main()