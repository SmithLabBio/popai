import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input, build_predictors
from popai.dataset import PopaiTrainingData
from popai.build_predictors import CnnSFS, CnnNpy, NeuralNetSFS, RandomForestsSFS, train_model, test_model
import json

class ParseEpochsBatchSize(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Default epochs and batch_size
        epochs = 10
        batch_size = 10
        learning_rate = 0.001
        
        if values:
            try:
                # Split the string by ':' and extract key-value pairs
                params = dict(param.split('=') for param in values.split(':'))
                epochs = int(params.get('epochs', epochs))
                batch_size = int(params.get('batch_size', batch_size))
                learning_rate = float(params.get('learning_rate', learning_rate))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid format for {option_string}. Expected 'epochs=XX:batch_size=YY'.")
        
        # Set the parsed values
        setattr(namespace, self.dest, {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate})

class ParseTrees(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Default epochs and batch_size
        ntrees=500
        
        if values:
            try:
                # Split the string by ':' and extract key-value pairs
                params = dict(param.split('=') for param in values.split(':'))
                ntrees = int(params.get('ntrees', ntrees))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid format for {option_string}. Expected 'trees=XX'.")
        
        # Set the parsed values
        setattr(namespace, self.dest, {'ntrees': ntrees})

def main():
    parser = argparse.ArgumentParser(description='Command-line interface for my_package')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--simulations', help='Path to directory with simulated data.')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--low-memory', action='store_true', default=False, 
            help="Reads training datasets into memory as needed during training rather than all at once. Slows training due to increased file reads.")


    parser.add_argument('--cnn', action=ParseEpochsBatchSize, nargs="?", help='Train CNN classifier of SFS. Optionally specify "epochs=XX:batch_size=YY:learning_rate=ZZ".')
    parser.add_argument('--fcnn', action=ParseEpochsBatchSize, nargs="?", help='Train FCNN classifier of SFS. Optionally specify "epochs=XX:batch_size=YY:learning_rate=ZZ".')
    parser.add_argument('--cnnnpy', action=ParseEpochsBatchSize, nargs="?", help='Train CNN classifier of alignments. Optionally specify "epochs=XX:batch_size=YY:learning_rate=ZZ".')
    parser.add_argument('--rf', action=ParseTrees, nargs="?", help='Train RF classifier of SFS. Optionally specify "ntrees=XX".')

    args = parser.parse_args()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory, or use --force.")
    # create output directory
    os.system('mkdir -p %s' % args.output)

    # Parse the configuration file
    config_parser = parse_input.ModelConfigParser(args.config)
    config_values = config_parser.parse_config()

    # set whether user
    if config_values['user models'] is None:
        user = False
    else:
        user = True

    if args.rf: # Random Forest
        random_forest_sfs_predictor = RandomForestsSFS(config_values, args.simulations, user=user)
        random_forest_sfs_model, random_forest_sfs_cm, random_forest_sfs_cm_plot = random_forest_sfs_predictor.build_rf_sfs(ntrees=args.rf['ntrees'])
        with open(os.path.join(args.output, 'rf.model.pickle'), 'wb') as f:
            pickle.dump(random_forest_sfs_model, f)
        random_forest_sfs_cm_plot.savefig(os.path.join(args.output, 'rf_confusion.png'))

    if args.fcnn: # Fully conncted neural network with multidimensional SFS 
        data = PopaiTrainingData(args.simulations, "simulated_mSFS_*.pickle",
                config_values["seed"], args.low_memory, batch_size=args.fcnn['batch_size'])
        model = NeuralNetSFS(data.dataset.n_classes)
        train_model(model, data, args.output, "fcnn", epochs=args.fcnn['epochs'], batch_size=args.fcnn['batch_size'], learning_rate = args.fcnn['learning_rate'])
        test_model(model, data, args.output, "fcnn")

    if args.cnn: # CNN with 2D SFS
        data = PopaiTrainingData(args.simulations, "simulated_2dSFS_*.pickle",
                config_values["seed"], args.low_memory, batch_size=args.cnn['batch_size'], method='cnn')
        n_pairs = data.dataset[0][0].shape[0]
        model = CnnSFS(n_pairs, data.dataset.n_classes)
        train_model(model, data, args.output, "cnn", epochs=args.cnn['epochs'], batch_size=args.cnn['batch_size'], learning_rate = args.cnn['learning_rate'])
        test_model(model, data, args.output, "cnn")

    if args.cnnnpy: # CNN with SNP alignment
        data = PopaiTrainingData(args.simulations, "simulated_arrays_*.pickle",
                config_values["seed"], args.low_memory, batch_size=args.cnnnpy['batch_size'])
        n_sites = data.dataset[0][0].shape[1]
        model = CnnNpy(n_sites, config_values["sampling dict"], data.dataset.n_classes)
        train_model(model, data, args.output, "cnn_npy", epochs=args.cnnnpy['epochs'], batch_size=args.cnnnpy['batch_size'], learning_rate = args.cnnnpy['learning_rate'])
        test_model(model, data, args.output, "cnn_npy")

if __name__ == '__main__':
    main()