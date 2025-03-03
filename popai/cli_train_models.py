import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input, build_predictors
from popai.dataset import PopaiTrainingData
from popai.build_predictors import CnnSFS, CnnNpy, NeuralNetSFS, RandomForestsSFS, train_model, test_model
import json



def main():
    parser = argparse.ArgumentParser(description='Command-line interface for my_package')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--simulations', help='Path to directory with simulated data.')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--rf', action='store_true', help='Train RF classifier.')
    parser.add_argument('--fcnn', action='store_true', help='Train FCNN classifier.')
    parser.add_argument('--cnn', action='store_true', help='Train CNN classifier of SFS.')
    parser.add_argument('--cnnnpy', action='store_true', help='Train CNN classifier on alignments.')
    parser.add_argument('--ntrees', type=int, default=500, help='Number of trees to use in the RF classifier (default=500).')
    parser.add_argument('--downsampling', help="Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).")
    parser.add_argument('--subset', default=None, help="Path to a file listing the models to retain. List indices only (e.g., 0, 1, 5, 6). One integer per line")
    parser.add_argument('--low-memory', action='store_true', default=False, 
            help="Reads training datasets into memory as needed during training rather than all at once. Slows training due to increased file reads.")

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

    try:
        downsampling_dict = ast.literal_eval(args.downsampling)
    except ValueError:
        print('Error: Invalid downsampling dictionary. Please provide a valid dictionary string.')

    if args.rf: # Random Forest
        random_forest_sfs_predictor = RandomForestsSFS(config_values, args.simulations, 
                                                       subset=args.subset, user=user)
        random_forest_sfs_model, random_forest_sfs_cm, random_forest_sfs_cm_plot = random_forest_sfs_predictor.build_rf_sfs(ntrees=args.ntrees)
        with open(os.path.join(args.output, 'rf.model.pickle'), 'wb') as f:
            pickle.dump(random_forest_sfs_model, f)
        random_forest_sfs_cm_plot.savefig(os.path.join(args.output, 'rf_confusion.png'))

    if args.fcnn: # Fully conncted neural network with multidimensional SFS 
        data = PopaiTrainingData(args.simulations, "simulated_mSFS_*.pickle",
                config_values["seed"], args.low_memory)
        model = NeuralNetSFS(data.dataset.n_classes)
        train_model(model, data, args.output, "fcnn")
        test_model(model, data, args.output, "fcnn")

    if args.cnn: # CNN with 2D SFS
        data = PopaiTrainingData(args.simulations, "simulated_2dSFS_*.pickle",
                config_values["seed"], args.low_memory)
        n_pairs = len(data.dataset[0][0].keys())
        model = CnnSFS(n_pairs, data.dataset.n_classes)
        train_model(model, data, args.output, "cnn")
        test_model(model, data, args.output, "cnn")

    if args.cnnnpy: # CNN with SNP alignment
        data = PopaiTrainingData(args.simulations, "simulated_arrays_*.pickle",
                config_values["seed"], args.low_memory)
        n_sites = data.dataset[0][0].shape[1]
        model = CnnNpy(n_sites, downsampling_dict, data.dataset.n_classes)
        train_model(model, data, args.output, "cnn_npy")
        test_model(model, data, args.output, "cnn_npy")

if __name__ == '__main__':
    main()