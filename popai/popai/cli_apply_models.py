import argparse
import ast
import os
import pickle
import numpy as np
from keras import models
from popai import parse_input, build_predictors

def main():
    parser = argparse.ArgumentParser(description='Command-line interface for my_package')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--models', help='Path to directory with trained models.')
    parser.add_argument('--empirical', help='Path to directory with empirical SFS.')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--rf', action='store_true', help='Apply RF classifier.')
    parser.add_argument('--fcnn', action='store_true', help='Apply FCNN classifier.')
    parser.add_argument('--cnn', action='store_true', help='Apply CNN classifier on jSFS.')
    parser.add_argument('--cnnnpy', action='store_true', help='Apply CNN classifier on alignments.')
    parser.add_argument('--simulations', help='Path to directory with simulated data.')
    parser.add_argument('--downsampling', help="Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).")

    args = parser.parse_args()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory, or use --force.")
    # create output directory
    os.system('mkdir -p %s' % args.output)

    try:
        downsampling_dict = ast.literal_eval(args.downsampling)
    except ValueError:
        print('Error: Invalid downsampling dictionary. Please provide a valid dictionary string.')


    # Parse the configuration file
    config_parser = parse_input.ModelConfigParser(args.config)
    config_values = config_parser.parse_config()

    # read empirical data into correct format
    msfs_files = os.listdir(args.empirical)
    msfs_files = [x for x in msfs_files if x.endswith('_DSFS.obs')]
    msfs_files = sorted(msfs_files, key=lambda x: int(x.split('_')[0][3:]))
    msfs = []
    for file in msfs_files:
        with open(os.path.join(args.empirical, file), 'r') as f:
            lines = f.readlines()
            this_sfs = lines[2].split(' ')
            this_sfs = [int(x) for x in this_sfs]
        msfs.append(this_sfs)
    
    jsfs = []
    jsfs_files = os.listdir(args.empirical)
    jsfs_files = [x for x in jsfs_files if x.endswith('.jsfs')]
    populations = list(config_values["sampling dict"].keys())

    for i in range(len(msfs)):
        current_dict = {}
        current_files = [x for x in jsfs_files if f"rep{i}" in x]
        for i, pop1 in enumerate(populations):
            for j, pop2 in enumerate(populations):
                if i < j:
                    target = [x for x in current_files if f"{pop1}_{pop2}_rep" in x]
                    with open(os.path.join(args.empirical, target[0]), 'r') as f:
                        next(f)
                        next(f)
                        data = []
                        for line in f:
                            values = line.strip().split()[1:]
                            data.append([float(value) for value in values])
                        array = np.array(data)
                    current_dict[(pop1,pop2)] = array
        jsfs.append(current_dict)

    empirical_array = np.load(os.path.join(args.empirical,"empirical.npy"))

    # read the training data
    training_labels = np.load(os.path.join(args.simulations, 'labels.npy'), allow_pickle=True)
    
    with open(os.path.join(args.simulations, 'simulated_msfs.pickle'), 'rb') as f:
        training_msfs = pickle.load(f)
    with open(os.path.join(args.simulations, 'simulated_jsfs.pickle'), 'rb') as f:
        training_sfs_2d = pickle.load(f)
    with open(os.path.join(args.simulations, 'simulated_arrays.pickle'), 'rb') as f:
        training_array = pickle.load(f)


    if args.rf:
        # apply Random Forest model
        random_forest_sfs_predictor = build_predictors.RandomForestsSFS(config_values, {}, {})
        with open(os.path.join(args.models, 'rf.model.pickle'), 'rb') as f:
            random_forest_sfs_model = pickle.load(f)
        results_rf = random_forest_sfs_predictor.predict(random_forest_sfs_model, msfs)
        with open(os.path.join(args.output, 'rf_predictions.txt'), 'w') as f:
            f.write(results_rf)

    if args.fcnn:
        # apply FCNN model
        neural_network_sfs_predictor = build_predictors.NeuralNetSFS(config_values, {}, {})
        neural_network_sfs_model = models.load_model(os.path.join(args.models, 'fcnn.keras'), compile=True)
        results_fcnn = neural_network_sfs_predictor.predict(neural_network_sfs_model, msfs)
        with open(os.path.join(args.output, 'fcnn_predictions.txt'), 'w') as f:
            f.write(results_fcnn)

    if args.cnn:

         # apply FCNN model
        cnn_2d_sfs_predictor = build_predictors.CnnSFS(config_values, training_sfs_2d, training_labels)
        cnn_2d_sfs_model = models.load_model(os.path.join(args.models, 'cnn.keras'), compile=True)
        cnn_2d_sfs_featureextracter = models.load_model(os.path.join(args.models, 'cnn_featureextractor.keras'), compile=True)
        results_cnn = cnn_2d_sfs_predictor.predict(cnn_2d_sfs_model, jsfs)
        with open(os.path.join(args.output, 'cnn_predictions.txt'), 'w') as f:
            f.write(results_cnn)

        cnn_2d_sfs_predictor.check_fit(cnn_2d_sfs_featureextracter, jsfs, args.output, training_labels)


    if args.cnnnpy:
        # apply FCNN model
        cnn_npy_predictor = build_predictors.CnnNpy(config_values, {}, downsampling_dict, '')
        cnn_npy_model = models.load_model(os.path.join(args.models, 'cnn_npy.keras'), compile=True)
        results_cnn_npy = cnn_npy_predictor.predict(cnn_npy_model, empirical_array)
        with open(os.path.join(args.output, 'cnn_npy_predictions.txt'), 'w') as f:
            f.write(results_cnn_npy)


if __name__ == '__main__':
    main()