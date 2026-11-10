import argparse
import os
import pickle
import numpy as np
from popai.build_predictors import predict


from popai import parse_input, build_predictors

def main():
    parser = argparse.ArgumentParser(description='Apply trained predictors to the empirical data.')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--models', help='Path to the output folder of train_models.')
    parser.add_argument('--empirical', help='Path to the output folder of process_empirical_data.')
    parser.add_argument('--output', help="Path to output folder for the predictions.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--rf', action='store_true', help='Apply RF classifier.')
    parser.add_argument('--fcnn', action='store_true', help='Apply FCNN classifier.')
    parser.add_argument('--cnn', action='store_true', help='Apply CNN classifier on jSFS.')
    parser.add_argument('--cnnnpy', action='store_true', help='Apply CNN classifier on alignments.')
    parser.add_argument('--simulations', help='Path to the output folder of simulate_data (the predictors were trained on these; used to name the models).')
    args = parser.parse_args()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory, or use --force.")
    # create output directory
    os.makedirs(args.output, exist_ok=True)



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
    populations = list(config_values["sampling dict"].keys())
    for replicate in range(len(msfs)):
        current_dict = []
        for i, pop1 in enumerate(populations):
            for pop2 in populations[i + 1:]:
                # the file process_empirical_data wrote for this pair and replicate
                with open(os.path.join(args.empirical, f"{pop1}_{pop2}_rep{replicate}.jsfs"), 'r') as f:
                    next(f)
                    next(f)
                    data = [[float(value) for value in line.strip().split()[1:]] for line in f]
                current_dict.append(np.array(data))
        jsfs.append(current_dict)
    empirical_array = np.load(os.path.join(args.empirical,"empirical.npy"))


    if args.rf:
        # apply Random Forest model
        random_forest_sfs_predictor = build_predictors.RandomForestsSFS(config_values, args.simulations)
        with open(os.path.join(args.models, 'rf.model.pickle'), 'rb') as f:
            random_forest_sfs_model = pickle.load(f)
        results_rf = random_forest_sfs_predictor.predict(random_forest_sfs_model, msfs)
        with open(os.path.join(args.output, 'rf_predictions.txt'), 'w') as f:
            f.write(results_rf)

    if args.fcnn:
        predict(args.models, "fcnn.keras", np.array(msfs), args.output, "fcnn", args.simulations, "simulated_mSFS_*.pickle", 'fcnn_features.npy', 'fcnn_labels.npy')

    if args.cnn:
        predict(args.models, "cnn.keras", jsfs, args.output, "cnn", args.simulations, "simulated_2dSFS_*.pickle", 'cnn_features.npy', 'cnn_labels.npy')


    if args.cnnnpy:
        empirical_array = np.expand_dims(empirical_array, axis=0)
        predict(args.models, "cnn_npy.keras", empirical_array, args.output, "cnn_npy", args.simulations, "simulated_arrays_*.pickle", 'cnn_npy_features.npy', 'cnn_npy_labels.npy')
 


if __name__ == '__main__':
    main()