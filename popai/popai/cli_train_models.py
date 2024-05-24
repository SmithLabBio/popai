import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input, build_predictors

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
    parser.add_argument('--ntrees', type=int, help='Number of trees to use in the RF classifier (default=500).', default=500)

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

    # read the data
    labels = np.load(os.path.join(args.simulations, 'labels.npy'), allow_pickle=True)
    
    with open(os.path.join(args.simulations, 'simulated_msfs.pickle'), 'rb') as f:
        msfs = pickle.load(f)
    with open(os.path.join(args.simulations, 'simulated_jsfs.pickle'), 'rb') as f:
        sfs_2d = pickle.load(f)
    with open(os.path.join(args.simulations, 'simulated_arrays.pickle'), 'rb') as f:
        array = pickle.load(f)

    if args.rf:
        # train RF and save model and confusion matrix
        random_forest_sfs_predictor = build_predictors.RandomForestsSFS(config_values, msfs, labels, user=user)
        random_forest_sfs_model, random_forest_sfs_cm, random_forest_sfs_cm_plot = random_forest_sfs_predictor.build_rf_sfs(ntrees=args.ntrees)
        with open(os.path.join(args.output, 'rf.model.pickle'), 'wb') as f:
            pickle.dump(random_forest_sfs_model, f)
        random_forest_sfs_cm_plot.savefig(os.path.join(args.output, 'rf_confusion.png'))

    if args.fcnn:
        # train FCNN and save model and confusion matrix
        neural_network_sfs_predictor = build_predictors.NeuralNetSFS(config_values, msfs, labels, user = user)
        neural_network_sfs_model, neural_network_sfs_cm, neural_network_sfs_cm_plot = neural_network_sfs_predictor.build_neuralnet_sfs()
        neural_network_sfs_model.save(os.path.join(args.output, 'fcnn.keras'))
        neural_network_sfs_cm_plot.savefig(os.path.join(args.output, 'fcnn_confusion.png'))

    if args.cnn:
        # train CNN and save model and confusion matrix
        cnn_2d_sfs_predictor = build_predictors.CnnSFS(config_values, sfs_2d, labels, user=user)
        cnn_2d_sfs_model, cnn_2d_sfs_cm, cnn_2d_sfs_cm_plot = cnn_2d_sfs_predictor.build_cnn_sfs()
        cnn_2d_sfs_model.save(os.path.join(args.output, 'cnn.keras'))
        cnn_2d_sfs_cm_plot.savefig(os.path.join(args.output, 'cnn_confusion.png'))

    if args.cnnnpy:
        # train CNN and save model and confusion matrix
        cnn_2d_npy_predictor = build_predictors.CnnNpy(config_values, array, labels, user=user)
        cnn_2d_npy_model, cnn_2d_npy_cm, cnn_2d_npy_cm_plot = cnn_2d_npy_predictor.build_cnn_npy()
        cnn_2d_npy_model.save(os.path.join(args.output, 'cnn_npy.keras'))
        cnn_2d_npy_cm_plot.savefig(os.path.join(args.output, 'cnn_npy_confusion.png'))


if __name__ == '__main__':
    main()