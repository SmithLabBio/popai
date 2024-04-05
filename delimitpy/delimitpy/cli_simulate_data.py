import argparse
import ast
import os
import pickle
import numpy as np
from delimitpy import parse_input, generate_models, simulate_data

def main():
    parser = argparse.ArgumentParser(description='Command-line interface for my_package')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--plot', action='store_true', help='Plot the delimitpy models.')
    parser.add_argument('--downsampling', help="Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).")
    #parser.add_argument('-r', '--reps', type=int, help="Number of replicate downsampled SFS to build.")
    parser.add_argument('--nbins', type=int, default=None, help='Number of bins for creating a binned SFSsimu (default: None)')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--maxsites', type=int, help="Max number of sites to use when building SFS from simulated")
    parser.add_argument('--cores', type=int, default=1, help="Number of cores to use when simulating data.")

    args = parser.parse_args()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory.")
    # create output directory
    os.system('mkdir -p %s' % args.output)

    
    # Parse the configuration file
    config_parser = parse_input.ModelConfigParser(args.config)
    config_values = config_parser.parse_config()

    # Build models and draw parameters
    model_builder = generate_models.ModelBuilder(config_values)
    divergence_demographies, sc_demographies, dwg_demographies = model_builder.build_models()
    parameterized_models, labels = model_builder.draw_parameters(divergence_demographies, sc_demographies, dwg_demographies)

    if args.plot:
        
        # validate the models
        model_builder.validate_models(parameterized_models, labels, outplot=os.path.join(args.output, 'models.pdf'))


    # get dict for downsampling
    try:
        downsampling_dict = ast.literal_eval(args.downsampling)
    except ValueError:
        print('Error: Invalid downsampling dictionary. Please provide a valid dictionary string.')
        return

    # simulate data
    data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values, cores=args.cores, downsampling=downsampling_dict, max_sites = args.maxsites)
    arrays = data_simulator.simulate_ancestry()

    # build SFS for simulate data
    sfs_2d = data_simulator.mutations_to_2d_sfs(arrays)
    msfs = data_simulator.mutations_to_sfs(arrays)

    # save these simulated data.
    with open(os.path.join(args.output, 'simulated_jsfs.pickle'), 'wb') as f:
        pickle.dump(sfs_2d, f)
    np.save(os.path.join(args.output, 'simulated_msfs.npy'), np.array(msfs), allow_pickle=True)
    np.save(os.path.join(args.output, 'labels.npy'), np.array(labels), allow_pickle=True)


if __name__ == '__main__':
    main()