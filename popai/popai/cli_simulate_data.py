import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input, generate_models, simulate_data, process_user_models

def main():
    parser = argparse.ArgumentParser(description='Command-line interface for my_package')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--plot', action='store_true', help='Plot the popai models.')
    parser.add_argument('--simulate', action='store_true', help='Simulate data under the popai models.')
    parser.add_argument('--downsampling', help="Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).")
    parser.add_argument('--nbins', type=int, default=None, help='Number of bins for creating a binned SFSsimu (default: None)')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--maxsites', type=int, help="Max number of sites to use when building SFS from simulated")
    parser.add_argument('--cores', type=int, default=1, help="Number of cores to use when simulating data.")

    args = parser.parse_args()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory, or use --force.")
    # create output directory
    os.system('mkdir -p %s' % args.output)

    
    # Parse the configuration file
    config_parser = parse_input.ModelConfigParser(args.config)
    config_values = config_parser.parse_config()

    if config_values['user models'] is None:

        # Build models and draw parameters
        model_builder = generate_models.ModelBuilder(config_values)
        divergence_demographies, sc_demographies, dwg_demographies = model_builder.build_models()
        parameterized_models, labels, sp_tree_index = model_builder.draw_parameters(divergence_demographies, sc_demographies, dwg_demographies)

        if args.plot:

            # validate the models
            model_builder.validate_models(parameterized_models, labels, outplot=os.path.join(args.output, 'models.pdf'))

        if args.simulate:
            # get dict for downsampling
            try:
                downsampling_dict = ast.literal_eval(args.downsampling)
            except ValueError:
                print('Error: Invalid downsampling dictionary. Please provide a valid dictionary string.')
                return

            # simulate data
            data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values, cores=args.cores, downsampling=downsampling_dict, max_sites = args.maxsites, sp_tree_index=sp_tree_index)
            arrays = data_simulator.simulate_ancestry()

    else:

        # Build models and draw parameters
        model_reader = process_user_models.ModelReader(config_values=config_values)
        parameterized_models, labels = model_reader.read_models()

        # validate the models
        if args.plot:
                model_reader.validate_models(parameterized_models, labels, outplot=os.path.join(args.output, 'models.pdf'))
        
        if args.simulate:
            # get dict for downsampling
            try:
                downsampling_dict = ast.literal_eval(args.downsampling)
            except ValueError:
                print('Error: Invalid downsampling dictionary. Please provide a valid dictionary string.')
                return

            # simulate data
            data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values, cores=args.cores, downsampling=downsampling_dict, max_sites = args.maxsites, user=True)
            arrays = data_simulator.simulate_ancestry()

    
    if args.simulate:

        # build SFS for simulate data
        sfs_2d = data_simulator.mutations_to_2d_sfs(arrays)
        msfs = data_simulator.mutations_to_sfs(arrays)

        data_simulator.plot_2dsfs(sfs_2d,output_directory=args.output)

        # save these simulated data.
        with open(os.path.join(args.output, 'simulated_jsfs.pickle'), 'wb') as f:
            pickle.dump(sfs_2d, f)
        with open(os.path.join(args.output, 'simulated_arrays.pickle'), 'wb') as f:
            pickle.dump(arrays, f)
        with open(os.path.join(args.output, 'simulated_msfs.pickle'), 'wb') as f:
            pickle.dump(msfs, f)
        np.save(os.path.join(args.output, 'labels.npy'), np.array(labels), allow_pickle=True)

if __name__ == '__main__':
    main()