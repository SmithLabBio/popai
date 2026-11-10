import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input, generate_models, simulate_data, process_user_models

def main():
    parser = argparse.ArgumentParser(description='Simulate data under the popai models, and build their SFS.')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--plot', action='store_true', help='Plot the popai models (models.pdf in the output folder).')
    parser.add_argument('--simulate', action='store_true', help='Simulate data under the popai models.')
    parser.add_argument('--downsampling', help="Downsampling dict as a literal string, giving the number of haplotypes to use from each population (e.g., {'A': 10, 'B': 10, 'C': 6}). Use the same values as when processing the empirical data. Each must be even.")
    parser.add_argument('--nbins', type=int, default=None, help='Number of bins for creating a binned mSFS. Use the same value as when processing the empirical data (default: None, no binning).')
    parser.add_argument('--output', help="Path to output folder for storing the simulated data and SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--maxsites', type=int, help="Number of SNPs to keep from each simulated dataset: the first maxsites SNPs, padded with missing data if a dataset has fewer. Use the number of SNPs in the empirical data. Needed for the CNN on arrays, which needs every dataset to be the same size (default: keep all SNPs).")
    parser.add_argument('--cores', type=int, default=1, help="Number of cores to use when simulating data.")
    parser.add_argument('--checkpoint', action='store_true', help='Keep models whose simulations already exist in the output folder, and only simulate the rest.')
    parser.add_argument('--infinite', action='store_true', help='Simulate one SNP from each of a number of unlinked loci (the number of loci in the empirical data), under an infinite sites model.')
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
            data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values, cores=args.cores, downsampling=downsampling_dict, max_sites = args.maxsites, output=args.output, sp_tree_index=sp_tree_index, checkpoint=args.checkpoint, infinite=args.infinite)
            arrays = data_simulator.simulate_ancestry_parallel()

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
            data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values, cores=args.cores, downsampling=downsampling_dict, max_sites = args.maxsites, user=True, output=args.output, checkpoint=args.checkpoint, infinite=args.infinite)
            arrays = data_simulator.simulate_ancestry_parallel()

    
    if args.simulate:

        # build SFS for simulate data
        sfs_2d = data_simulator.mutations_to_2d_sfs()
        msfs = data_simulator.mutations_to_sfs(arrays, nbins=args.nbins)

        data_simulator.plot_2dsfs(sfs_2d,output_directory=args.output)

if __name__ == '__main__':
    main()