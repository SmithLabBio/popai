import argparse
import ast
import os
import pickle
import numpy as np
from popai import parse_input, process_empirical

def main():
    parser = argparse.ArgumentParser(description='Command-line interface for processing empirical data.')
    parser.add_argument('--config', help='Path to config file.')
    parser.add_argument('--preview', action='store_true', help='Preview number of SNPs used for different down-projections')
    parser.add_argument('--downsampling', help="Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).")
    parser.add_argument('--reps', type=int, help="Number of replicate downsampled SFS to build.")
    parser.add_argument('--nbins', type=int, default=None, help='Number of bins for creating a binned SFS (default: None)')
    parser.add_argument('--output', help="Path to output folder for storing SFS.")
    parser.add_argument('--force', action='store_true', help='Overwrite existing results.')

    args = parser.parse_args()
    
    # Parse the configuration file
    config_parser = parse_input.ModelConfigParser(args.config)
    config_values = config_parser.parse_config()

    # Process empirical data
    data_processor = process_empirical.DataProcessor(config=config_values)
    if "fastas" in config_values:
        empirical_array = data_processor.fasta_to_numpy()
    else:
        empirical_array = data_processor.vcf_to_numpy()

    # check if output exists
    if os.path.exists(args.output) and not args.force:
        raise RuntimeError(f"Error: output directory, {args.output} already exists. Please specify a different directory, or use --force.")
    # create output directory
    os.system('mkdir -p %s' % args.output)
       
    # If we are checking downsampling, print downsampling results
    if args.preview:
        empirical_2d_sfs_sampling = data_processor.find_downsampling(empirical_array)
        with open(os.path.join(args.output, 'preview_SFS.txt'), 'w') as f:
            f.write(f"Population order: {config_values['sampling dict']}\n")
            for key,value in empirical_2d_sfs_sampling.items():
                f.write(f"{str(key)}\t{str(value)}")
                f.write('\n')
        print(f"Preview printed to file {os.path.join(args.output, 'preview_SFS.txt')}.")
    
    else:
        try:
            downsampling_dict = ast.literal_eval(args.downsampling)
        except ValueError:
            print('Error: Invalid downsampling dictionary. Please provide a valid dictionary string.')
            return
    
        empirical_2d_sfs = data_processor.numpy_to_2d_sfs(empirical_array, downsampling=downsampling_dict, replicates = args.reps)
        empirical_msfs, average_snps = data_processor.numpy_to_msfs(empirical_array, downsampling=downsampling_dict, replicates = args.reps, nbins=args.nbins)

        # save numpy array
        np.save(file=os.path.join(args.output, 'empirical.npy'), arr=empirical_array)

        # make plots of empirical sfs
        data_processor.plot_2dsfs(empirical_2d_sfs, output_directory=os.path.join(args.output))

        # save to output directory
        for replicate in range(args.reps):

            with open(os.path.join(args.output, f"rep{replicate}_DSFS.obs"), 'w') as f:
                f.write("1 observations. No. of demes and sample sizes are on next line\n")
                f.write(str(len(list(downsampling_dict.keys()))))
                f.write(" ")
                for key in config_values['sampling dict']:
                    f.write(f"{downsampling_dict[key]}")
                    f.write(" ")
                f.write("\n")
                f.write(' '.join(map(str, list(empirical_msfs[replicate]))))

            for key, value in empirical_2d_sfs[replicate].items():
                name = f"{key[0]}_{key[1]}_rep{replicate}.jsfs"
                with open(os.path.join(args.output, name), 'w') as f:
                    f.write("1 observations\n")
                    f.write(" ")
                    f.write(' '.join([f'd_{i}' for i in range(value.shape[1])]) + '\n')
                    for i in range(value.shape[0]):
                        f.write(f'd_{i} ' + ' '.join(map(str, value[i])) + '\n')                    

if __name__ == '__main__':
    main()