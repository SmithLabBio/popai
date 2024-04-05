##############################
Running delimitpy through the command line
##############################

==========================================
Step 0: Downloading example input data
==========================================

Example input data is available `here <https://www.github.com/SmithLabBio/delimitpy/tutorial_data>`

==========================================
Step 1: Processing Empirical data
==========================================

The first step is to process the user's empirical data. This involves reading the data from fasta files, deciding which values to use for down-projection, and building the SFS.

For more information on input formats, please see the `input instructions <https://delimitpy.readthedocs.io/en/latest/usage/parsinginput.html>`.

The command line tool process_empirical_data can be used to process empirical data. It takes the following arguments::

    Command-line interface for processing empirical data.

    options:
        -h, --help            show this help message and exit
        --config CONFIG       Path to config file.
        --preview             Preview number of SNPs used for different down-projections
        --downsampling DOWNSAMPLING
                              Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).
        --reps REPS           Number of replicate downsampled SFS to build.
        --nbins NBINS         Number of bins (default: None)
        --output OUTPUT       Path to output folder for storing SF


First, we use the preview tool to decide what thresholds to use for downsampling.

.. code-block:: python

    process_empirical data -config config.txt --preview

    process_empirical_data --config private/config_local.txt --downsampling "{'A':20, 'B':20, 'C':20}" --reps 1 --output private/test_cli/empirical
    simulate_data --config private/config_local.txt --downsampling "{'A':20, 'B':20, 'C':20}" --output private/test_cli/simulated --maxsites 1009
    train_models --config private/config_local.txt --simulations private/test_cli/simulated --output private/test_cli/trained_models --rf --fcnn --cnn
    apply_models --config private/config_local.txt --models private/test_cli/trained_models  --output private/test_cli/results --empirical private/test_cli/empirical --rf --fcnn --cnn
