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

The command line tool *process_empirical_data* can be used to process empirical data. It takes the following arguments::

    usage: process_empirical_data [-h] [--config CONFIG] [--preview] [--downsampling DOWNSAMPLING] [--reps REPS] [--nbins NBINS] [--output OUTPUT] [--force]

    Command-line interface for processing empirical data.

    options:
        -h, --help            show this help message and exit
        --config CONFIG       Path to config file.
        --preview             Preview number of SNPs used for different down-projections
        --downsampling DOWNSAMPLING
                              Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).
        --reps REPS           Number of replicate downsampled SFS to build.
        --nbins NBINS         Number of bins for creating a binned SFS (default: None)
        --output OUTPUT       Path to output folder for storing SFS
        --force               Overwrite existing results.

First, we use the preview tool to decide what thresholds to use for downsampling:

.. code-block:: python

    process_empirical data -config config.txt --preview

This will print to the screen information about how many SNPs are available at different downsampling thresholds. We want to maximize the number of individuals and SNPs we can use. In this tutorial, since the input data were simulated without missing data, we can use all individuals and still retain all the SNPs.

Now, we are ready to build our empirical SFS:

.. code-block:: python

    process_empirical_data --config private/config_local.txt --downsampling "{'A':20, 'B':20, 'C':20}" --reps 1 --output private/test_cli/empirical

We are only using a single replicate for this test. This makes sense because our 'empirical' data are actually simulated data, and we are not downsampling. Because of this, we do not expect much noise. For messier empirical data, use ~10 reps and ensure that results do not differ across replicates.

Notice that this will print to the screen the number of sites used to build the SFS on average. Please record this, as we will use it in the next step.

This script will also output in the output directory the joint and multidimensional site frequency spectra for each replicate.

==========================================
Step 2: Simulate data
==========================================

Next, we need to simulate data under the models of interest. We will do so using the command line tool *simulate_data*. It takes the folloiwng arugments::

    usage: simulate_data [-h] [--config CONFIG] [--plot] [--downsampling DOWNSAMPLING] [--nbins NBINS] [--output OUTPUT] [--force] [--maxsites MAXSITES] [--cores CORES]

    Command-line interface for my_package

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --plot                Plot the delimitpy models.
      --downsampling DOWNSAMPLING
                            Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).
      --nbins NBINS         Number of bins for creating a binned SFS (default: None)
      --output OUTPUT       Path to output folder for storing SFS.
      --force               Overwrite existing results.
      --maxsites MAXSITES   Max number of sites to use when building SFS from simulated
      --cores CORES         Number of cores to use when simulating data.

The parameter maxsites should be set equal to the number of sites used to build the empirical SFS (which printed to the screen when you ran the simulate_data command.)

It is essential to use the same downsampling dictionary here that you used to process your empirical data.


.. code-block:: python

    simulate_data --config private/config_local.txt --downsampling "{'A':20, 'B':20, 'C':20}" --output private/test_cli/simulated --maxsites 1009 --plot

In the output directory, you should see a pdf showing your models (models.pdf), a pickled object storing the simulated jSFS, and a numpy matrix storing the mSFS. 

==========================================
Step 3: Train networks
==========================================

    Now, we are ready to train the networks implemented in delimitpy. delimitpy includes three network architectures:
        1. a Random Forest classifier that takes as input the bins of the multidimensional SFS (mSFS).
        2. a Fully Connected Neural Network that takes as input the bins of the multidimensional SFS (mSFS).
        3. A Convolutional Neural Network that takes as input the jSFS between all pairs of populations.

To train networks, we will use the command-line tool *train_models*. It takes the following arguments::

    usage: train_models [-h] [--config CONFIG] [--simulations SIMULATIONS] [--output OUTPUT] [--force] [--rf] [--fcnn] [--cnn]

    Command-line interface for my_package

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --simulations SIMULATIONS
                            Path to directory with simulated data.
      --output OUTPUT       Path to output folder for storing SFS.
      --force               Overwrite existing results.
      --rf                  Train RF classifier.
      --fcnn                Train FCNN classifier.
      --cnn                 Train CNN classifier.

The argument *--simulations* takes as input the output directory from the previous step.

.. code-block:: python
    train_models --config private/config_local.txt --simulations private/test_cli/simulated --output private/test_cli/trained_models --rf --fcnn --cnn

This will output to the output directory the trained.model files for the FCNN and the CNN, and a pickled object storing the RF Classifier. It will also output confusion matrices showing the performance of each approach on the validation data, for which we hold out 20% of our simulated datasets. 

==========================================
Step 4: Apply networks
==========================================

Finally, we can apply the networks to make classifications on our empirical data using the function *apply_models*. It has the following parameters::

    usage: apply_models [-h] [--config CONFIG] [--models MODELS] [--empirical EMPIRICAL] [--output OUTPUT] [--force] [--rf] [--fcnn] [--cnn]

    Command-line interface for my_package

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --models MODELS       Path to directory with trained models.
      --empirical EMPIRICAL
                            Path to directory with empirical SFS.
      --output OUTPUT       Path to output folder for storing SFS.
      --force               Overwrite existing results.
      --rf                  Train RF classifier.
      --fcnn                Train FCNN classifier.
      --cnn                 Train CNN classifier.

Provide the output paths from Step 3 and Step 1 for the --models and --empirical arguments, respectively. 

.. code-block:: python
    apply_models --config private/config_local.txt --models private/test_cli/trained_models  --output private/test_cli/results --empirical private/test_cli/empirical --rf --fcnn --cnn

This should save to the output directory tables showing the predicted probabilities for each model for each classifier.