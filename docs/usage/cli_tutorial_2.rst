##############################
Tutorial 2: Multiple population trees.
##############################

If there is uncertainty regarding relationships between populations, the user can include multiple species trees in their species tree input file, as seen `here <https://github.com/SmithLabBio/popai/tree/main/tutorial_data/tutorial_2_data/species.nex>`_.

If the user does this, they will also need to provide paths to a migration file for each species tree, as seen in this `config file <https://github.com/SmithLabBio/popai/tree/main/tutorial_data/tutorial_2_data/config.txt>`_.

==========================================
Step 1: Prepare Input
==========================================

Example input data is available `here <https://github.com/SmithLabBio/popai/tree/main/tutorial_data/tutorial_2_data>`_.

You should have these in the directory you cloned when installing popai, in the subfolder tutorial_data.

Create a directory in which to run the tutorial, and copy these data to that directory::

    mkdir tutorial_2
    cd tutorial_2
    cp -r /path/to/downloaded/data/tutorial_2_data ./

==========================================
Step 2: Processing Empirical data
==========================================

The first step is to process the user's empirical data. This involves reading the data from fasta files (or a vcf), deciding which values to use for down-projection, and building the SFS.

For more information on input formats, please see the `input instructions <https://popai.readthedocs.io/en/latest/usage/parsinginput.html>`_.

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

First, we use the preview tool to decide what thresholds to use for downsampling. 

SFS cannot be generated from datasets that include missing data. To circumvent this, we use a downsampling approach such as that described in `Satler and Carstens (2017) <https://doi.org/10.1111/mec.14137>`_. We must choose thresholds for each population (i.e., the minumum number of individuals that must be sampled for a SNP to be used.)

We will use a folded SFS, meaning that we will build the SFS based on minor allele frequencies.

Since our data should be phased, and we will simulate diploid individuals, we will only consider multiples of 2.

Preview the SFS:


.. code-block:: python

    process_empirical_data --config tutorial_2_data/config.txt --preview --output preview

This will create a file called 'preview_SFS.txt' in the output directory with information about how many SNPs are available at different downsampling thresholds. We want to maximize the number of (haploid) individuals and SNPs we can use. In this tutorial, since the input data were simulated without missing data, we can use all individuals and still retain all the SNPs. 

Note that this prints the thresholds as the minimum number of chromosomes. We simulate diploid individuals in msprime, so popai requires that you use even values. This is why only even-valued thresholds are printed.

Now, we are ready to build our empirical SFS:

.. code-block:: python

    process_empirical_data --config tutorial_2_data/config.txt --downsampling "{'A':20, 'B':20, 'C':20}" --reps 1 --output empirical/

We are only using a single replicate for this test. This makes sense because our 'empirical' data are actually simulated data, and we are not downsampling. Because of this, we do not expect much noise. For messier empirical data, use ~10 reps and ensure that results do not differ across replicates.

Notice that this will print to the screen the number of SNPs in your empirical data. Please record this, as we will use it in the next step.

This script will also output in the output directory the joint and multidimensional site frequency spectra for each replicate.

==========================================
Step 3: Simulate data
==========================================

Next, we need to simulate data under the models of interest. We will do so using the command line tool *simulate_data*. It takes the following arugments::

    usage: simulate_data [-h] [--config CONFIG] [--plot] [--downsampling DOWNSAMPLING] [--nbins NBINS] [--output OUTPUT] [--force] [--maxsites MAXSITES] [--cores CORES]

    Command-line interface for my_package

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --plot                Plot the popai models.
      --simulate            Simulate data under the popai models.
      --downsampling DOWNSAMPLING
                            Input downsampling dict as literal string (e.g., {'A': 10, 'B': 10, 'C': 5} to downsample to 10 individuals in populations A and B and 5 in population C).
      --nbins NBINS         Number of bins for creating a binned SFS (default: None)
      --output OUTPUT       Path to output folder for storing SFS.
      --force               Overwrite existing results.
      --maxsites MAXSITES   Max number of sites to use when building SFS from simulated
      --cores CORES         Number of cores to use when simulating data.

The parameter maxsites should be set equal to the number of sites used to build the empirical SFS (which printed to the screen when you ran the *process_empirical_data* command.)

It is essential to use the same downsampling dictionary here that you used to process your empirical data.


.. code-block:: python

    simulate_data --config tutorial_2_data/config.txt --downsampling "{'A':20, 'B':20, 'C':20}" --output simulated/ --maxsites 1598 --plot --simulate

In the output directory, you should see a pdf showing your models (models.pdf), a pickled object storing the simulated jSFS, and a numpy matrix storing the mSFS. 

==========================================
Step 4: Train networks
==========================================

Now, we are ready to train the networks implemented in popai. popai includes three network architectures:
    1. A Random Forest classifier that takes as input the bins of the multidimensional SFS (mSFS).
    2. A Fully Connected Neural Network that takes as input the bins of the multidimensional SFS (mSFS).
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

    train_models --config tutorial_2_data/config.txt --simulations simulated/ --output trained_models --rf --fcnn --cnn --cnnnpy

This will output to the output directory the trained.model files for the FCNN and the CNN, and a pickled object storing the RF Classifier. It will also output confusion matrices showing the performance of each approach on the validation data, for which we hold out 20% of our simulated datasets. 

==========================================
Step 5: Apply networks
==========================================

Finally, we can apply the networks to make classifications on our empirical data using the function *apply_models*. It takes the following arguments::

    usage: apply_models [-h] [--config CONFIG] [--models MODELS] [--empirical EMPIRICAL] [--output OUTPUT] [--force] [--rf] [--fcnn] [--cnn]

    Command-line interface for my_package

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --models MODELS       Path to directory with trained models.
      --empirical EMPIRICAL
                            Path to directory with empirical SFS.
      --simulations         Path to simulated training data.
      --output OUTPUT       Path to output folder for storing SFS.
      --force               Overwrite existing results.
      --rf                  Train RF classifier.
      --fcnn                Train FCNN classifier.
      --cnn                 Train CNN classifier on jSFS
      --cnnnpy              Train a CNN classifier on alignments.

Provide the output paths from Step 5 and Step 3 for the --models and --empirical arguments, respectively. 

.. code-block:: python

    apply_models --config tutorial_2_data/config.txt --models trained_models/  --output results/ --empirical empirical/ --rf --fcnn --cnn --cnnnpy --simulations simulated/

This should save to the output directory tables showing the predicted probabilities for each model for each classifier.