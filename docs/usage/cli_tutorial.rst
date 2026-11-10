##############################
Tutorial 1a: Running popai through the command line
##############################

==========================================
Step 1: Install popai
==========================================

Use pip to install popai::

    git clone https://github.com/SmithLabBio/popai.git
    cd popai/popai
    pip install .
    cd ../../

==========================================
Step 2: Prepare Input
==========================================

Example input data is available `here <https://github.com/SmithLabBio/popai/tree/main/tutorial_data/tutorial_1_data>`_.

You should have these in the directory you cloned when installing popai, in the subfolder tutorial_data.

Create a directory in which to run the tutorial, and copy these data to that directory::

    mkdir tutorial_1
    cd tutorial_1
    cp -r /path/to/downloaded/data/tutorial_1_data ./

==========================================
Step 3: Processing Empirical data
==========================================

The first step is to process the user's empirical data. This involves reading the data from fasta files (or a vcf), deciding which values to use for down-projection, and building the SFS.

For more information on input formats, please see the `input instructions <https://popai.readthedocs.io/en/latest/usage/parsinginput.html>`_.

The command line tool *process_empirical_data* can be used to process empirical data. It takes the following arguments::

    usage: process_empirical_data [-h] [--config CONFIG] [--preview] [--downsampling DOWNSAMPLING]
                                  [--reps REPS] [--nbins NBINS] [--output OUTPUT] [--force]
                                  [--maxsites MAXSITES]

    Command-line interface for processing empirical data.

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
        --preview             Preview number of SNPs used for different down-projections
      --downsampling DOWNSAMPLING
                            Downsampling dict as a literal string, giving the number of haplotypes to
                            use from each population (e.g., {'A': 10, 'B': 10, 'C': 6}). Each must be
                            even.
      --reps REPS           Number of replicate downsampled SFS to build.
      --nbins NBINS         Number of bins for creating a binned mSFS (default: None, no binning).
      --output OUTPUT       Path to output folder for storing SFS.
      --force               Overwrite existing results.
      --maxsites MAXSITES   Number of SNPs to keep from the empirical data: the first maxsites SNPs
                            (default: keep all SNPs).

First, we use the preview tool to decide what thresholds to use for downsampling. 

SFS cannot be generated from datasets that include missing data. To circumvent this, we use a downsampling approach such as that described in `Satler and Carstens (2017) <https://doi.org/10.1111/mec.14137>`_. We must choose thresholds for each population (i.e., the minimum number of haplotypes that must be sampled for a SNP to be used.)

We will use a folded SFS, meaning that we will build the SFS based on minor allele frequencies.

Since our data should be phased, and we will simulate diploid individuals, we will only consider multiples of 2.

Preview the SFS:


.. code-block:: python

    process_empirical_data --config tutorial_1_data/config.txt --preview --output preview

This will create a file called 'preview_SFS.txt' in the output directory with information about how many SNPs are available at different downsampling thresholds. We want to maximize the number of (haploid) individuals and SNPs we can use. In this tutorial, since the input data were simulated without missing data, we can use all individuals and still retain all the SNPs. 

Note that this prints the thresholds as the minimum number of chromosomes. We simulate diploid individuals in msprime, so popai requires that you use even values. This is why only even-valued thresholds are printed.

Now, we are ready to build our empirical SFS:

.. code-block:: python

    process_empirical_data --config tutorial_1_data/config.txt --downsampling "{'A':20, 'B':20, 'C':20}" --reps 1 --output empirical/

We are only using a single replicate for this test. This makes sense because our 'empirical' data are actually simulated data, and we are not downsampling. Because of this, we do not expect much noise. For messier empirical data, use ~10 reps and ensure that results do not differ across replicates.

Notice that this will print to the screen the number of SNPs in your empirical data. Please record this, as we will use it in the next step.

This script will also output in the output directory the joint site frequency spectra between each pair of populations (.jsfs files) and the multidimensional site frequency spectrum (_DSFS.obs) for each replicate, plots of the joint site frequency spectra (.png files), and the empirical alignments as an array (empirical.npy) for the CNN in Steps 5 and 6.

==========================================
Step 4: Simulate data
==========================================

Next, we need to simulate data under the models of interest. We will do so using the command line tool *simulate_data*. It takes the following arguments::

    usage: simulate_data [-h] [--config CONFIG] [--plot] [--simulate] [--downsampling DOWNSAMPLING]
                         [--nbins NBINS] [--output OUTPUT] [--force] [--maxsites MAXSITES]
                         [--cores CORES] [--checkpoint] [--infinite]

    Simulate data under the popai models, and build their SFS.

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --plot                Plot the popai models (models.pdf in the output folder).
      --simulate            Simulate data under the popai models.
      --downsampling DOWNSAMPLING
                            Downsampling dict as a literal string, giving the number of haplotypes to
                            use from each population (e.g., {'A': 10, 'B': 10, 'C': 6}). Use the same
                            values as when processing the empirical data. Each must be even.
      --nbins NBINS         Number of bins for creating a binned mSFS. Use the same value as when
                            processing the empirical data (default: None, no binning).
      --output OUTPUT       Path to output folder for storing the simulated data and SFS.
      --force               Overwrite existing results.
      --maxsites MAXSITES   Number of SNPs to keep from each simulated dataset: the first maxsites
                            SNPs, padded with missing data if a dataset has fewer. Use the number of
                            SNPs in the empirical data. Needed for the CNN on arrays, which needs
                            every dataset to be the same size (default: keep all SNPs).
      --cores CORES         Number of cores to use when simulating data.
      --checkpoint          Keep models whose simulations already exist in the output folder, and only
                            simulate the rest.
      --infinite            Simulate one SNP from each of a number of unlinked loci (the number of
                            loci in the empirical data), under an infinite sites model.

Notes on the arguments:

* ``--downsampling`` must be the same dictionary you used to process your empirical data. The values are numbers of haplotypes, and each must be even.
* If you used ``--nbins`` to process your empirical data, use the same value here.
* ``--maxsites`` should be the number of SNPs in your empirical data, which *process_empirical_data* printed to the screen (1608 in this tutorial). Each simulated dataset keeps its first maxsites SNPs, and a dataset with fewer SNPs is padded with missing data, so every simulated dataset is the same size as the empirical one. The CNN that is trained on the alignments (``--cnnnpy`` in Steps 5 and 6) needs this.
* ``--cores`` sets how many simulations run at once. The results are the same for any number of cores.
* ``--checkpoint`` is for restarting a run that stopped partway. Models whose simulations are already in the output folder are kept, and only the remaining models are simulated. Because the output folder already exists, use it together with ``--force``.
* ``--infinite`` is for data with one SNP from each of many unlinked loci (for example, one SNP per RAD locus). Instead of simulating whole alignments, popai simulates one SNP per locus, with as many loci as in your empirical data.

Run the simulations:

.. code-block:: python

    simulate_data --config tutorial_1_data/config.txt --downsampling "{'A':20, 'B':20, 'C':20}" --output simulated/ --maxsites 1608 --plot --simulate

For each replicate of each model, popai draws the model parameters and a mutation rate from the priors in the configuration file. It then simulates one locus for each empirical alignment, at that alignment's length. Loci are unlinked, and there is no recombination within a locus. Each population is sampled with as many haplotypes as it has in the pop file, as diploid individuals.

For each model, *simulate_data* prints the median number of SNPs in its simulated datasets. If this is very different from the number of SNPs in your empirical data, you may want to change the priors (for example, the mutation rate).

In the output directory, you should see:

* models.pdf: a plot of each model (from ``--plot``). The model numbers in the other file names match the numbers in this plot.
* simulated_arrays_<model>.pickle: the simulated alignments for each replicate of the model, with one row per haplotype and one column per SNP. Rows are in pop-file order, and are sorted within each population.
* simulated_mSFS_<model>.pickle: the mSFS for each replicate of the model.
* simulated_2dSFS_<model>.pickle: the jSFS between each pair of populations, for each replicate of the model.
* 2D_SFS_<pair>_<model>.png: plots of the average jSFS between each pair of populations under each model.

==========================================
Step 5: Train networks
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

    train_models --config tutorial_1_data/config.txt --simulations simulated/ --output trained_models --rf --fcnn --cnn --cnnnpy --downsampling "{'A':20, 'B':20, 'C':20}"

This will output to the output directory the trained.model files for the FCNN and the CNN, and a pickled object storing the RF Classifier. It will also output confusion matrices showing the performance of each approach on the validation data, for which we hold out 20% of our simulated datasets. 

==========================================
Step 6: Apply networks
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

    apply_models --config tutorial_1_data/config.txt --models trained_models/  --output results/ --empirical empirical/ --rf --fcnn --cnn --cnnnpy --simulations simulated/ --downsampling "{'A':20, 'B':20, 'C':20}"

This should save to the output directory tables showing the predicted probabilities for each model for each classifier.
