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

Now, we are ready to train the predictors implemented in popai. popai includes four:

1. A Random Forest classifier that takes as input the multidimensional SFS (mSFS) (``--rf``).
2. A Fully Connected Neural Network that takes as input the mSFS (``--fcnn``).
3. A Convolutional Neural Network that takes as input the jSFS between all pairs of populations (``--cnn``).
4. A Convolutional Neural Network that takes as input the alignments (``--cnnnpy``). This one needs ``--maxsites`` in Step 4, so that every simulated dataset is the same size.

To train them, we will use the command-line tool *train_models*. It takes the following arguments::

    usage: train_models [-h] [--config CONFIG] [--simulations SIMULATIONS] [--output OUTPUT] [--force]
                        [--low-memory] [--cnn [CNN]] [--fcnn [FCNN]] [--cnnnpy [CNNNPY]] [--rf [RF]]

    Train predictors on the simulated data.

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --simulations SIMULATIONS
                            Path to the output folder of simulate_data.
      --output OUTPUT       Path to output folder for the trained predictors.
      --force               Overwrite existing results.
      --low-memory          Read the simulated datasets from disk as they are needed, instead of
                            reading them all into memory at the start. Uses less memory, but trains
                            more slowly.
      --cnn [CNN]           Train CNN classifier of SFS. Optionally specify
                            "epochs=XX:batch_size=YY:learning_rate=ZZ" (defaults: 10, 10, 0.001).
      --fcnn [FCNN]         Train FCNN classifier of SFS. Optionally specify
                            "epochs=XX:batch_size=YY:learning_rate=ZZ" (defaults: 10, 10, 0.001).
      --cnnnpy [CNNNPY]     Train CNN classifier of alignments. Optionally specify
                            "epochs=XX:batch_size=YY:learning_rate=ZZ" (defaults: 10, 10, 0.001).
      --rf [RF]             Train RF classifier of SFS. Optionally specify "ntrees=XX" (default: 500).

Notes on the arguments:

* ``--simulations`` is the output directory from Step 4.
* Train any of the four predictors by giving its option. Each option can also take settings, separated by ``:``, in any order. For example, ``--fcnn "epochs=20:batch_size=32"`` trains the FCNN for 20 epochs, with batches of 32 datasets, and the default learning rate. A setting popai doesn't recognize stops *train_models* with an error.
* ``--low-memory`` reads each simulated dataset from disk when it is needed, instead of reading them all into memory at the start. Use it if the simulations don't fit in memory. Training is slower.

Train all four predictors with their default settings:

.. code-block:: python

    train_models --config tutorial_1_data/config.txt --simulations simulated/ --output trained_models/ --rf --fcnn --cnn --cnnnpy

Each predictor is trained on 80% of the simulated datasets, and tested on the other 20%, with each model equally represented in both. *train_models* prints the out-of-bag error of the Random Forest, and the loss and accuracy of the networks after each epoch, on the training data and on the 20% held out (val_loss and val_accuracy). Training uses the seed in the configuration file, so the same configuration file and simulations give the same trained predictors.

In the output directory, you should see:

* rf.model.pickle, fcnn.keras, cnn.keras and cnn_npy.keras: the trained predictors.
* <predictor>_confusion.png and <predictor>_confusion.tsv: confusion matrices for each predictor on the 20% of simulated datasets held out, with one row for each true model and one column for each predicted model.
* <network>_features.npy and <network>_labels.npy: for each network, the features of the training datasets (the values in the layer before the output layer) and their models. *apply_models* uses these in Step 6.

==========================================
Step 6: Apply networks
==========================================

Finally, we can apply the predictors to our empirical data using the command-line tool *apply_models*. It takes the following arguments::

    usage: apply_models [-h] [--config CONFIG] [--models MODELS] [--empirical EMPIRICAL]
                        [--output OUTPUT] [--force] [--rf] [--fcnn] [--cnn] [--cnnnpy]
                        [--simulations SIMULATIONS]

    Apply trained predictors to the empirical data.

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to config file.
      --models MODELS       Path to the output folder of train_models.
      --empirical EMPIRICAL
                            Path to the output folder of process_empirical_data.
      --output OUTPUT       Path to output folder for the predictions.
      --force               Overwrite existing results.
      --rf                  Apply RF classifier.
      --fcnn                Apply FCNN classifier.
      --cnn                 Apply CNN classifier on jSFS.
      --cnnnpy              Apply CNN classifier on alignments.
      --simulations SIMULATIONS
                            Path to the output folder of simulate_data (the predictors were trained on
                            these; used to name the models).

Give the output directories from Step 5, Step 3 and Step 4 for ``--models``, ``--empirical`` and ``--simulations``, and the predictors you trained in Step 5:

.. code-block:: python

    apply_models --config tutorial_1_data/config.txt --models trained_models/ --empirical empirical/ --simulations simulated/ --output results/ --rf --fcnn --cnn --cnnnpy

In the output directory, you should see:

* <predictor>_predictions.txt: tables of the predicted probability of each model. The SFS predictors (rf, fcnn and cnn) give one row for each replicate SFS from Step 3 (``--reps``), and cnn_npy gives one row for the alignments.
* <network>_empirical_features.npy: the features of the empirical data, as in Step 5.
* <network>_pca.png: a principal component analysis of the features of the simulated datasets, colored by model, with the empirical data marked with red crosses. If the empirical data fall far from all of the simulated datasets, none of the models may fit the data well.
