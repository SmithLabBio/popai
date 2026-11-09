##############################
Input Data
##############################

Below, we provide information on the required input data. As an example, we use the data in tutorial_data/tutorial_1_data. popai requires the inputs listed below. We will discuss each of these in turn.

* **Configuration File**
* Empirical Data

  * Alignments or a VCF
  * Pop file

* Model Building Information

  * Species Tree
  * Migration Matrix

popai checks these inputs when it reads the configuration file, and stops with a message saying what is wrong if something can't be used.

========================================
Configuration File
========================================

The user must provide popai with a `configuration file <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/tutorial_1_data/config.txt>`_::

    [Model]
    species tree file = ./tutorial_1_data/tree.nex # Path to a species tree in nexus format.
    migration matrix = ./tutorial_1_data/migration.txt # Path to a migration matrix
    symmetric = True # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = True # True if you wish to consider secondary contact models.
    divergence with gene flow = False # True if you wish to consider divergence with gene flow models.
    max migration events = 1 # Maximum number of migration events to consider in one model.
    migration rate = U(1e-5, 1e-4) # Prior from which to draw migration rates. Only uniform priors are supported at present.
    constant Ne = True # population sizes equal across all populations
    user models = None

    [Other]
    seed = 1234 # Random seed.
    replicates = 1000 # Number of replicates to simulate per model.

    [Simulations]
    mutation rate = U(5e-9, 5e-8) # Prior from which to draw mutation rates. Only uniform priors are supported at present.
    substitution model = JC69 # Substitution model to use in simulations.

    [Data]
    alignments = ./tutorial_1_data/alignments # Path to alignments
    popfile = ./tutorial_1_data/populations.txt # Path to popfile

Notes on the settings:

* Paths are read relative to the folder you run popai from, not the folder the configuration file is in.
* Anything after a ``#`` on a line is a comment.
* A setting that is not used can be set to None (in any case) or left out. ``user models`` is only needed for `user-specified models <https://popai.readthedocs.io/en/latest/usage/usermodels.html>`_; when it is given, the other settings in ``[Model]`` are not used.
* ``symmetric``, ``secondary contact``, ``divergence with gene flow`` and ``constant Ne`` are True or False.
* ``max migration events`` is a whole number, 0 or more.
* ``seed`` is a whole number, 0 or more, and ``replicates`` is a whole number, 1 or more.
* ``mutation rate`` and ``migration rate`` are uniform priors, written ``U(min, max)`` with the smaller value first, e.g. ``U(5e-9, 5e-8)``. They can't be negative.
* The migration rate can instead be given as a number of migrants per generation (Nm, the size of the population receiving migrants times the migration rate), e.g. ``U(0.5Nm, 2Nm)``. Both values must be Nm values, or both must be rates.
* ``substitution model`` is JC69 or binary.
* With ``constant Ne = True``, every population in the species tree must have the same ne prior.

========================================
Empirical Data
========================================

------------
Alignments
------------

The user will provide a path to a folder of alignments in the configuration file (see above). These alignments must be in fasta format, and there must be one alignment per locus. The files must end with .fa, .fas, or .fasta; other files in the folder are ignored. The alignments are read in alphabetical order of their file names.

An `example set of alignments <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/tutorial_1_data/alignments>`_ is provided on GitHub.

------------
VCF
------------

As an alternative to a set of gene alignments, the user can provide a vcf. In the config file, set alignments = None (or leave it out), and add vcf = /path/to/input.vcf::

    [Data]
    alignments = None
    vcf = ./data/input.vcf
    popfile = ./data/populations_vcf.txt

Samples in the VCF must be diploid.

Each contig with data in the VCF is treated as one locus. Contigs that are listed in the VCF header but have no data are ignored. Contig lengths are read from the VCF header's contig lines, e.g.::

    ##contig=<ID=1,length=1000>

If your VCF does not contain information on contig lengths, you must supply the expected contig length via a "length" argument in the config file's Data section::

    length = 1000

Contigs with a length in the header use that length, and the rest use ``length``. If a contig has no length either way, popai stops and lists the contigs without one.

------------
Pop File
------------

The user will provide a path to a file assigning individuals to populations in the config file (see above). This file must be tab-separated, with a header row naming the columns 'individual' and 'population', and then one individual and its population on each line::

    individual	population
    n0	A
    n1	A
    n2	B

An `example population file <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/tutorial_1_data/populations.txt>`_ is provided on GitHub.

If you use fasta files, then "individuals" should correspond to haplotypes. If you use a VCF, "individuals" should correspond to individuals.

Notes:

* Spaces around names are ignored.
* Every individual in the pop file must be in the alignments or VCF, and every sample in the alignments or VCF must be in the pop file.
* Populations are used in the order they first appear in the pop file. This is the order of populations in popai's site frequency spectra and arrays.

========================================
Model Building Information
========================================

------------
Species Tree
------------

The user must provide a path to a nexus file with a species tree. There are some specific requirements for the `species tree file <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/tutorial_1_data/tree.nex>`_::

    #NEXUS
    BEGIN TAXA;
          Dimensions NTax=3;
          TaxLabels A B C;
    END;

    BEGIN TREES;
          Tree species=((A[&ne=10000-50000],B[&ne=10000-50000])AB[&ne=10000-50000,div=10000-50000],C[&ne=10000-50000])ABC[&ne=10000-50000,div=100000-500000];
    END;

Requirements:

* Internal nodes must be labeled with names.
* For each leaf and internal node, include an annotation indicating the minimum and maximum values of the uniform distribution on the effective population size for the corresponding population::

    [&ne=10000-50000]

* For each internal node, include an annotation indicating the minimum and maximum values of the uniform distribution on the divergence time (in generations before the present)::

    [&div=10000-50000]

* The minimum and maximum are whole numbers, with the smaller value first (10000-50000, not 1e4-5e4).

The file can contain more than one species tree. Each tree then needs its own migration matrix (see below).

----------------
Migration Matrix
----------------

The user must provide a path to a file with a `migration matrix <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/tutorial_1_data/migration.txt>`_ indicating whether migration is allowed between all pairs of lineages::

    ,A,B,C,AB,ABC
    A,F,F,T,F,F
    B,F,F,F,F,F
    C,T,F,F,T,F
    AB,F,F,T,F,F
    ABC,F,F,F,F,F

Note that T indicates that migration is allowed between two taxa, while F indicates that migration is not allowed (TRUE and FALSE also work). The elements along the diagonal will be ignored.

The rows and columns must name the same populations as the species tree, so ancestral populations must be included. Every entry must be T or F.

If the species tree file contains more than one tree, give one migration matrix per tree, in the same order, separated by ``;``::

    migration matrix = ./tutorial_2_data/migration.txt;./tutorial_2_data/migration2.txt